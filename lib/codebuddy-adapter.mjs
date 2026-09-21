/**
 * Native ctx.llm adapter for CodeBuddy. Ported from shatyuka/dsh-llm-codebuddy
 * (MIT). Owns the chat wire: CLI identity headers, SSE streaming, and message
 * serialization — the pieces the shared pi-ai route could not control.
 * Plain ESM, no dependencies, Node >= 18.
 */

import {
  LlmAdapter,
  LlmError,
  ToolCallId as CallId,
  ReasoningEffortId,
  ProviderRequestId,
  contentHasImage,
  QUOTA_EXCEEDED_CODE,
  CONTEXT_WINDOW_EXCEEDED_CODE,
  isQuotaExceededError,
  isContextWindowExceededError,
  EMPTY_RESPONSE_CODE,
} from '@deepseek-ai/dsh-llm';

export const SSE_DONE = '[DONE]';
const CATALOG_TTL_MS = 5 * 60 * 1000;

/* ------------------------------------------------------------------ *
 * SSE framing (no eventsource-parser): reassemble chunks, join multi-
 * `data:` lines, stop at [DONE], throw STREAM_CLOSED on truncation.
 * ------------------------------------------------------------------ */

export async function* parseSse(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let dataLines = [];
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const rawLine = buffer.slice(0, nl).replace(/\r$/, '');
        buffer = buffer.slice(nl + 1);
        if (rawLine.length === 0 || rawLine.startsWith(':')) {
          // event boundary or comment: flush collected data lines
          if (dataLines.length > 0) {
            const payload = dataLines.join('\n');
            dataLines = [];
            yield payload;
            if (payload === SSE_DONE) return;
          }
        } else if (rawLine.startsWith('data:')) {
          dataLines.push(rawLine.slice(5).replace(/^ /, ''));
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (dataLines.length > 0 && dataLines.join('\n') === SSE_DONE) return;
  throw new LlmError('CodeBuddy SSE stream ended without [DONE]', 'STREAM_CLOSED');
}

/* ------------------------------------------------------------------ *
 * Request serialization: harness messages → OpenAI-compatible wire.
 * ------------------------------------------------------------------ */

function textOf(blocks) {
  return (blocks || []).filter((b) => b.type === 'text').map((b) => b.text).join('');
}

function serializeAssistant(message) {
  const content = message.content || [];
  const toolCalls = content
    .filter((b) => b.type === 'tool-call')
    .map((b) => ({ id: b.id, type: 'function', function: { name: b.name, arguments: b.arguments } }));
  const reasoning = content.filter((b) => b.type === 'reasoning').map((b) => b.text).join('');
  return {
    role: 'assistant',
    // content is always a string, never null: a reasoning-only or tool-call
    // turn sits in the durable log, and null content breaks later turns.
    content: textOf(content),
    // reasoning is replayed only on tool-call turns (thinking-mode passback).
    ...(toolCalls.length > 0 && reasoning.length > 0 ? { reasoning_content: reasoning } : {}),
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };
}

/** Tool results ride inside user messages in the harness but the wire wants
 *  one `role: 'tool'` entry per result. */
export function serializeMessages(messages, supportsImages) {
  const wire = [];
  for (const message of messages) {
    if (!supportsImages && contentHasImage(message.content || [])) {
      throw new LlmError('The selected CodeBuddy model does not accept image content.', 'UNSUPPORTED_CONTENT');
    }
    if (message.role === 'system') {
      wire.push({ role: 'system', content: textOf(message.content) });
      continue;
    }
    if (message.role === 'assistant') {
      wire.push(serializeAssistant(message));
      continue;
    }
    const results = (message.content || []).filter((b) => b.type === 'tool-result');
    const text = textOf(message.content);
    if (text.length > 0 || results.length === 0) wire.push({ role: 'user', content: text });
    for (const r of results) {
      wire.push({ role: 'tool', tool_call_id: r.toolCallId, content: textOf(r.content) || '(no output)' });
    }
  }
  return wire;
}

export function serializeRequest(options, supportsImages) {
  const messages = [];
  if (options.system !== undefined) messages.push({ role: 'system', content: options.system });
  messages.push(...serializeMessages(options.messages || [], supportsImages));
  const tools = options.tools && options.tools.length > 0
    ? options.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }))
    : undefined;
  return {
    model: options.model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    ...(tools ? { tools } : {}),
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
    ...(options.maxTokens === undefined ? {} : { max_tokens: options.maxTokens }),
    ...(options.stop === undefined ? {} : { stop: options.stop }),
    ...(options.reasoningEffort === undefined ? {} : { reasoning_effort: options.reasoningEffort }),
  };
}

/* ------------------------------------------------------------------ *
 * Response translation: SSE payloads → harness StreamChunk protocol.
 * ------------------------------------------------------------------ */

const EFFORT_NAMES = { low: 'Low', medium: 'Medium', high: 'High', xhigh: 'Extra high', max: 'Max' };

/** A property schema with no `type` accepts any JSON value. */
function acceptsAnyJsonValue(schema) {
  if (schema === true) return true;
  if (schema === null || typeof schema !== 'object' || Array.isArray(schema)) return false;
  return !('type' in schema) && !('$ref' in schema) && !('const' in schema) && !('enum' in schema)
    && !('allOf' in schema) && !('anyOf' in schema) && !('oneOf' in schema) && !('not' in schema);
}

function decodeNestedComposite(value) {
  if (typeof value !== 'string') return value;
  const t = value.trim();
  const composite = (t.startsWith('{') && t.endsWith('}')) || (t.startsWith('[') && t.endsWith(']'));
  if (!composite) return value;
  try {
    const decoded = JSON.parse(t);
    return decoded !== null && typeof decoded === 'object' ? decoded : value;
  } catch {
    return value;
  }
}

/** CodeBuddy sometimes double-encodes object/array values of unconstrained
 *  tool fields as JSON strings; decode exactly that shape. */
function normalizeToolArguments(name, argumentsText, tools) {
  const tool = (tools || []).find((t) => t.name === name);
  const properties = tool && tool.parameters ? tool.parameters.properties : undefined;
  if (properties === undefined) return argumentsText;
  let args;
  try {
    args = JSON.parse(argumentsText);
  } catch {
    return argumentsText;
  }
  if (args === null || typeof args !== 'object' || Array.isArray(args)) return argumentsText;
  let changed = false;
  for (const [key, schema] of Object.entries(properties)) {
    if (!acceptsAnyJsonValue(schema) || !(key in args)) continue;
    const decoded = decodeNestedComposite(args[key]);
    if (decoded !== args[key]) { args[key] = decoded; changed = true; }
  }
  return changed ? JSON.stringify(args) : argumentsText;
}

function mapFinishReason(reason) {
  switch (reason) {
    case 'stop': return { kind: 'stop' };
    case 'tool_calls': return { kind: 'tool-calls' };
    case 'length': return { kind: 'max-tokens' };
    default: return { kind: 'error', failure: { message: `model stopped: ${reason}`, code: reason.toUpperCase() } };
  }
}

function mapUsage(usage) {
  const cacheRead = (usage.prompt_tokens_details && usage.prompt_tokens_details.cached_tokens)
    ?? usage.prompt_cache_hit_tokens;
  const reasoning = usage.completion_tokens_details && usage.completion_tokens_details.reasoning_tokens;
  const prompt = usage.prompt_tokens ?? 0;
  return {
    inputTokens: Math.max(0, prompt - (cacheRead ?? 0)),
    outputTokens: usage.completion_tokens ?? 0,
    ...(cacheRead === undefined ? {} : { cacheReadTokens: cacheRead }),
    ...(reasoning === undefined ? {} : { reasoningTokens: reasoning }),
  };
}

function closeBlock(block, tools) {
  if (block.kind === 'text') return { type: 'text', text: block.text };
  if (block.kind === 'reasoning') return { type: 'reasoning', text: block.text };
  return {
    type: 'tool-call',
    id: CallId(block.callId ?? ''),
    name: block.name ?? '',
    arguments: normalizeToolArguments(block.name ?? '', block.text, tools),
  };
}

/** Deltas stream as they arrive; block ends, usage, and finish flush at
 *  [DONE] — usage strictly before finish, nothing after finish. */
export async function* translate(payloads, tools = []) {
  let nextIndex = 0;
  let textBlock;
  let reasoningBlock;
  const toolBlocks = new Map();
  const order = [];
  let pendingFinish;
  let pendingUsage;
  const open = (kind) => { const b = { index: nextIndex++, kind, text: '' }; order.push(b); return b; };

  for await (const payload of payloads) {
    if (payload === SSE_DONE) {
      for (const block of order) yield { type: 'block-end', index: block.index, block: closeBlock(block, tools) };
      if (pendingUsage !== undefined) yield { type: 'usage', usage: pendingUsage };
      const reason = pendingFinish ?? { kind: 'stop' };
      yield {
        type: 'finish',
        reason: reason.kind === 'stop' && order.length === 0
          ? { kind: 'error', failure: { message: 'model returned a completed response with no content', code: EMPTY_RESPONSE_CODE } }
          : reason,
      };
      return;
    }

    let chunk;
    try {
      chunk = JSON.parse(payload);
    } catch {
      throw new LlmError(`malformed CodeBuddy SSE payload: ${payload.slice(0, 120)}`, 'MALFORMED_RESPONSE');
    }

    for (const choice of chunk.choices ?? []) {
      const delta = choice.delta;
      // reasoning first: thinking models interleave it ahead of text; accept
      // both field spellings; an empty first delta opens nothing.
      const reasoning = (delta && (delta.reasoning_content ?? delta.reasoning)) || undefined;
      if (typeof reasoning === 'string' && reasoning.length > 0) {
        if (reasoningBlock === undefined) {
          reasoningBlock = open('reasoning');
          yield { type: 'block-start', index: reasoningBlock.index, blockType: 'reasoning' };
        }
        reasoningBlock.text += reasoning;
        yield { type: 'reasoning-delta', index: reasoningBlock.index, text: reasoning };
      }
      const content = delta && delta.content;
      if (typeof content === 'string' && content.length > 0) {
        if (textBlock === undefined) {
          textBlock = open('text');
          yield { type: 'block-start', index: textBlock.index, blockType: 'text' };
        }
        textBlock.text += content;
        yield { type: 'text-delta', index: textBlock.index, text: content };
      }
      for (const call of (delta && delta.tool_calls) || []) {
        let block = toolBlocks.get(call.index);
        if (block === undefined) {
          block = open('tool-call');
          toolBlocks.set(call.index, block);
          yield { type: 'block-start', index: block.index, blockType: 'tool-call' };
        }
        // only the opening delta carries id/name; later frames repeat "".
        if (call.id !== undefined && call.id.length > 0) block.callId = call.id;
        const name = call.function && call.function.name;
        if (name !== undefined && name.length > 0) block.name = name;
        const fragment = (call.function && call.function.arguments) || '';
        block.text += fragment;
        yield {
          type: 'tool-call-delta',
          index: block.index,
          id: CallId(block.callId ?? ''),
          ...(block.name === undefined ? {} : { name: block.name }),
          argumentsDelta: fragment,
        };
      }
      if (typeof choice.finish_reason === 'string') pendingFinish = mapFinishReason(choice.finish_reason);
    }
    // `usage: null` rides every non-final chunk — null-tolerant test.
    if (chunk.usage !== undefined && chunk.usage !== null) pendingUsage = mapUsage(chunk.usage);
  }

  throw new LlmError('CodeBuddy SSE payload stream ended without [DONE]', 'STREAM_CLOSED');
}

/* ------------------------------------------------------------------ *
 * HTTP status mapping + the adapter.
 * ------------------------------------------------------------------ */

function providerRetryAfterMs(value) {
  if (value === null) return undefined;
  if (/^\d+$/.test(value)) {
    const delay = Number(value) * 1000;
    return Number.isFinite(delay) && delay > 0 ? delay : undefined;
  }
  const delay = Date.parse(value) - Date.now();
  return Number.isFinite(delay) && delay > 0 ? delay : undefined;
}

function requestId(headers) {
  const value = headers.get('x-request-id') ?? headers.get('x-requestid');
  return value === null || value.length === 0 ? undefined : ProviderRequestId(value);
}

export function httpErrorCode(status, error) {
  if (status === 401 || status === 403) return 'AUTH';
  const detail = [error && error.code, error && error.type, error && error.message].filter(Boolean).join(' ');
  if (isQuotaExceededError(detail)) return QUOTA_EXCEEDED_CODE;
  if (status === 429) return 'RATE_LIMIT';
  if (status === 400) {
    if (isContextWindowExceededError(detail)) return CONTEXT_WINDOW_EXCEEDED_CODE;
    return 'INVALID_REQUEST';
  }
  if (status >= 500) return 'SERVER';
  return `HTTP_${status}`;
}

function modelInfo(provider, model) {
  return {
    provider,
    id: model.id,
    name: model.name || model.id,
    inputModalities: model.supportsImages === true ? ['text', 'image'] : ['text'],
  };
}

/** Disclosed thinking levels → harness reasoning metadata; undefined when the
 *  catalog offers nothing selectable (a rejected catalog is worse than an
 *  absent capability). Ids pass through: they are the wire `reasoning_effort`
 *  spellings. */
function reasoningInfo(model) {
  const supported = model.reasoning && model.reasoning.supportedEfforts;
  if (!Array.isArray(supported)) return undefined;
  const seen = new Set();
  const efforts = [];
  for (const raw of supported) {
    const id = typeof raw === 'string' ? raw.trim() : '';
    if (id.length === 0 || seen.has(id)) continue;
    seen.add(id);
    efforts.push({ id: ReasoningEffortId(id), name: EFFORT_NAMES[id] ?? id });
  }
  if (efforts.length === 0) return undefined;
  const candidate = model.reasoning && (model.reasoning.defaultEffort ?? model.reasoning.effort);
  const defaultEffort = candidate !== undefined && seen.has(candidate)
    ? ReasoningEffortId(candidate)
    : undefined;
  return { efforts, ...(defaultEffort === undefined ? {} : { defaultEffort }) };
}

/**
 * One adapter serves the `codebuddy` route.
 * @param {object} deps
 * @param {() => Promise<string|undefined>} deps.getAccessToken
 * @param {() => {chatBaseURL: string, domain: string, cliVersion: string}} deps.connection
 * @param {() => Promise<readonly object[]>} deps.readCatalog — tool-call-capable
 *   /v3/config entries; throws when logged out.
 * @param {(token: string) => Record<string, string>} [deps.identityFromToken]
 * @param {object} [deps.defaults] — fallback capacities for unlisted ids.
 */
export class CodebuddyAdapter extends LlmAdapter {
  constructor({ getAccessToken, connection, readCatalog, identityFromToken, defaults = {} }) {
    super();
    this.getAccessToken = getAccessToken;
    this.connection = connection;
    this.readCatalog = readCatalog;
    this.identityFromToken = identityFromToken;
    this.defaults = { contextWindow: defaults.contextWindow ?? 128000, maxTokens: defaults.maxTokens ?? 8192 };
    this.catalog = undefined;
    this.catalogRead = undefined;
  }

  providerInfo(provider) {
    return { id: provider, name: 'CodeBuddy' };
  }

  /** Cached catalog, shared between concurrent readers. */
  async models(signal) {
    if (this.catalog && Date.now() - this.catalog.readAt < CATALOG_TTL_MS) return this.catalog.models;
    this.catalogRead ??= (async () => {
      try {
        const models = await this.readCatalog(signal);
        this.catalog = { readAt: Date.now(), models };
      } finally {
        this.catalogRead = undefined;
      }
    })();
    return this.catalogRead.then(() => (this.catalog ? this.catalog.models : []));
  }

  refreshCatalog() {
    this.catalog = undefined;
  }

  async listModels(provider) {
    let models;
    try {
      models = await this.models();
    } catch {
      return []; // logged out or unreachable: offer nothing rather than fail
    }
    // drop entries whose capacity the catalog withholds; they stay routable.
    return models
      .filter((m) => m.maxInputTokens !== undefined && m.maxInputTokens > 0)
      .map((m) => modelInfo(provider, m));
  }

  async resolveModel(provider, model, signal) {
    let entry;
    try {
      entry = (await this.models(signal)).find((m) => m.id === model);
    } catch {
      entry = undefined;
    }
    if (entry === undefined) {
      // unlisted id is still routable — declare a conservative text-only shape.
      return {
        provider,
        id: model,
        name: model,
        inputModalities: ['text'],
        context: { contextWindow: this.defaults.contextWindow },
        defaultMaxTokens: this.defaults.maxTokens,
      };
    }
    const reasoning = reasoningInfo(entry);
    return {
      ...modelInfo(provider, entry),
      context: { contextWindow: entry.maxInputTokens > 0 ? entry.maxInputTokens : this.defaults.contextWindow },
      defaultMaxTokens: entry.maxOutputTokens > 0 ? entry.maxOutputTokens : this.defaults.maxTokens,
      ...(reasoning === undefined ? {} : { reasoning }),
    };
  }

  async *stream(options) {
    const connection = this.connection();
    const token = await this.getAccessToken();
    if (token === undefined) {
      throw new LlmError('CodeBuddy is not logged in — tell the agent "log in with codebuddy", then retry.', 'MISSING_CREDENTIAL');
    }

    const identity = this.identityFromToken ? this.identityFromToken(token) : {};
    const headers = {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      Authorization: `Bearer ${token}`,
      // the CLI identity the chat plane expects — the attribute the shared
      // pi-ai route cannot set (its User-Agent is attribution-reserved).
      'User-Agent': `CLI/${connection.cliVersion} CodeBuddy/${connection.cliVersion}`,
      'X-Domain': connection.domain,
      ...identity,
    };

    const models = await this.models().catch(() => []);
    const entry = models.find((m) => m.id === options.model);
    const supportsImages = entry ? entry.supportsImages === true : false;
    if ((options.tools ?? []).length > 0 && entry && entry.supportsToolCall === false) {
      throw new LlmError(`CodeBuddy model "${options.model}" does not support tool calls`, 'UNSUPPORTED_OPTION');
    }

    const payload = JSON.stringify(serializeRequest(options, supportsImages));
    let response;
    try {
      response = await fetch(`${connection.chatBaseURL}/chat/completions`, {
        method: 'POST',
        headers,
        body: payload,
        ...(options.signal === undefined ? {} : { signal: options.signal }),
      });
    } catch (error) {
      if (options.signal && options.signal.aborted) {
        throw new LlmError('CodeBuddy request aborted by caller', 'ABORTED', { cause: error });
      }
      throw new LlmError(`CodeBuddy request to ${connection.chatBaseURL} failed`, 'TRANSPORT', { cause: error });
    }

    if (!response.ok) {
      let message = `CodeBuddy API error (HTTP ${response.status})`;
      let providerError;
      try {
        const parsed = await response.json();
        providerError = parsed.error;
        if (providerError && providerError.message) message = providerError.message;
      } catch {
        // malformed error body: status still identifies the failure
      }
      const delay = providerRetryAfterMs(response.headers.get('retry-after'));
      const id = requestId(response.headers);
      throw new LlmError(message, httpErrorCode(response.status, providerError), {
        status: response.status,
        ...(delay === undefined ? {} : { providerRetryAfterMs: delay }),
        ...(id === undefined ? {} : { requestId: id }),
      });
    }

    if (response.body === null) throw new LlmError('CodeBuddy API returned no response body', 'EMPTY_RESPONSE');

    yield* translate(parseSse(response.body), options.tools ?? []);
  }
}
