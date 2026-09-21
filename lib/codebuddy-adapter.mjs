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
  offloadRequestImagesWithPolicy,
  offloadedImageText,
  resolveImageAttachmentAccess,
  QUOTA_EXCEEDED_CODE,
  CONTEXT_WINDOW_EXCEEDED_CODE,
  isQuotaExceededError,
  isContextWindowExceededError,
  EMPTY_RESPONSE_CODE,
} from '@deepseek-ai/dsh-llm';
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout';
/** Accept one streamed identity field for a tool call. `id` and `name` are
 *  identity, not accumulation: the wire sends each once, on the call's first
 *  delta. A continuation delta that re-sends the field empty — or `null`,
 *  which some OpenAI-compatible gateways fill in — means "no update", never
 *  "clear". @param current established value @param incoming parsed field
 *  @returns the identity in force after this delta. */
function acceptIdentity(current, incoming) {
  return typeof incoming === 'string' && incoming.length > 0 ? incoming : current;
}

/** Effort value for the wire, or undefined to omit the field entirely.
 *  CodeBuddy rejects the literal "off" with HTTP 400 `invalid_reasoning_effort`
 *  (verified against the live chat plane), and "off" means "do not send a
 *  thinking directive" — exactly what omitting the field does. "none" is
 *  accepted as an alias so a future catalog spelling cannot reintroduce the
 *  failure. */
function reasoningEffort(effort) {
  if (typeof effort !== 'string') return undefined;
  const normalized = effort.trim().toLowerCase();
  if (normalized.length === 0 || normalized === 'off' || normalized === 'none') return undefined;
  return effort;
}

/** Idle window for the SSE stream: no chunk for this long aborts the request
 *  instead of hanging the agent loop. Matches the official route's default. */
const CODEBUDDY_STREAM_IDLE_TIMEOUT_MS = 300000;
const CODEBUDDY_STREAM_IDLE_TIMEOUT_CODE = 'LLM_STREAM_IDLE_TIMEOUT';
export const SSE_DONE = '[DONE]';
const CATALOG_TTL_MS = 5 * 60 * 1000;
/** Inline base64 target per request image. CodeBuddy takes OpenAI-style
 *  `data:` URLs, so the encoded bytes ride inside the request body. */
const CODEBUDDY_IMAGE_MAX_BYTES = 4 * 1024 * 1024;
/** Aspect-preserving pixel ceiling for a projected request image. Mirrors the
 *  harness's own request-image budget (64e4) so CodeBuddy downscales by the
 *  same rule as the built-in routes. */
const CODEBUDDY_IMAGE_MAX_PIXELS = 640000;
/** Above these, the oldest images degrade to text placeholders instead of
 *  riding the request as base64. CodeBuddy has no Files API to fall back on,
 *  so the inline byte budget is the binding constraint; the values mirror the
 *  official DeepSeek route's defaults so both routes degrade alike. */
const CODEBUDDY_MAX_IMAGES_PER_REQUEST = 600;
const CODEBUDDY_MAX_REQUEST_IMAGE_BYTES = 64 * 1024 * 1024;
const CODEBUDDY_IMAGE_OFFLOAD_BYTE_QUANTUM = 64 * 1024 * 1024;
const CODEBUDDY_IMAGE_OFFLOAD_COUNT_QUANTUM = 20;
/** Prefix for the user message that carries images recovered from tool results. */
const TOOL_RESULT_IMAGE_TEXT = 'Attached image(s) from tool result:';

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
/** Text of every top-level text block — the plain-string projection a user
 *  message keeps when it carries no image. Tool results are serialized as their
 *  own `role: 'tool'` entries, so this layer stays deliberately shallow. */
function textOf(blocks) {
  return (blocks || []).filter((b) => b.type === 'text').map((b) => b.text).join('');
}

/** Flatten the text of every non-image block, including nested tool results.
 *  Assistant/system turns and the wire's `tool` entry cannot carry image parts,
 *  so they keep the plain-text form while user turns go multimodal. */
function flattenText(blocks) {
  const parts = [];
  for (const b of blocks || []) {
    if (b.type === 'text') parts.push(b.text);
    else if (b.type === 'tool-result') parts.push(flattenText(b.content));
  }
  return parts.join('');
}

/** Ordered image blocks, including the ones nested inside tool results. */
function imageBlocks(blocks, into = []) {
  for (const b of blocks || []) {
    if (b.type === 'image') into.push(b);
    else if (b.type === 'tool-result') imageBlocks(b.content, into);
  }
  return into;
}

/** Resolve one image block into an OpenAI `image_url` part carrying base64 bytes.
 *  The bytes come from the request version the adapter prepared, never from the
 *  durable session log. */
function imagePart(block, images) {
  const version = images.get(block.attachment.attachmentId);
  if (version === undefined) {
    throw new LlmError(
      `CodeBuddy request image ${block.attachment.attachmentId} was not prepared.`,
      'INVALID_REQUEST',
    );
  }
  return {
    type: 'image_url',
    image_url: { url: `data:${version.mediaType};base64,${Buffer.from(version.data).toString('base64')}` },
  };
}

/** Build OpenAI multimodal `content` for one user message: text first, then every
 *  image in durable order. A message without images keeps the plain string form,
 *  so text-only turns serialize byte-for-byte as before. */
function userContent(blocks, images, text) {
  const found = imageBlocks(blocks);
  if (found.length === 0) return text;
  if (images === undefined) {
    throw new LlmError('CodeBuddy image conversion requires the durable attachment service.', 'UNSUPPORTED_CONTENT');
  }
  const parts = [];
  if (text.length > 0) parts.push({ type: 'text', text });
  for (const block of found) parts.push(imagePart(block, images));
  return parts;
}
/** Image parts of one tool result, ready to ride a later multimodal user
 *  message. Returns [] when the result carries no image or none was prepared. */
function toolResultImageParts(content, images) {
  const found = imageBlocks(content);
  if (found.length === 0) return [];
  if (images === undefined) {
    throw new LlmError('CodeBuddy image conversion requires the durable attachment service.', 'UNSUPPORTED_CONTENT');
  }
  return found.map((block) => imagePart(block, images));
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
    content: flattenText(content),
    // reasoning is replayed only on tool-call turns (thinking-mode passback).
    ...(toolCalls.length > 0 && reasoning.length > 0 ? { reasoning_content: reasoning } : {}),
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };
}

/** Tool results ride inside user messages in the harness but the wire wants
 *  one `role: 'tool'` entry per result. */
export function serializeMessages(messages, supportsImages, images) {
  // Reject roles whose OpenAI-compatible history cannot carry image input,
  // rather than letting a text-flattening path erase the image silently.
  for (const message of messages) {
    if (message.role !== 'user' && contentHasImage(message.content || [])) {
      throw new LlmError(`The CodeBuddy adapter cannot represent image content in a ${message.role} message.`, 'UNSUPPORTED_CONTENT');
    }
  }
  const wire = [];
  // Images nested in tool results cannot ride the wire's `tool` entry, so they
  // are collected and flushed as one user message at the next turn boundary.
  let pendingToolImages = [];
  const flushToolImages = () => {
    if (pendingToolImages.length === 0) return;
    wire.push({
      role: 'user',
      content: [{ type: 'text', text: TOOL_RESULT_IMAGE_TEXT }, ...pendingToolImages],
    });
    pendingToolImages = [];
  };
  for (const message of messages) {
    if (!supportsImages && contentHasImage(message.content || [])) {
      throw new LlmError('The selected CodeBuddy model does not accept image content.', 'UNSUPPORTED_CONTENT');
    }
    if (message.role === 'system') {
      flushToolImages();
      wire.push({ role: 'system', content: flattenText(message.content) });
      continue;
    }
    if (message.role === 'assistant') {
      flushToolImages();
      wire.push(serializeAssistant(message));
      continue;
    }
    const results = (message.content || []).filter((b) => b.type === 'tool-result');
    const blocks = message.content || [];
    const text = textOf(blocks);
    if (text.length > 0 || results.length === 0) {
      flushToolImages();
      wire.push({ role: 'user', content: userContent(blocks, images, text) });
    }
    for (const r of results) {
      // The wire's `tool` entry carries text only; the images are set aside for
      // the flush above so the model still receives them on the next turn.
      wire.push({ role: 'tool', tool_call_id: r.toolCallId, content: flattenText(r.content) || '(no output)' });
      pendingToolImages.push(...toolResultImageParts(r.content, images));
    }
  }
  flushToolImages();
  return wire;
}

export function serializeRequest(options, supportsImages, images) {
  const messages = [];
  if (options.system !== undefined) messages.push({ role: 'system', content: options.system });
  messages.push(...serializeMessages(options.messages || [], supportsImages, images));
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
    ...(reasoningEffort(options.reasoningEffort) === undefined
      ? {}
      : { reasoning_effort: reasoningEffort(options.reasoningEffort) }),
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
export async function* translate(payloads, tools = [], onPayload) {
  let nextIndex = 0;
  let textBlock;
  let reasoningBlock;
  const toolBlocks = new Map();
  const order = [];
  let pendingFinish;
  let pendingUsage;
  const open = (kind) => { const b = { index: nextIndex++, kind, text: '' }; order.push(b); return b; };

  for await (const payload of payloads) {
    // Every arriving payload proves the stream is alive; the idle watchdog is
    // re-armed here rather than on a timer of its own.
    if (onPayload !== undefined) onPayload();
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
        // Identity, not accumulation: the wire sends id/name once, on the
        // call's first delta. A later frame that repeats the field empty — or
        // `null`, which some OpenAI-compatible gateways fill in — means "no
        // update", never "clear". Reading `.length` off null would throw here
        // and kill the whole stream.
        block.callId = acceptIdentity(block.callId, call.id);
        const name = call.function && call.function.name;
        block.name = acceptIdentity(block.name, call.function && call.function.name);
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
  constructor({ getAccessToken, connection, readCatalog, identityFromToken, resolveAttachments, defaults = {} }) {
    super();
    this.getAccessToken = getAccessToken;
    this.connection = connection;
    this.readCatalog = readCatalog;
    this.identityFromToken = identityFromToken;
    this.resolveAttachments = resolveAttachments;
    this.defaults = { contextWindow: defaults.contextWindow ?? 128000, maxTokens: defaults.maxTokens ?? 8192 };
    this.catalog = undefined;
    this.catalogRead = undefined;
    this.retryPolicy = defaults.retryPolicy;
  }

  /** Resolve model metadata and bind the SAME catalog generation to the stream
   *  entry point. Without this the base class calls resolveModel and stream
   *  independently, and a catalog refresh landing between them would pair one
   *  generation's capabilities (image / tool-call support) with another's
   *  endpoint. Returns the frozen entry so stream() never re-reads the catalog. */
  async prepareCall(provider, model, signal) {
    const info = await this.resolveModel(provider, model, signal);
    const entry = await this.modelEntry(model, signal);
    return {
      model: info,
      stream: (options) => this.streamWithEntry(options, entry),
    };
  }

  /** One catalog entry, or undefined when the id is unlisted (still routable). */
  async modelEntry(model, signal) {
    try {
      return (await this.models(signal)).find((m) => m.id === model);
    } catch {
      return undefined;
    }
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
  /** Resolve every retained image in one request to encodable bytes.
   *  Returns undefined when the history carries no image, so text-only requests
   *  never touch the attachment service. */
  async prepareImages(options, signal) {
    const refs = new Map();
    const collect = (blocks) => {
      for (const b of blocks || []) {
        if (b.type === 'image') refs.set(b.attachment.attachmentId, b.attachment);
        else if (b.type === 'tool-result') collect(b.content);
      }
    };
    for (const message of options.messages) collect(message.content);
    if (refs.size === 0) return undefined;
    const attachments = this.resolveAttachments ? this.resolveAttachments() : undefined;
    if (attachments === undefined) {
      throw new LlmError('CodeBuddy image conversion requires the durable attachment service.', 'UNSUPPORTED_CONTENT');
    }
    // ImageRequestPolicy is exactly { maxPixels, maxBytes }: the pixel budget is
    // one product, not a width/height pair, and a missing maxPixels makes the
    // attachment store reject the request outright.
    const ordered = [...refs.values()];
    const versions = await Promise.all(ordered.map((ref) => attachments.readImageRequest(ref, {
      maxPixels: CODEBUDDY_IMAGE_MAX_PIXELS,
      maxBytes: CODEBUDDY_IMAGE_MAX_BYTES,
    }, signal)));
    return new Map(ordered.map((ref, index) => [ref.attachmentId, versions[index]]));
  }

  /** Degrade the oldest images to text placeholders once the request would
   *  exceed CodeBuddy's inline budget. CodeBuddy has no Files API, so every
   *  retained image rides the body as base64; without this, a long
   *  screenshot-heavy history would grow the request until the server rejects
   *  it. The projection mirrors the official DeepSeek route: oldest-first,
   *  quantum-rounded, and the placeholder keeps the attachment identity so the
   *  model can ask for it again. */
  offloadImages(messages) {
    const attachments = this.resolveAttachments ? this.resolveAttachments() : undefined;
    const resolveAccess = attachments === undefined
      ? undefined
      : (ref) => resolveImageAttachmentAccess(attachments, (p) => p, ref);
    return offloadRequestImagesWithPolicy(messages, {
      representation: 'base64',
      maxImages: CODEBUDDY_MAX_IMAGES_PER_REQUEST,
      maxBytes: CODEBUDDY_MAX_REQUEST_IMAGE_BYTES,
      byteQuantum: CODEBUDDY_IMAGE_OFFLOAD_BYTE_QUANTUM,
      countQuantum: CODEBUDDY_IMAGE_OFFLOAD_COUNT_QUANTUM,
      byteLength: (ref) => Math.min(ref.bytes ?? 0, CODEBUDDY_IMAGE_MAX_BYTES),
      placeholder: (ref) => offloadedImageText(ref, resolveAccess ? resolveAccess(ref) : undefined),
    });
  }

  /** Provider backoff policy for the harness retry layer. Declaring it keeps
   *  CodeBuddy's retry behaviour aligned with the built-in routes instead of
   *  silently inheriting the framework default. */
  providerRetryPolicy(_provider) {
    return this.retryPolicy;
  }

  async *stream(options) {
    return yield* this.streamWithEntry(options, undefined);
  }

  /** One request bound to a catalog entry already resolved by prepareCall
   *  (undefined falls back to a fresh lookup, keeping direct stream() usable). */
  async *streamWithEntry(options, entryOverride) {
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

    const entry = entryOverride !== undefined ? entryOverride : await this.modelEntry(options.model, options.signal);
    const supportsImages = entry ? entry.supportsImages === true : false;
    if ((options.tools ?? []).length > 0 && entry && entry.supportsToolCall === false) {
      throw new LlmError(`CodeBuddy model "${options.model}" does not support tool calls`, 'UNSUPPORTED_OPTION');
    }

    // Oldest images degrade to text before the bytes are read, so an oversized
    // history never reaches the encoder or the request body.
    const offloaded = this.offloadImages(options.messages);
    const requestOptions = offloaded === options.messages ? options : { ...options, messages: offloaded };
    const payload = JSON.stringify(serializeRequest(requestOptions, supportsImages, await this.prepareImages(requestOptions, options.signal)));
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

    // A stream that goes quiet forever would otherwise hang the agent loop: the
    // watchdog aborts when no chunk arrives within the idle window, while any
    // chunk pulses it. Mirrors the official DeepSeek route.
    const watchdog = idleWatchdog(
      options.signal === undefined ? undefined : options.signal,
      CODEBUDDY_STREAM_IDLE_TIMEOUT_MS,
      CODEBUDDY_STREAM_IDLE_TIMEOUT_CODE,
    );
    try {
      for await (const chunk of translate(parseSse(response.body), options.tools ?? [], () => watchdog.pulse())) {
        yield chunk;
      }
    } catch (error) {
      if (timeoutOf(watchdog.signal, CODEBUDDY_STREAM_IDLE_TIMEOUT_CODE) !== undefined) {
        throw new LlmError(`CodeBuddy stream idle timeout after ${CODEBUDDY_STREAM_IDLE_TIMEOUT_MS}ms`, 'TIMEOUT', { cause: error });
      }
      throw error;
    } finally {
      watchdog[Symbol.dispose]();
    }
  }
}
