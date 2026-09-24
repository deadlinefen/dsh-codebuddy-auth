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
  // Image offload is a host decision in this harness version: a route reports
  // IMAGE_OFFLOAD_REQUIRED and the host records the omission and retries. The
  // older offloadRequestImagesWithPolicy (removed in 0.1.7) let the route
  // rewrite history itself, which left no session record.
  IMAGE_OFFLOAD_REQUIRED_CODE,
  offloadedImageText,
  projectOffloadedImages,
  requiredImageOffload,
  resolveImageAttachmentAccess,
  QUOTA_EXCEEDED_CODE,
  CONTEXT_WINDOW_EXCEEDED_CODE,
  isQuotaExceededError,
  isContextWindowExceededError,
  EMPTY_RESPONSE_CODE,
} from '@deepseek-ai/dsh-llm';
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout';
import { requestImageDimensions } from '@deepseek-ai/dsh-attachment';
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
  // The catalog's per-model supportedEfforts are all lowercase, and the live
  // plane answers a casing mismatch with HTTP 400 invalid_reasoning_effort
  // (code 11150). A display name can reach here — the effort is a bare string
  // in config, so "Max" is accepted there — so normalize case and map the
  // display spelling back to its id.
  return normalized === 'extra high' ? 'xhigh' : normalized;
}

/** Idle window for the SSE stream: no chunk for this long aborts the request
 *  instead of hanging the agent loop. Matches the official route's default. */
const CODEBUDDY_STREAM_IDLE_TIMEOUT_MS = 60000;
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
  // EOF is a frame boundary too. Two things can still be pending when the
  // response ends:
  //   1. a final line with no trailing newline, which the loop above never
  //      sees (the real CodeBuddy API ends with "data: [DONE]\n\n", but a
  //      server is not obliged to, and dropping it lost the whole stream);
  //   2. data lines buffered since the last blank line.
  // Flush both before concluding the stream was truncated.
  const tail = buffer.trim();
  if (tail.startsWith('data:')) dataLines.push(tail.slice(5).replace(/^ /, ''));
  buffer = '';
  if (dataLines.length > 0) {
    const payload = dataLines.join('\n');
    dataLines = [];
    yield payload;
    if (payload === SSE_DONE) return;
  }
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
  // rather than letting a text-flattening path erase the image silently. Tool
  // results are the exception: the wire's `tool` entry carries text only, so
  // their images are collected below and sent as the next user message.
  for (const message of messages) {
    if (message.role !== 'user' && message.role !== 'tool' && contentHasImage(message.content || [])) {
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
    // Session format v4 lifts a tool result to a message-level field: the wire
    // shape is {role:'tool', toolCallId, content} rather than a tool-result
    // block inside content. Without this the message falls through to the user
    // branch below and the assistant's tool_calls end up unpaired, which the
    // server rejects with "tool calls and tool results do not match".
    if (message.role === 'tool' && message.toolCallId !== undefined) {
      wire.push({
        role: 'tool',
        tool_call_id: message.toolCallId,
        content: flattenText(message.content) || '(no output)',
        // Without this a failed tool result reads as a success to the model.
        ...(message.isError === undefined ? {} : { is_error: message.isError }),
      });
      pendingToolImages.push(...toolResultImageParts(message.content, images));
      continue;
    }
    const results = (message.content || []).filter((b) => b.type === 'tool-result');
    // Regular blocks only. imageBlocks() recurses into tool-result content, so
    // passing the whole message here would pull a tool-result's image into this
    // user message AND leave it queued for flushToolImages() — sending the same
    // image twice, once attached to the wrong turn.
    const blocks = (message.content || []).filter((b) => b.type !== 'tool-result');
    const text = textOf(blocks);
    if (text.length > 0 || results.length === 0) {
      flushToolImages();
      wire.push({ role: 'user', content: userContent(blocks, images, text) });
    }
    for (const r of results) {
      // The wire's `tool` entry carries text only; the images are set aside for
      // the flush above so the model still receives them on the next turn.
      wire.push({
        role: 'tool',
        tool_call_id: r.toolCallId,
        content: flattenText(r.content) || '(no output)',
        ...(r.isError === undefined ? {} : { is_error: r.isError }),
      });
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
      // Every non-final chunk carries finish_reason "" (some gateways null).
      // Treating that as a stop reason forges an error whose code is the empty
      // string, and the agent loop replays it through new LlmError(message,
      // code), which rejects an empty code and ends the turn with an exception.
      const finished = choice.finish_reason;
      if (typeof finished === 'string' && finished.trim().length > 0) pendingFinish = mapFinishReason(finished.trim());
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
        // An occurrence the host offloaded renders as a placeholder, so its
        // bytes must not be read.
        if (b.type === 'image') {
          if (b.offloaded !== true) refs.set(b.attachment.attachmentId, b.attachment);
        } else if (b.type === 'tool-result') collect(b.content);
      }
    };
    for (const message of options.messages) collect(message.content);
    if (refs.size === 0) return undefined;
    const attachments = this.resolveAttachments ? this.resolveAttachments() : undefined;
    if (attachments === undefined) {
      throw new LlmError('CodeBuddy image conversion requires the durable attachment service.', 'UNSUPPORTED_CONTENT');
    }
    // The store validates a request target as { width, height, maxBytes } and
    // rejects a missing width outright, so the pixel budget is converted here
    // rather than passed as maxPixels.
    const ordered = [...refs.values()];
    const versions = await Promise.all(ordered.map((ref) => attachments.readImageRequest(ref, {
      ...requestImageDimensions(ref.width, ref.height, CODEBUDDY_IMAGE_MAX_PIXELS),
      maxBytes: CODEBUDDY_IMAGE_MAX_BYTES,
    }, signal)));
    return new Map(ordered.map((ref, index) => [ref.attachmentId, versions[index]]));
  }

  /** Placeholder text for one image the host has decided to omit. Keeps the
   *  attachment identity so the model can ask for the image again. */
  offloadedText(ref) {
    const attachments = this.resolveAttachments ? this.resolveAttachments() : undefined;
    const access = attachments === undefined
      ? undefined
      : resolveImageAttachmentAccess(attachments, (p) => p, ref);
    return offloadedImageText(ref, access);
  }

  /** Render the history the host has already projected, then require host
   *  offload for whatever still exceeds CodeBuddy's inline budget.
   *
   *  CodeBuddy has no Files API, so every retained image rides the body as
   *  base64 and the byte budget is the binding constraint. This route does not
   *  rewrite history itself: it throws IMAGE_OFFLOAD_REQUIRED naming how many
   *  oldest occurrences must go, the host records that omission in the session
   *  and retries, and the next attempt arrives with those occurrences marked
   *  `offloaded`. Mirrors the official DeepSeek and pi-ai routes so all three
   *  degrade identically and the omission is visible in the session log.
   *
   *  @param messages - derived history carrying the host's `offloaded` marks.
   *  @param versions - prepared bytes per attachment id, for the exact length.
   *  @returns the projected history.
   *  @throws LlmError IMAGE_OFFLOAD_REQUIRED with `offloadImages` when over. */
  projectImages(messages, versions) {
    const offloadImages = requiredImageOffload(messages, {
      representation: 'base64',
      maxImages: CODEBUDDY_MAX_IMAGES_PER_REQUEST,
      maxBytes: CODEBUDDY_MAX_REQUEST_IMAGE_BYTES,
      byteQuantum: CODEBUDDY_IMAGE_OFFLOAD_BYTE_QUANTUM,
      countQuantum: CODEBUDDY_IMAGE_OFFLOAD_COUNT_QUANTUM,
    }, (block) => {
      const version = versions.get(block.attachment.attachmentId);
      // The store reports the encoded length; fall back to the raw size so a
      // missing version cannot silently under-count and let the request through.
      return version === undefined ? Math.min(block.attachment.bytes ?? 0, CODEBUDDY_IMAGE_MAX_BYTES) : version.bytes;
    });
    if (offloadImages > 0) {
      throw new LlmError(
        `CodeBuddy request images exceed the inline budget; ${offloadImages} more oldest occurrence(s) must be offloaded.`,
        IMAGE_OFFLOAD_REQUIRED_CODE,
        { offloadImages },
      );
    }
    return projectOffloadedImages(messages, (ref) => this.offloadedText(ref));
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

    // Placeholders the host already decided on render first, then the retained
    // images are measured: if they still exceed the inline budget this throws
    // IMAGE_OFFLOAD_REQUIRED and the host retries with more marked offloaded.
    const images = await this.prepareImages(options, options.signal);
    const projected = this.projectImages(options.messages, images ?? new Map());
    const requestOptions = projected === options.messages ? options : { ...options, messages: projected };
    const payload = JSON.stringify(serializeRequest(requestOptions, supportsImages, images));
    // The watchdog spans both phases. It must exist before the request, because
    // its signal is what aborts a connection that never produces a first byte.
    const watchdog = idleWatchdog(
      options.signal === undefined ? undefined : options.signal,
      CODEBUDDY_STREAM_IDLE_TIMEOUT_MS,
      CODEBUDDY_STREAM_IDLE_TIMEOUT_CODE,
    );
    let response;
    try {
      // The timer is armed inside next(), not by construction. fetch() is eager,
      // so the connect phase completes before any chunk is read: only routing
      // the request through next() puts a silent server under the timer.
      response = await watchdog.next(
        (async function* () {
          yield await fetch(`${connection.chatBaseURL}/chat/completions`, {
            method: 'POST',
            headers,
            body: payload,
            signal: watchdog.signal,
          });
        })()[Symbol.asyncIterator](),
      ).then((r) => r.value);
    } catch (error) {
      if (timeoutOf(watchdog.signal, CODEBUDDY_STREAM_IDLE_TIMEOUT_CODE) !== undefined) {
        watchdog[Symbol.dispose]();
        throw new LlmError(`CodeBuddy request idle timeout after ${CODEBUDDY_STREAM_IDLE_TIMEOUT_MS}ms`, 'TIMEOUT', { cause: error });
      }
      watchdog[Symbol.dispose]();
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
        // CodeBuddy answers {code, msg, extError:{code, message}} where the
        // OpenAI shape is {error:{message}}. Reading only parsed.error kept the
        // real cause out of the UI (the 400 that says "tool calls and tool
        // results do not match" surfaced as a bare "(HTTP 400)").
        const nested = parsed !== null && typeof parsed.extError === 'object' ? parsed.extError : undefined;
        const openai = parsed !== null && typeof parsed.error === 'object' ? parsed.error : undefined;
        const detail = openai?.message ?? nested?.message ?? parsed?.msg;
        // Keep the numeric business code (11148, 11150, ...): it is the fastest
        // way to look a CodeBuddy failure up, so prefer it over the slug.
        const code = parsed?.code ?? nested?.code ?? openai?.code;
        if (typeof detail === 'string' && detail.length > 0) message = detail;
        providerError = {};
        if (code !== undefined && code !== null && String(code).length > 0) providerError.code = String(code);
        if (typeof detail === 'string' && detail.length > 0) providerError.message = detail;
        if (Object.keys(providerError).length === 0) providerError = undefined;
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

    // Drive the stream through watchdog.next(): the timer is armed there, and
    // pulse() only re-arms while a next() is outstanding. Pulsing without ever
    // calling next() would leave the timer permanently unarmed.
    try {
      const iterator = translate(parseSse(response.body), options.tools ?? [])[Symbol.asyncIterator]();
      for (;;) {
        const result = await watchdog.next(iterator);
        if (result.done) break;
        watchdog.pulse();
        yield result.value;
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
