/**
 * Unit tests for the serialization layer, tool-call stream, and error mapping.
 *
 * Run: npm test
 *
 * These cover pure logic only; the network-facing paths live in adapter.test.mjs.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SSE_DONE,
  httpErrorCode,
  parseSse,
  serializeMessages,
  serializeRequest,
  translate,
} from '../lib/codebuddy-adapter.mjs';

const IMG_BLOCK = {
  type: 'image',
  attachment: { attachmentId: 'sha256:abc', mediaType: 'image/png', bytes: 10 },
};
const IMG_MAP = new Map([['sha256:abc', { mediaType: 'image/png', data: new Uint8Array([1, 2, 3]) }]]);

/** Build an async iterable of SSE payloads from raw text. */
async function* sseFrom(text) {
  // Real SSE frames end with a blank line after the last data line; the
  // parser flushes on that boundary. Add one if the caller omitted it.
  const framed = text.endsWith('\n\n') ? text : text.replace(/\n*$/, '\n\n');
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(framed));
      controller.close();
    },
  });
  for await (const payload of parseSse(stream)) yield payload;
}

// ---------------------------------------------------------------- parseSse

test('parseSse yields one payload per data line', async () => {
  const out = [];
  for await (const p of sseFrom('data: {"a":1}\n\ndata: {"a":2}\n\ndata: [DONE]\n\n')) out.push(p);
  assert.deepEqual(out, ['{"a":1}', '{"a":2}', SSE_DONE]);
});

test('parseSse joins multi-line data with a newline', async () => {
  const out = [];
  for await (const p of sseFrom('data: line1\ndata: line2\n\ndata: [DONE]\n\n')) out.push(p);
  assert.equal(out[0], 'line1\nline2');
});

test('parseSse tolerates CRLF and comment lines', async () => {
  const out = [];
  for await (const p of sseFrom(': keepalive\r\n\r\ndata: {"a":1}\r\n\r\ndata: [DONE]\r\n\r\n')) out.push(p);
  assert.deepEqual(out, ['{"a":1}', SSE_DONE]);
});

test('parseSse stops at DONE and ignores what follows', async () => {
  const out = [];
  for await (const p of sseFrom('data: [DONE]\n\ndata: {"late":true}\n\n')) out.push(p);
  assert.deepEqual(out, [SSE_DONE]);
});

test('parseSse rejects a stream that ends without DONE', async () => {
  await assert.rejects(async () => {
    for await (const _ of sseFrom('data: {"a":1}\n\n')) { /* drain */ }
  }, /without \[DONE\]/);
});

// --------------------------------------------------------------- translate

test('translate accumulates text deltas into one block', async () => {
  async function* frames() {
    yield JSON.stringify({ choices: [{ delta: { content: 'Hel' } }] });
    yield JSON.stringify({ choices: [{ delta: { content: 'lo' } }] });
    yield SSE_DONE;
  }
  const blocks = [];
  let finish;
  for await (const c of translate(frames(), [])) {
    if (c.type === 'block-end') blocks.push(c.block);
    if (c.type === 'finish') finish = c.reason;
  }
  assert.equal(blocks.length, 1);
  assert.deepEqual(blocks[0], { type: 'text', text: 'Hello' });
  assert.deepEqual(finish, { kind: 'stop' });
});

test('translate skips reasoning_content rather than treating it as text', async () => {
  async function* frames() {
    yield JSON.stringify({ choices: [{ delta: { reasoning_content: 'thinking…' } }] });
    yield JSON.stringify({ choices: [{ delta: { content: 'answer' } }] });
    yield SSE_DONE;
  }
  const texts = [];
  for await (const c of translate(frames(), [])) {
    if (c.type === 'block-end') texts.push(c.block.text);
  }
  assert.deepEqual(texts.sort(), ['answer', 'thinking…'].sort());
  // both blocks exist, but only one is text-typed
});

test('translate keeps a tool-call identity across later empty and null frames', async () => {
  // The regression this guards: reading `.length` off a null id threw and
  // killed the whole stream.
  async function* frames() {
    yield JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_x', function: { name: 'f', arguments: '{' } }] } }] });
    yield JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: null, function: { name: null, arguments: '}' } }] } }] });
    yield JSON.stringify({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] });
    yield SSE_DONE;
  }
  let block;
  for await (const c of translate(frames(), [])) if (c.type === 'block-end') block = c.block;
  assert.equal(block.id, 'call_x');
  assert.equal(block.name, 'f');
  assert.equal(block.arguments, '{}');
});

test('translate buckets parallel tool calls by index', async () => {
  async function* frames() {
    yield JSON.stringify({ choices: [{ delta: { tool_calls: [
      { index: 0, id: 'a', function: { name: 'f0', arguments: '{}' } },
      { index: 1, id: 'b', function: { name: 'f1', arguments: '{}' } },
    ] } }] });
    yield JSON.stringify({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] });
    yield SSE_DONE;
  }
  const calls = [];
  for await (const c of translate(frames(), [])) {
    if (c.type === 'block-end' && c.block.type === 'tool-call') calls.push(c.block.name);
  }
  assert.deepEqual(calls.sort(), ['f0', 'f1']);
});

test('translate reports max-tokens as a length finish', async () => {
  async function* frames() {
    yield JSON.stringify({ choices: [{ delta: { content: 'cut' } }] });
    yield JSON.stringify({ choices: [{ delta: {}, finish_reason: 'length' }] });
    yield SSE_DONE;
  }
  let finish;
  for await (const c of translate(frames(), [])) if (c.type === 'finish') finish = c.reason;
  assert.deepEqual(finish, { kind: 'max-tokens' });
});

test('translate surfaces an empty completed response as an error, not empty text', async () => {
  async function* frames() {
    yield JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] });
    yield SSE_DONE;
  }
  let finish;
  for await (const c of translate(frames(), [])) if (c.type === 'finish') finish = c.reason;
  assert.equal(finish.kind, 'error');
});

test('translate emits usage before finish', async () => {
  async function* frames() {
    yield JSON.stringify({ choices: [{ delta: { content: 'x' } }] });
    yield JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 5, completion_tokens: 1 } });
    yield SSE_DONE;
  }
  const order = [];
  for await (const c of translate(frames(), [])) if (c.type === 'usage' || c.type === 'finish') order.push(c.type);
  assert.deepEqual(order, ['usage', 'finish']);
});

// ------------------------------------------------------- serializeMessages

test('text-only user content stays a plain string', () => {
  const wire = serializeMessages([{ role: 'user', content: [{ type: 'text', text: 'hi' }] }], true, undefined);
  assert.equal(typeof wire[0].content, 'string');
  assert.equal(wire[0].content, 'hi');
});

test('a user image becomes an image_url part', () => {
  const wire = serializeMessages(
    [{ role: 'user', content: [IMG_BLOCK, { type: 'text', text: 'q' }] }],
    true,
    IMG_MAP,
  );
  assert.ok(Array.isArray(wire[0].content));
  assert.deepEqual(wire[0].content.map((p) => p.type), ['text', 'image_url']);
  assert.match(wire[0].content[1].image_url.url, /^data:image\/png;base64,/);
});

test('an image on a non-user role is rejected, not silently dropped', () => {
  for (const role of ['system', 'assistant']) {
    assert.throws(
      () => serializeMessages([{ role, content: [IMG_BLOCK] }], true, IMG_MAP),
      (e) => e.code === 'UNSUPPORTED_CONTENT',
      `${role} should throw`,
    );
  }
});

test('a model without image support rejects image content', () => {
  assert.throws(
    () => serializeMessages([{ role: 'user', content: [IMG_BLOCK] }], false, undefined),
    (e) => e.code === 'UNSUPPORTED_CONTENT',
  );
});

test('tool-result images are flushed as their own user message', () => {
  const wire = serializeMessages([{
    role: 'user',
    content: [
      { type: 'text', text: 'q' },
      { type: 'tool-result', toolCallId: 'c1', content: [{ type: 'text', text: 'out' }, IMG_BLOCK] },
    ],
  }], true, IMG_MAP);
  const tool = wire.find((m) => m.role === 'tool');
  assert.equal(tool.content, 'out');
  const flushed = wire.filter((m) => m.role === 'user' && Array.isArray(m.content)
    && m.content.some((p) => p.type === 'image_url'));
  assert.equal(flushed.length, 1);
  assert.equal(flushed[0].content[0].text, 'Attached image(s) from tool result:');
});

test('assistant text is flattened through nested tool results', () => {
  const wire = serializeMessages([{
    role: 'assistant',
    content: [{ type: 'text', text: 'A' }, { type: 'tool-result', content: [{ type: 'text', text: 'B' }] }],
  }], true, undefined);
  assert.equal(wire[0].content, 'AB');
});

// ------------------------------------------------------ serializeRequest

test('reasoning_effort is omitted for off and none, passed through otherwise', () => {
  const req = (e) => serializeRequest({ model: 'm', messages: [], reasoningEffort: e }, true, undefined);
  assert.equal(req('off').reasoning_effort, undefined);
  assert.equal(req('none').reasoning_effort, undefined);
  assert.equal(req('').reasoning_effort, undefined);
  assert.equal(req('low').reasoning_effort, 'low');
});

test('reasoning_effort normalizes case and the Extra high display name', () => {
  const req = (e) => serializeRequest({ model: 'm', messages: [], reasoningEffort: e }, true, undefined);
  // The catalog spells efforts in lowercase and the live plane rejects a
  // casing mismatch with HTTP 400 (code 11150); config accepts a bare string,
  // so a display name can arrive here.
  assert.equal(req('Max').reasoning_effort, 'max');
  assert.equal(req('HIGH').reasoning_effort, 'high');
  assert.equal(req('  Low  ').reasoning_effort, 'low');
  assert.equal(req('Extra high').reasoning_effort, 'xhigh');
});

test('tools are mapped to the function shape only when present', () => {
  const base = { model: 'm', messages: [] };
  assert.equal(serializeRequest(base, true, undefined).tools, undefined);
  const withTools = serializeRequest({ ...base, tools: [{ name: 'f', description: 'd', parameters: { type: 'object' } }] }, true, undefined);
  assert.equal(withTools.tools[0].type, 'function');
  assert.equal(withTools.tools[0].function.name, 'f');
});

// --------------------------------------------------------- httpErrorCode

test('httpErrorCode classifies the statuses that matter', () => {
  assert.equal(httpErrorCode(401, undefined), 'AUTH');
  assert.equal(httpErrorCode(403, undefined), 'AUTH');
  assert.equal(httpErrorCode(429, undefined), 'RATE_LIMIT');
  assert.equal(httpErrorCode(400, undefined), 'INVALID_REQUEST');
  assert.equal(httpErrorCode(500, undefined), 'SERVER');
  assert.equal(httpErrorCode(418, undefined), 'HTTP_418');
});

// -------------------------------------------------------- EOF as boundary

test('parseSse flushes a final frame that has no trailing blank line', async () => {
  // The fixed bug: the loop only sees newline-terminated lines, so a response
  // ending "data: [DONE]" with no trailing newline was discarded whole and
  // reported as truncated.
  const out = [];
  for await (const p of sseFrom('data: {"a":1}\n\ndata: [DONE]')) out.push(p);
  assert.deepEqual(out, ['{"a":1}', SSE_DONE]);
});

test('parseSse accepts a bare DONE with no newline at all', async () => {
  const out = [];
  for await (const p of sseFrom('data: [DONE]')) out.push(p);
  assert.deepEqual(out, [SSE_DONE]);
});

test('parseSse still rejects a stream with no DONE', async () => {
  await assert.rejects(async () => {
    for await (const _ of sseFrom('data: {"a":1}')) { /* drain */ }
  }, /without \[DONE\]/);
});
