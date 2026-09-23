/**
 * Watchdog and request-path tests.
 *
 * These exercise the request path against a real local HTTP server rather than
 * a mocked fetch, because the bug being guarded was structural: the timer was
 * armed inside next() and the code never called next(), so a mock of fetch
 * would have passed while the real thing hung forever.
 *
 * Run: npm test
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { CodebuddyAdapter } from '../lib/codebuddy-adapter.mjs';

/**
 * Build an adapter whose chat plane points at a local server.
 * `handler` decides how each request is answered.
 */
function adapterServing(handler) {
  const server = createServer(handler);
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      const adapter = new CodebuddyAdapter({
        getAccessToken: async () => 'test-token',
        connection: () => ({ chatBaseURL: `http://127.0.0.1:${port}/v2`, domain: 'x', cliVersion: '1' }),
        readCatalog: async () => [{ id: 'm', supportsToolCall: true, supportsImages: false }],
        identityFromToken: () => ({}),
      });
      resolve({ adapter, server, port });
    });
  });
}

/** Drain one stream call, returning the error it threw (or undefined). */
async function drain(adapter, options = {}) {
  const t0 = Date.now();
  let error;
  try {
    for await (const _ of adapter.stream({
      model: 'm',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'x' }] }],
      ...options,
    })) { /* drain */ }
  } catch (e) {
    error = e;
  }
  return { error, ms: Date.now() - t0 };
}

const SSE_OK = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

test('a normal stream completes and yields text', async () => {
  const { adapter, server } = await adapterServing((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.end(SSE_OK);
  });
  try {
    const chunks = [];
    for await (const c of adapter.stream({
      model: 'm',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'x' }] }],
    })) chunks.push(c);
    const text = chunks.filter((c) => c.type === 'text-delta').map((c) => c.text).join('');
    assert.equal(text, 'hi');
    assert.ok(chunks.some((c) => c.type === 'finish' && c.reason.kind === 'stop'));
  } finally {
    server.close();
  }
});

test('the idle watchdog aborts a connection that never responds', async (t) => {
  // The regression: a server that accepts and then stays silent used to hang
  // forever, because the timer was never armed.
  const { adapter, server } = await adapterServing(() => { /* accept, never answer */ });
  try {
    const { error, ms } = await drain(adapter);
    t.diagnostic(`idle watchdog fired after ${ms}ms`);
    assert.ok(error, 'a silent server must produce an error, not hang');
    assert.equal(error.code, 'TIMEOUT');
    // 60s window; allow slack for timer scheduling but catch "never fired".
    assert.ok(ms >= 55000 && ms < 90000, `expected ~60s, got ${ms}ms`);
  } finally {
    server.closeAllConnections?.();
    server.close();
  }
});

test('the idle watchdog aborts a stream that goes silent mid-body', async (t) => {
  const { adapter, server } = await adapterServing((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write('data: {"choices":[{"delta":{"content":"par"}}]}\n\n');
    // then never finish the response
  });
  try {
    const { error, ms } = await drain(adapter);
    t.diagnostic(`mid-body idle fired after ${ms}ms`);
    assert.ok(error, 'a stalled body must produce an error');
    assert.equal(error.code, 'TIMEOUT');
    assert.ok(ms >= 55000 && ms < 90000, `expected ~60s, got ${ms}ms`);
  } finally {
    server.closeAllConnections?.();
    server.close();
  }
});

test('a caller abort is reported as ABORTED, not TIMEOUT', async () => {
  const { adapter, server } = await adapterServing(() => { /* hold */ });
  try {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 300);
    const { error, ms } = await drain(adapter, { signal: controller.signal });
    assert.ok(error, 'an aborted call must reject');
    assert.equal(error.code, 'ABORTED');
    assert.ok(ms < 5000, `abort should be prompt, took ${ms}ms`);
  } finally {
    server.closeAllConnections?.();
    server.close();
  }
});

test('an HTTP error is mapped to its code, not a timeout', async () => {
  const { adapter, server } = await adapterServing((req, res) => {
    res.writeHead(429, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'slow down' } }));
  });
  try {
    const { error, ms } = await drain(adapter);
    assert.ok(error);
    assert.equal(error.code, 'RATE_LIMIT');
    assert.ok(ms < 5000, 'a fast error should not wait for the watchdog');
  } finally {
    server.close();
  }
});

test('a 400 naming the context window maps to CONTEXT_WINDOW_EXCEEDED', async () => {
  const { adapter, server } = await adapterServing((req, res) => {
    res.writeHead(400, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'maximum context length exceeded' } }));
  });
  try {
    const { error } = await drain(adapter);
    assert.ok(error);
    assert.match(error.code, /CONTEXT_WINDOW/);
  } finally {
    server.close();
  }
});

test('a model marked as not tool-calling refuses a tool request up front', async () => {
  const server = createServer(() => {});
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address();
  const adapter = new CodebuddyAdapter({
    getAccessToken: async () => 't',
    connection: () => ({ chatBaseURL: `http://127.0.0.1:${port}/v2`, domain: 'x', cliVersion: '1' }),
    readCatalog: async () => [{ id: 'm', supportsToolCall: false }],
    identityFromToken: () => ({}),
  });
  try {
    const { error, ms } = await drain(adapter, { tools: [{ name: 'f', parameters: {} }] });
    assert.ok(error);
    assert.equal(error.code, 'UNSUPPORTED_OPTION');
    assert.ok(ms < 3000, 'should fail before issuing a request');
  } finally {
    server.close();
  }
});

test('a missing credential fails fast with MISSING_CREDENTIAL', async () => {
  const adapter = new CodebuddyAdapter({
    getAccessToken: async () => undefined,
    connection: () => ({ chatBaseURL: 'http://127.0.0.1:1/v2', domain: 'x', cliVersion: '1' }),
    readCatalog: async () => [],
    identityFromToken: () => ({}),
  });
  const { error, ms } = await drain(adapter);
  assert.ok(error);
  assert.equal(error.code, 'MISSING_CREDENTIAL');
  assert.ok(ms < 3000);
});

// ------------------------------------------------------------- prepareCall

test('prepareCall freezes one catalog generation for the stream it returns', async () => {
  let reads = 0;
  let supportsImages = true;
  const server = createServer((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.end('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address();
  const adapter = new CodebuddyAdapter({
    getAccessToken: async () => 't',
    connection: () => ({ chatBaseURL: `http://127.0.0.1:${port}/v2`, domain: 'x', cliVersion: '1' }),
    readCatalog: async () => { reads++; return [{ id: 'm', supportsToolCall: true, supportsImages }]; },
    identityFromToken: () => ({}),
  });
  try {
    const prepared = await adapter.prepareCall('codebuddy', 'm', undefined);
    assert.equal(prepared.model.id, 'm');
    assert.deepEqual(prepared.model.inputModalities, ['text', 'image']);
    const afterPrepare = reads;

    // Flip the catalog before consuming the prepared stream: the prepared
    // generation must win, or capabilities and endpoint could disagree.
    supportsImages = false;
    const chunks = [];
    for await (const c of prepared.stream({ model: 'm', messages: [{ role: 'user', content: [{ type: 'text', text: 'x' }] }] })) chunks.push(c);
    assert.ok(chunks.some((c) => c.type === 'finish'));
    assert.equal(reads, afterPrepare, 'the prepared call must not re-read the catalog');
  } finally {
    server.close();
  }
});

test('prepareCall still works for a model the catalog does not list', async () => {
  const adapter = new CodebuddyAdapter({
    getAccessToken: async () => 't',
    connection: () => ({ chatBaseURL: 'http://127.0.0.1:1/v2', domain: 'x', cliVersion: '1' }),
    readCatalog: async () => [{ id: 'other', supportsToolCall: true }],
    identityFromToken: () => ({}),
  });
  const prepared = await adapter.prepareCall('codebuddy', 'unlisted', undefined);
  assert.equal(prepared.model.id, 'unlisted');
  assert.deepEqual(prepared.model.inputModalities, ['text']);
});

test('providerRetryPolicy reports the configured policy', () => {
  const adapter = new CodebuddyAdapter({
    getAccessToken: async () => 't',
    connection: () => ({}),
    readCatalog: async () => [],
    identityFromToken: () => ({}),
    defaults: { retryPolicy: { maxAttempts: 3 } },
  });
  assert.deepEqual(adapter.providerRetryPolicy('codebuddy'), { maxAttempts: 3 });
});

// ----------------------------------------------------------- image prep

test('an image request without the attachment service fails clearly', async () => {
  const adapter = new CodebuddyAdapter({
    getAccessToken: async () => 't',
    connection: () => ({ chatBaseURL: 'http://127.0.0.1:1/v2', domain: 'x', cliVersion: '1' }),
    readCatalog: async () => [{ id: 'm', supportsToolCall: true, supportsImages: true }],
    identityFromToken: () => ({}),
    // resolveAttachments deliberately absent
  });
  const { error } = await drain(adapter);
  assert.ok(error);
});

test('offloadImages is a no-op when nothing exceeds the budget', () => {
  const adapter = new CodebuddyAdapter({
    getAccessToken: async () => 't',
    connection: () => ({}),
    readCatalog: async () => [],
    identityFromToken: () => ({}),
    resolveAttachments: () => ({ imageHostPath: () => undefined }),
  });
  const messages = [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }];
  const out = adapter.offloadImages(messages);
  assert.equal(out, messages, 'an under-budget request should pass through unchanged');
});
