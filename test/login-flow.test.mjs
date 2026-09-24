/**
 * Tests for the standalone login CLI's credential writer.
 *
 * `writeCredentials` edits a document that `dsh-credentials-local` parses
 * strictly: the only legal top-level keys are version/refs/records. Writing a
 * flat top-level token key into an already-v1 file makes the whole file
 * unparseable, and that takes the credentials service down with it — which
 * fails DSH startup outright, not just this plugin.
 *
 * The assertions run the writer's output through the provider's real parser
 * rather than a hand-rolled shape check: the parser is the thing that has to
 * accept it.
 *
 * Run: npm test
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { upsertCredentialLine, writeCredentials } from '../bin/login-flow.mjs';

const require = createRequire(import.meta.url);
const HARNESS = process.env.DSH_HARNESS_DIR
  ?? '/root/.nvm/versions/node/v24.20.0/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai';
const { parseCredentialsDocument } = await import(`${HARNESS}/dsh-credentials-local/lib/index.js`);

/** A v1 document shaped like the real one, with a folded multi-line scalar. */
const V1 = [
  'version: 1',
  'refs:',
  '  CODEBUDDY_ACCESS_TOKEN: "old-access-part-one' + '\\',
  '    old-access-part-two"',
  '  CODEBUDDY_REFRESH_TOKEN: "old-refresh"',
  '  OTHER_KEY: "keep-me"',
  'records:',
  '  client-connection/browser-session:',
  '    kind: grant',
  '    payload:',
  '      version: 1',
  '      secret: "abc"',
  '',
].join('\n');

const parse = (text) => parseCredentialsDocument(text, 'test.yaml');

test('replacing a folded token leaves a document the real parser accepts', () => {
  const out = upsertCredentialLine(V1, 'CODEBUDDY_ACCESS_TOKEN', 'new-access');
  const doc = parse(out);
  assert.equal(doc.refs.get('CODEBUDDY_ACCESS_TOKEN'), 'new-access');
  // The fold continuation must be gone, or the old bytes stay in the value.
  assert.ok(!out.includes('old-access-part-two'), 'continuation line was left behind');
  // Neighbours survive.
  assert.equal(doc.refs.get('CODEBUDDY_REFRESH_TOKEN'), 'old-refresh');
  assert.equal(doc.refs.get('OTHER_KEY'), 'keep-me');
  assert.deepEqual([...doc.records.keys()], ['client-connection/browser-session']);
});

test('the token lands under refs, never as a top-level key', () => {
  const out = upsertCredentialLine(V1, 'CODEBUDDY_ACCESS_TOKEN', 'x');
  assert.match(out, /^  CODEBUDDY_ACCESS_TOKEN:/m, 'must be indented under refs');
  assert.ok(!/^CODEBUDDY_ACCESS_TOKEN:/m.test(out), 'must not be a top-level key');
});

test('replacing is idempotent and does not accumulate duplicates', () => {
  const once = upsertCredentialLine(V1, 'CODEBUDDY_ACCESS_TOKEN', 'a');
  const twice = upsertCredentialLine(once, 'CODEBUDDY_ACCESS_TOKEN', 'b');
  const occurrences = twice.split('\n').filter((l) => /CODEBUDDY_ACCESS_TOKEN:/.test(l));
  assert.equal(occurrences.length, 1);
  assert.equal(parse(twice).refs.get('CODEBUDDY_ACCESS_TOKEN'), 'b');
});

test('a key that is absent is inserted into refs', () => {
  const out = upsertCredentialLine(V1, 'CODEBUDDY_NEW_KEY', 'v');
  assert.equal(parse(out).refs.get('CODEBUDDY_NEW_KEY'), 'v');
  assert.equal(parse(out).refs.get('CODEBUDDY_ACCESS_TOKEN'), 'old-access-part-oneold-access-part-two');
});

test('an empty file becomes a valid v1 document', () => {
  const out = upsertCredentialLine('', 'CODEBUDDY_ACCESS_TOKEN', 'v');
  const doc = parse(out);
  assert.equal(doc.refs.get('CODEBUDDY_ACCESS_TOKEN'), 'v');
});

test('normalizes an empty refs mapping', () => {
  const out = upsertCredentialLine('version: 1\nrefs: {}\n', 'CODEBUDDY_ACCESS_TOKEN', 'v');
  assert.equal(parse(out).refs.get('CODEBUDDY_ACCESS_TOKEN'), 'v');
});

test('refuses to write into a document it does not understand', () => {
  // A flat, pre-release layout: guessing would risk clobbering a file the
  // provider may still migrate on its own.
  assert.throws(
    () => upsertCredentialLine('CODEBUDDY_ACCESS_TOKEN: "old"\n', 'CODEBUDDY_ACCESS_TOKEN', 'v'),
    /refs/,
  );
});

test('values containing quotes and YAML metacharacters survive a round trip', () => {
  const value = 'a"b\\c: d # e';
  const out = upsertCredentialLine(V1, 'CODEBUDDY_ACCESS_TOKEN', value);
  assert.equal(parse(out).refs.get('CODEBUDDY_ACCESS_TOKEN'), value);
});

test('writeCredentials persists both tokens in one pass', () => {
  const dir = mkdtempSync(join(tmpdir(), 'cb-login-'));
  const file = join(dir, '.credentials.yaml');
  writeFileSync(file, V1);
  writeCredentials('acc', 'ref', file);
  const doc = parse(readFileSync(file, 'utf8'));
  assert.equal(doc.refs.get('CODEBUDDY_ACCESS_TOKEN'), 'acc');
  assert.equal(doc.refs.get('CODEBUDDY_REFRESH_TOKEN'), 'ref');
  assert.equal(doc.refs.get('OTHER_KEY'), 'keep-me');
});
