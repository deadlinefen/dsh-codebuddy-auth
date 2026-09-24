#!/usr/bin/env node
/**
 * CodeBuddy (Tencent IOA) login CLI for dsh-codebuddy-auth — zero npm deps.
 *
 * Runs the same OAuth dance the in-harness `codebuddy` tool uses, for setups
 * where the harness is not running (headless bootstrap, CI token rotation):
 *
 *   1. POST /v2/plugin/auth/state — get a state + login URL
 *   2. open the URL in a browser (or print it with --no-browser)
 *   3. poll GET /v2/plugin/auth/token every 3s until the token lands
 *   4. write CODEBUDDY_ACCESS_TOKEN / CODEBUDDY_REFRESH_TOKEN into
 *      $DSH_HOME/.credentials.yaml (flat YAML, line-level edit, mode 0600)
 *
 * That is all it writes. Since the plugin became a native ctx.llm adapter
 * there is no settings route to manage: the mounted adapter owns the
 * `codebuddy` provider and reads the catalog straight from /v3/config. The
 * running plugin's edition comes from its mount-row config (`edition: intl`);
 * this flag only selects which OAuth endpoints the login itself uses.
 *
 * Usage:
 *   node bin/login-flow.mjs                 # fresh flow, opens the browser
 *   node bin/login-flow.mjs --no-browser    # fresh flow, prints the URL only
 *   node bin/login-flow.mjs --state FILE    # resume a flow started elsewhere
 *   node bin/login-flow.mjs --international # OAuth against codebuddy.ai (international)
 */
import { existsSync, readFileSync, writeFileSync, chmodSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  EDITIONS,
  cfgForEdition,
  requestAuthState,
  pollForToken,
  tokenExpiresAt,
  decodeJwtPayload,
} from '../lib/codebuddy-core.mjs';

const ACCESS_REF = 'CODEBUDDY_ACCESS_TOKEN';
const REFRESH_REF = 'CODEBUDDY_REFRESH_TOKEN';

function credentialsPath() {
  return join(process.env.DSH_HOME || join(homedir(), '.dsh'), '.credentials.yaml');
}

/**
 * Rewrite one flat `KEY: "value"` pair in the credentials document without a
 * YAML library: the file is a plain top-level mapping, so line-level surgery
 * with JSON-string quoting (a YAML double-quoted scalar) is exact.
 */
/**
 * Insert or replace one ref inside a credentials document.
 *
 * The document is the provider's v1 layout: the only legal top-level keys are
 * `version`, `refs`, and `records`, and every ref lives indented under `refs:`.
 * Writing a bare top-level `${key}:` line — which a flat pre-release layout
 * allowed — makes the whole file unparseable, and the credentials service is a
 * required plugin, so DSH then fails to start rather than merely losing a token.
 *
 * An existing value may be a folded multi-line scalar; the continuation lines
 * are removed along with the key so no fragment of the old token survives.
 * A document without a top-level `refs:` is refused rather than guessed at.
 *
 * @param text - current file contents (empty for a new file).
 * @param key - ref name, e.g. CODEBUDDY_ACCESS_TOKEN.
 * @param value - new secret.
 * @returns the rewritten document.
 * @throws when the document is not the v1 layout this writer understands.
 */
export function upsertCredentialLine(text, key, value) {
  if (text.trim().length === 0) return `version: 1\nrefs:\n  ${key}: ${JSON.stringify(value)}\n`;

  const lines = text.replace(/\n+$/, '').split('\n');
  const refsIndex = lines.findIndex((l) => /^refs:\s*(\{\s*\})?\s*$/.test(l));
  if (refsIndex < 0) {
    throw new Error(
      `login-flow: refusing to write ${key}: the credentials document has no top-level "refs:" mapping `
      + '(expected the provider v1 layout: version / refs / records).',
    );
  }

  // `refs: {}` is an empty flow mapping: indented block entries cannot follow
  // it, so the braces are replaced by the mapping form before inserting.
  if (/^refs:\s*\{\s*\}\s*$/.test(lines[refsIndex])) lines[refsIndex] = 'refs:';

  // The refs block runs until the next top-level key.
  let end = lines.length;
  for (let i = refsIndex + 1; i < lines.length; i += 1) {
    if (/^[A-Za-z0-9_-]+:/.test(lines[i])) { end = i; break; }
  }

  const body = lines.slice(refsIndex + 1, end);
  const kept = [];
  for (let i = 0; i < body.length; i += 1) {
    const line = body[i];
    const m = line.match(/^\s+([A-Za-z0-9_-]+):/);
    if (m && m[1] === key) {
      // Drop the key line and any folded continuation lines that follow it.
      while (i + 1 < body.length && /^\s+\S/.test(body[i + 1]) && !/^\s+[A-Za-z0-9_-]+:/.test(body[i + 1])) i += 1;
      continue;
    }
    kept.push(line);
  }
  kept.push(`  ${key}: ${JSON.stringify(value)}`);

  return [...lines.slice(0, refsIndex + 1), ...kept, ...lines.slice(end)].join('\n') + '\n';
}

/** Persist both tokens; returns the path written. */
export function writeCredentials(access, refresh, filePath = credentialsPath()) {
  let text = '';
  if (existsSync(filePath)) text = readFileSync(filePath, 'utf8');
  text = upsertCredentialLine(text, ACCESS_REF, access);
  if (refresh) text = upsertCredentialLine(text, REFRESH_REF, refresh);
  writeFileSync(filePath, text);
  chmodSync(filePath, 0o600);
  return filePath;
}

async function main() {
  const args = process.argv.slice(2);
  const noBrowser = args.includes('--no-browser');
  const international = args.includes('--international');
  const stateIdx = args.indexOf('--state');
  const stateFile = stateIdx >= 0 ? args[stateIdx + 1] : null;

  // Region: China edition (default) vs international (codebuddy.ai). Region
  // values are centralized in EDITIONS; --international just picks that one.
  const cfg = cfgForEdition(international ? 'intl' : 'cn');
  if (international) console.log(`[login] using ${EDITIONS.intl.label} endpoints (${EDITIONS.intl.serverUrl})`);
  else console.log(`[login] using ${EDITIONS.cn.label} edition endpoints (${EDITIONS.cn.serverUrl})`);

  let state;
  if (stateFile && existsSync(stateFile)) {
    state = readFileSync(stateFile, 'utf8').trim();
    console.log(`[login] resuming state ${state}`);
  } else {
    const authState = await requestAuthState(cfg);
    state = authState.state;
    console.log(`[login] login URL: ${authState.url}`);
    if (!noBrowser) {
      try {
        const { exec } = await import('node:child_process');
        const cmd = process.platform === 'darwin' ? 'open'
          : process.platform === 'win32' ? 'start' : 'xdg-open';
        const target = process.platform === 'win32' ? `"${authState.url}"` : authState.url;
        exec(`${cmd} ${target}`, () => {});
        console.log('[login] attempted to open your browser');
      } catch {
        console.log('[login] could not open a browser; open the URL above manually');
      }
    }
  }

  const hit = await pollForToken(state, { timeoutMs: 9 * 60 * 1000, cfg });
  if (!hit) {
    console.error('[login] FAILED: timed out waiting for browser login (state is one-shot; run again)');
    process.exitCode = 2;
    return;
  }

  const access = hit.accessToken;
  const refresh = hit.refreshToken || '';
  const path = writeCredentials(access, refresh);
  const exp = tokenExpiresAt(access);
  const sub = decodeJwtPayload(access);
  console.log(`[login] credentials written to ${path}`);
  console.log(`[login] token expires: ${exp ? new Date(exp).toISOString() : 'unknown (JWT has no exp)'}`);
  console.log(`[login] refresh token: ${refresh ? 'stored' : 'none returned'}`);
  if (sub && (sub.user_id || sub.sub)) {
    console.log(`[login] user id: ${sub.user_id || sub.sub}`);
  }
  console.log('[login] next: start (or restart) dsh — the native adapter reads the model catalog from this token at its next request, or ask the agent to run the codebuddy tool (login / sync-models).');
}

const invokedAsScript = process.argv[1]
  && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedAsScript) await main();
