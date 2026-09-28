/**
 * Core CodeBuddy (Tencent IOA) OAuth + API helpers, shared by the plugin
 * (lib/index.js + lib/codebuddy-adapter.mjs) and the standalone login CLI
 * (bin/login-flow.mjs). Plain ESM, no dependencies, Node >= 18.
 *
 * Every request speaks as the CodeBuddy CLI client (`User-Agent: CLI/...`),
 * one identity for login, chat, and model discovery.
 */

/** The two service editions: endpoints for OAuth + /v3/config + chat.
 *  The plugin's edition comes from its mount-row config (`edition: intl`);
 *  the CLI picks it with --international. */
export const EDITIONS = {
  cn: {
    label: 'China',
    serverUrl: 'https://copilot.tencent.com',
    domain: 'www.codebuddy.cn',
    chatBaseURL: 'https://copilot.tencent.com/v2',
  },
  intl: {
    label: 'International',
    serverUrl: 'https://www.codebuddy.ai',
    domain: 'www.codebuddy.ai',
    chatBaseURL: 'https://www.codebuddy.ai/v2',
  },
};

/** The CLI client identity (login `platform` + chat/catalog User-Agent) plus
 *  the China-edition endpoints. Bump versions here only. */
export const DEFAULTS = {
  ...EDITIONS.cn,
  platform: 'CLI',
  appVersion: '4.10.35413651',
  cliVersion: '2.96.0',
  envId: 'production',
  product: 'SaaS',
};

/** DEFAULTS with one edition's endpoints swapped in; 'cn' returns DEFAULTS. */
export function cfgForEdition(editionKey = 'cn') {
  return editionKey === 'intl' ? { ...DEFAULTS, ...EDITIONS.intl } : DEFAULTS;
}

const NO_AUTH_HEADERS = {
  Accept: 'application/json',
  'X-No-Authorization': 'true',
  'X-No-User-Id': 'true',
  'X-No-Enterprise-Id': 'true',
  'X-No-Department-Info': 'true',
};

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Decode a JWT payload without verifying the signature. */
export function decodeJwtPayload(token) {
  try {
    const parts = token.split('.');
    if (parts.length < 2) return null;
    const payload = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const pad = '='.repeat((4 - (payload.length % 4)) % 4);
    return JSON.parse(Buffer.from(payload + pad, 'base64').toString('utf8'));
  } catch {
    return null;
  }
}

export function resolveTenantId(accessToken) {
  const p = decodeJwtPayload(accessToken);
  if (!p) return '';
  const m = (p.iss || '').match(/realms\/sso-([^/]+)$/);
  return p.tenant_id || p.tenantId || (m ? m[1] : '') || '';
}

/** Enterprise id for the identity headers. The explicit claim wins over the
 *  group-admin role: a stale admin group can name an org the SaaS side no
 *  longer knows, which makes /v3/config answer 500. */
export function resolveEnterpriseId(accessToken) {
  const p = decodeJwtPayload(accessToken);
  if (!p) return '';

  const fromClaim = p.enterprise_id || p.enterpriseId || p.ent_id || p.entId;
  if (fromClaim) return fromClaim;

  const roles = (p.realm_access && p.realm_access.roles) ||
    (p.resource_access && p.resource_access.account && p.resource_access.account.roles);
  for (const r of roles || []) {
    const m = r.match(/group-admin:([A-Za-z0-9-]+)/);
    if (m && m[1]) return m[1];
  }

  return '';
}

export function resolveUserId(accessToken) {
  const p = decodeJwtPayload(accessToken);
  return (p && (p.user_id || p.userId || p.uid || p.sub)) || '';
}

export function tokenExpiresAt(accessToken) {
  const p = decodeJwtPayload(accessToken);
  return p && typeof p.exp === 'number' ? p.exp * 1000 : undefined;
}

/** POST /v2/plugin/auth/state — begin the browser OAuth dance. */
export async function requestAuthState(cfg = DEFAULTS) {
  const params = new URLSearchParams({ platform: cfg.platform, ioa: '1' });
  const response = await fetch(`${cfg.serverUrl}/v2/plugin/auth/state?${params}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...NO_AUTH_HEADERS },
  });
  if (!response.ok) throw new Error(`auth state request failed: ${response.status}`);
  const data = await response.json();
  if (data.code !== 0 || !data.data || !data.data.state) {
    throw new Error(`invalid auth state response: ${JSON.stringify(data)}`);
  }
  const url = data.data.authUrl ||
    `${cfg.serverUrl}/login?platform=${cfg.platform}&state=${data.data.state}&ioa=1`;
  return { state: data.data.state, url };
}

/** GET /v2/plugin/auth/token — one poll attempt; null while still pending. */
export async function pollTokenOnce(state, cfg = DEFAULTS) {
  const response = await fetch(`${cfg.serverUrl}/v2/plugin/auth/token?state=${state}`, {
    headers: NO_AUTH_HEADERS,
  });
  if (!response.ok) return null;
  const data = await response.json();
  if (data.code === 0 && data.data && data.data.accessToken) return data.data;
  return null;
}

/** Poll until the token lands or the flow expires. */
export async function pollForToken(state, { timeoutMs = 10 * 60 * 1000, intervalMs = 3000, onTick, cfg = DEFAULTS } = {}) {
  const deadline = Date.now() + timeoutMs;
  let attempt = 0;
  while (Date.now() < deadline) {
    await sleep(intervalMs);
    attempt += 1;
    if (onTick) onTick(attempt);
    try {
      const hit = await pollTokenOnce(state, cfg);
      if (hit) return hit;
    } catch {
      // transient network error — keep polling
    }
  }
  return null;
}

/** POST /v2/plugin/auth/token/refresh */
export async function refreshAccessToken(refreshToken, cfg = DEFAULTS) {
  try {
    const response = await fetch(`${cfg.serverUrl}/v2/plugin/auth/token/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        Authorization: `Bearer ${refreshToken}`,
      },
    });
    if (!response.ok) return null;
    const data = await response.json();
    if (data.code !== 0) return null;
    return data.data || null;
  } catch {
    return null;
  }
}

/** Identity headers derivable from the JWT (stable for one account). */
export function identityHeaders(accessToken) {
  const headers = {};
  const tenantId = resolveTenantId(accessToken);
  const enterpriseId = resolveEnterpriseId(accessToken);
  const userId = resolveUserId(accessToken);
  if (tenantId) headers['X-Tenant-Id'] = tenantId;
  if (enterpriseId) headers['X-Enterprise-Id'] = enterpriseId;
  if (userId) headers['X-User-Id'] = userId;
  return headers;
}

/** GET /v3/config — model discovery. This one call uses the craft/VSCode
 *  identity (X-Agent-Intent + X-IDE-* headers): the craft catalog is the only
 *  one that discloses per-model reasoning metadata (supportedEfforts,
 *  canDisableThinking, defaultEffort) and the craft agent's curated model
 *  list. The CLI catalog answers the same endpoint but carries only fixed
 *  `effort` values and no agent list. Chat requests stay on the CLI identity
 *  (see the adapter) — that split is deliberate, not a leftover. */
export async function fetchRemoteModels(accessToken, cfg = DEFAULTS) {
  const headers = {
    Accept: 'application/json, text/plain, */*',
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
    Authorization: `Bearer ${accessToken}`,
    'User-Agent': 'VSCode/1.133.0 CodeBuddy/4.10.35413651',
    'X-Agent-Intent': 'craft',
    'X-IDE-Type': 'VSCode',
    'X-IDE-Name': 'VSCode',
    'X-IDE-Version': '1.133.0',
    'X-Product-Version': '4.10.35413651',
    'X-Env-ID': cfg.envId,
    'X-Domain': cfg.domain,
    'X-Product': cfg.product,
    ...identityHeaders(accessToken),
  };
  const resp = await fetch(`${cfg.serverUrl}/v3/config`, { headers });
  if (!resp.ok) throw new Error(`/v3/config answered ${resp.status}`);
  const body = await resp.json();
  if (body.code !== 0 || !body.data) throw new Error(`/v3/config replied code ${body.code}`);
  const allModels = body.data.models || [];
  const modelMap = new Map(allModels.map((m) => [m.id, m]));
  const craft = (body.data.agents || []).find((a) => a.name === 'craft');
  const ids = (craft && craft.models) || [];
  return ids
    .map((id) => modelMap.get(id))
    .filter((m) => m && m.supportsToolCall);
}
