/**
 * Unit tests for the identity resolvers in codebuddy-core.mjs.
 *
 * Run: npm test
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolveEnterpriseId } from '../lib/codebuddy-core.mjs';

/** Build an unsigned JWT carrying the given payload. */
function fakeJwt(payload) {
  const enc = (obj) => Buffer.from(JSON.stringify(obj)).toString('base64url');
  return `${enc({ alg: 'none' })}.${enc(payload)}.`;
}

// ------------------------------------------------------- resolveEnterpriseId

test('resolveEnterpriseId: explicit claim beats a group-admin role', () => {
  // Regression: a token whose realm roles carry a stale admin group made the
  // catalog request fail with HTTP 500, while the claim names the live org.
  const token = fakeJwt({
    enterprise_id: 'etahzsqej0n4',
    realm_access: { roles: ['group-admin:a33eec95-42b9-4baa-b64b-468fac0bea74'] },
  });
  assert.equal(resolveEnterpriseId(token), 'etahzsqej0n4');
});

test('resolveEnterpriseId: group-admin role is the fallback when no claim', () => {
  const token = fakeJwt({
    realm_access: { roles: ['offline_access', 'group-admin:ent-from-role'] },
  });
  assert.equal(resolveEnterpriseId(token), 'ent-from-role');
});

test('resolveEnterpriseId: camelCase claims are accepted', () => {
  assert.equal(resolveEnterpriseId(fakeJwt({ enterpriseId: 'ent-camel' })), 'ent-camel');
  assert.equal(resolveEnterpriseId(fakeJwt({ entId: 'ent-short' })), 'ent-short');
});

test('resolveEnterpriseId: no claim and no role resolves to empty', () => {
  assert.equal(resolveEnterpriseId(fakeJwt({ sub: 'user-1' })), '');
  assert.equal(resolveEnterpriseId('not-a-jwt'), '');
});
