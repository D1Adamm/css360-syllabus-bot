import { beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * Composed URLs for the session, classroom-code and administration clients,
 * against the deployed base — the same guarantee `dbApi.paths.test.ts` gives
 * the persistence client. Each expectation is the FastAPI route as mounted.
 */

const fetchMock = vi.fn();
vi.stubGlobal('fetch', fetchMock);
vi.stubEnv('VITE_API_BASE_URL', 'https://aiswe.uwb.edu/api');

const HOST = 'https://aiswe.uwb.edu';
const COURSE = 'css-360-winter-2026-a7rp';
const TOKEN = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG';

function requested(callIndex = 0): { url: string; method: string } {
  const [url, init] = fetchMock.mock.calls[callIndex] as [string, RequestInit];
  return { url, method: init.method ?? 'GET' };
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
});

const CASES: Array<{ name: string; call: () => Promise<unknown>; method: string; route: string }> = [
  { name: 'fetchSession', call: async () => (await import('./authApi')).fetchSession(), method: 'GET', route: '/api/auth/session' },
  { name: 'login', call: async () => (await import('./authApi')).login('a@uw.edu', 'p'), method: 'POST', route: '/api/auth/login' },
  { name: 'logout', call: async () => (await import('./authApi')).logout(), method: 'POST', route: '/api/auth/logout' },
  { name: 'joinCourse', call: async () => (await import('./authApi')).joinCourse('7K4P9X'), method: 'POST', route: '/api/auth/join' },
  { name: 'previewInvitation', call: async () => (await import('./authApi')).previewInvitation(TOKEN), method: 'GET', route: `/api/auth/invitations/${TOKEN}` },
  { name: 'acceptInvitation', call: async () => (await import('./authApi')).acceptInvitation(TOKEN, { password: 'x' }), method: 'POST', route: `/api/auth/invitations/${TOKEN}/accept` },
  { name: 'changePassword', call: async () => (await import('./authApi')).changePassword('a', 'b'), method: 'POST', route: '/api/auth/password' },
  { name: 'listStudentInvites', call: async () => (await import('./inviteApi')).listStudentInvites(COURSE), method: 'GET', route: `/api/courses/${COURSE}/student-invites` },
  { name: 'createStudentInvite', call: async () => (await import('./inviteApi')).createStudentInvite(COURSE), method: 'POST', route: `/api/courses/${COURSE}/student-invites` },
  { name: 'revokeStudentInvite', call: async () => (await import('./inviteApi')).revokeStudentInvite(COURSE, 'inv-1'), method: 'POST', route: `/api/courses/${COURSE}/student-invites/inv-1/revoke` },
  { name: 'listUsers', call: async () => (await import('./adminPeopleApi')).listUsers(), method: 'GET', route: '/api/admin/users' },
  { name: 'updateUser', call: async () => (await import('./adminPeopleApi')).updateUser('u-1', { disabled: true }), method: 'PATCH', route: '/api/admin/users/u-1' },
  { name: 'addMembership', call: async () => (await import('./adminPeopleApi')).addMembership('u-1', COURSE), method: 'PUT', route: `/api/admin/users/u-1/courses/${COURSE}` },
  { name: 'removeMembership', call: async () => (await import('./adminPeopleApi')).removeMembership('u-1', COURSE), method: 'DELETE', route: `/api/admin/users/u-1/courses/${COURSE}` },
  { name: 'createInvitation', call: async () => (await import('./adminPeopleApi')).createInvitation({ kind: 'admin' }), method: 'POST', route: '/api/admin/invitations' },
  { name: 'listInvitations', call: async () => (await import('./adminPeopleApi')).listInvitations('active'), method: 'GET', route: '/api/admin/invitations?status=active' },
  { name: 'revokeInvitation', call: async () => (await import('./adminPeopleApi')).revokeInvitation('inv-1'), method: 'POST', route: '/api/admin/invitations/inv-1/revoke' },
  { name: 'createResetInvite', call: async () => (await import('./adminPeopleApi')).createResetInvite('u-1'), method: 'POST', route: '/api/admin/users/u-1/reset-invite' },
  { name: 'listAudit', call: async () => (await import('./adminPeopleApi')).listAudit(50, 900), method: 'GET', route: '/api/admin/audit?limit=50&before=900' },
];

describe('session, classroom-code and administration clients', () => {
  for (const { name, call, method, route } of CASES) {
    it(`${name} → ${method} ${route}`, async () => {
      await call();
      expect(requested()).toEqual({ url: `${HOST}${route}`, method });
    });
  }

  it('never doubles the /api prefix', async () => {
    const { fetchSession } = await import('./authApi');
    await fetchSession();
    expect(requested().url).not.toContain('/api/api/');
  });

  it('builds the join links students are given', async () => {
    const { joinLinkUrl, joinPageUrl } = await import('./inviteApi');
    expect(joinPageUrl('https://aiswe.uwb.edu/')).toBe('https://aiswe.uwb.edu/join');
    expect(joinLinkUrl('https://aiswe.uwb.edu', '7K4P9X')).toBe('https://aiswe.uwb.edu/join/7K4P9X');
  });
});
