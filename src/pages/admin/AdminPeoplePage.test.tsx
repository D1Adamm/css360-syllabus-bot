/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const api = vi.hoisted(() => ({
  listUsers: vi.fn(),
  updateUser: vi.fn(),
  addMembership: vi.fn(),
  removeMembership: vi.fn(),
  createInvitation: vi.fn(),
  listInvitations: vi.fn(),
  revokeInvitation: vi.fn(),
  createResetInvite: vi.fn(),
}));

vi.mock('../../lib/adminPeopleApi', async () => {
  const actual = await vi.importActual<typeof import('../../lib/adminPeopleApi')>(
    '../../lib/adminPeopleApi',
  );
  return { ...actual, ...api };
});

vi.mock('../../hooks/useCourses', () => ({
  useCourses: () => ({
    state: {
      status: 'ready',
      courses: [
        { courseId: 'css-360-winter-2026-a7rp', metadata: { name: 'CSS 360' } },
        { courseId: 'css-350-spring-2026-n3h9', metadata: { name: 'CSS 350' } },
      ],
    },
    retry: vi.fn(),
  }),
}));

import { SessionProvider } from '../../context/SessionContext';
import { AdminPeoplePage } from './AdminPeoplePage';

const ADMIN_SESSION = {
  user: { userId: 'u-admin', email: 'admin@uw.edu', displayName: 'Admin', role: 'admin' as const, courseIds: [] },
  participant: null,
};

const PROF = {
  userId: 'u-prof',
  email: 'prof@uw.edu',
  displayName: 'Prof Example',
  role: 'professor' as const,
  disabled: false,
  courseIds: ['css-360-winter-2026-a7rp'],
};

function renderPage() {
  return render(
    <SessionProvider initialSession={ADMIN_SESSION}>
      <AdminPeoplePage />
    </SessionProvider>,
  );
}

beforeEach(() => {
  for (const mock of Object.values(api)) {
    mock.mockReset();
  }
  api.listUsers.mockResolvedValue({ count: 2, users: [{ ...ADMIN_SESSION.user, disabled: false }, PROF] });
  api.listInvitations.mockResolvedValue({ count: 0, invitations: [] });
  Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
});

afterEach(() => {
  cleanup();
});

describe('AdminPeoplePage', () => {
  it('lists accounts with their role and courses', async () => {
    renderPage();
    const list = await screen.findByRole('list', { name: 'Accounts' });
    expect(within(list).getByText('Prof Example')).toBeInTheDocument();
    expect(within(list).getByText('instructor')).toBeInTheDocument();
    expect(within(list).getByText('administrator')).toBeInTheDocument();
    expect(within(list).getByText('CSS 360')).toBeInTheDocument();
  });

  it('creates an instructor invitation for the chosen courses and shows the link once', async () => {
    api.createInvitation.mockResolvedValue({
      invitationId: 'inv-1',
      kind: 'professor',
      status: 'active',
      createdAt: '2026-09-04T12:00:00+00:00',
      expiresAt: '2026-09-11T12:00:00+00:00',
      courseIds: ['css-350-spring-2026-n3h9'],
      token: 'tok',
      path: '/invite/tok',
    });
    renderPage();
    await screen.findByRole('list', { name: 'Accounts' });

    fireEvent.click(screen.getByLabelText('CSS 350'));
    fireEvent.change(screen.getByLabelText(/Note/), { target: { value: 'Dr. Y' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create invitation' }));

    await waitFor(() =>
      expect(api.createInvitation).toHaveBeenCalledWith({
        kind: 'professor',
        courseIds: ['css-350-spring-2026-n3h9'],
        label: 'Dr. Y',
      }),
    );
    expect(await screen.findByText('Instructor invitation created')).toBeInTheDocument();
    expect(screen.getByText(`${window.location.origin}/invite/tok`)).toBeInTheDocument();
  });

  it('an administrator invitation takes no courses', async () => {
    api.createInvitation.mockResolvedValue({
      invitationId: 'inv-2', kind: 'admin', status: 'active', createdAt: 'x', courseIds: [],
      token: 'tok2', path: '/invite/tok2',
    });
    renderPage();
    await screen.findByRole('list', { name: 'Accounts' });

    fireEvent.change(screen.getByLabelText(/Role/), { target: { value: 'admin' } });
    expect(screen.queryByLabelText('CSS 350')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Create invitation' }));

    await waitFor(() => expect(api.createInvitation).toHaveBeenCalledWith({ kind: 'admin' }));
  });

  it('removes and adds course memberships', async () => {
    api.removeMembership.mockResolvedValue({ userId: 'u-prof', courseId: 'css-360-winter-2026-a7rp', member: false });
    api.addMembership.mockResolvedValue({ userId: 'u-prof', courseId: 'css-350-spring-2026-n3h9', member: true });
    renderPage();
    await screen.findByRole('list', { name: 'Accounts' });

    fireEvent.click(screen.getByRole('button', { name: /Remove CSS 360 from Prof Example/ }));
    await waitFor(() =>
      expect(api.removeMembership).toHaveBeenCalledWith('u-prof', 'css-360-winter-2026-a7rp'),
    );

    fireEvent.change(screen.getByLabelText(/Assign a course to Prof Example/), {
      target: { value: 'css-350-spring-2026-n3h9' },
    });
    fireEvent.click(screen.getAllByRole('button', { name: 'Add' })[0]);
    await waitFor(() =>
      expect(api.addMembership).toHaveBeenCalledWith('u-prof', 'css-350-spring-2026-n3h9'),
    );
  });

  it('disables an account only after confirmation, and never offers to disable yourself', async () => {
    api.updateUser.mockResolvedValue({ ...PROF, disabled: true });
    renderPage();
    const list = await screen.findByRole('list', { name: 'Accounts' });

    // One Disable button: the professor's. The signed-in admin has none.
    const disableButtons = within(list).getAllByRole('button', { name: 'Disable' });
    expect(disableButtons).toHaveLength(1);
    fireEvent.click(disableButtons[0]);
    expect(api.updateUser).not.toHaveBeenCalled();

    const dialog = await screen.findByRole('alertdialog');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Disable' }));
    await waitFor(() => expect(api.updateUser).toHaveBeenCalledWith('u-prof', { disabled: true }));
  });

  it('lists invitations without any link and lets an active one be revoked', async () => {
    api.listInvitations.mockResolvedValue({
      count: 1,
      invitations: [{
        invitationId: 'inv-9', kind: 'professor', status: 'active', createdAt: 'x',
        courseIds: ['css-360-winter-2026-a7rp'], label: 'Dr. Z',
      }],
    });
    api.revokeInvitation.mockResolvedValue({ invitationId: 'inv-9', kind: 'professor', status: 'revoked', createdAt: 'x', courseIds: [] });
    renderPage();
    const list = await screen.findByRole('list', { name: 'Invitations' });
    expect(within(list).getByText(/Instructor · Dr. Z/)).toBeInTheDocument();
    expect(within(list).queryByText(/\/invite\//)).not.toBeInTheDocument();

    fireEvent.click(within(list).getByRole('button', { name: 'Revoke' }));
    await waitFor(() => expect(api.revokeInvitation).toHaveBeenCalledWith('inv-9'));
  });
});
