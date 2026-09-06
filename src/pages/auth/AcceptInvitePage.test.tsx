/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const previewMock = vi.hoisted(() => vi.fn());
const acceptMock = vi.hoisted(() => vi.fn());

vi.mock('../../lib/authApi', async () => {
  const actual = await vi.importActual<typeof import('../../lib/authApi')>('../../lib/authApi');
  return { ...actual, previewInvitation: previewMock, acceptInvitation: acceptMock };
});

import { ApiError } from '../../lib/httpClient';
import { SessionProvider } from '../../context/SessionContext';
import { AcceptInvitePage } from './AcceptInvitePage';

const TOKEN = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG';
const COURSE = 'css-360-winter-2026-a7rp';

function Probe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderInvite() {
  return render(
    <MemoryRouter initialEntries={[`/invite/${TOKEN}`]}>
      <SessionProvider initialSession={{ user: null, participant: null }}>
        <Probe />
        <Routes>
          <Route path="/invite/:token" element={<AcceptInvitePage />} />
          <Route path="/professor/courses" element={<p>Courses</p>} />
          <Route path="/admin" element={<p>Admin</p>} />
        </Routes>
      </SessionProvider>
    </MemoryRouter>,
  );
}

function fill(overrides: Partial<Record<'email' | 'name' | 'password' | 'confirm', string>> = {}) {
  const values = {
    email: 'prof@uw.edu',
    name: 'Prof Example',
    password: 'a long enough passphrase',
    confirm: 'a long enough passphrase',
    ...overrides,
  };
  fireEvent.change(screen.getByLabelText(/Email address/), { target: { value: values.email } });
  fireEvent.change(screen.getByLabelText(/Your name/), { target: { value: values.name } });
  fireEvent.change(screen.getByLabelText(/^Password/), { target: { value: values.password } });
  fireEvent.change(screen.getByLabelText(/Confirm password/), { target: { value: values.confirm } });
}

beforeEach(() => {
  previewMock.mockReset();
  acceptMock.mockReset();
  previewMock.mockResolvedValue({
    kind: 'professor',
    courses: [{ courseId: COURSE, name: 'CSS 360' }],
    expiresAt: '2026-09-11T12:00:00+00:00',
  });
});

afterEach(() => {
  cleanup();
});

describe('AcceptInvitePage', () => {
  it('says what accepting will do before asking for anything', async () => {
    renderInvite();
    expect(await screen.findByText(/instructor account for CSS 360/)).toBeInTheDocument();
    expect(previewMock).toHaveBeenCalledWith(TOKEN);
  });

  it('creates the account, signs in, and lands on the role home', async () => {
    acceptMock.mockResolvedValue({
      user: { userId: 'u', email: 'prof@uw.edu', displayName: 'Prof Example', role: 'professor', courseIds: [COURSE] },
      participant: null,
    });
    renderInvite();
    await screen.findByText(/instructor account for CSS 360/);

    fill();
    fireEvent.click(screen.getByRole('button', { name: /Create account and sign in/ }));

    await waitFor(() =>
      expect(acceptMock).toHaveBeenCalledWith(TOKEN, {
        password: 'a long enough passphrase',
        email: 'prof@uw.edu',
        displayName: 'Prof Example',
      }),
    );
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/professor/courses'));
  });

  it('refuses a short or mismatched password before calling the backend', async () => {
    renderInvite();
    await screen.findByRole('button', { name: /Create account/ });

    fill({ password: 'short', confirm: 'short' });
    fireEvent.click(screen.getByRole('button', { name: /Create account/ }));
    expect(await screen.findByText(/Use at least 12 characters/)).toBeInTheDocument();

    fill({ confirm: 'a different passphrase!' });
    fireEvent.click(screen.getByRole('button', { name: /Create account/ }));
    expect(await screen.findByText(/do not match/)).toBeInTheDocument();
    expect(acceptMock).not.toHaveBeenCalled();
  });

  it('explains a link that cannot be used', async () => {
    previewMock.mockRejectedValue(new ApiError('This invitation link is not valid.', 404));
    renderInvite();
    expect(await screen.findByText(/This invitation can't be used/)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Password/)).not.toBeInTheDocument();
  });

  it('a reset invitation asks only for the new password', async () => {
    previewMock.mockResolvedValue({ kind: 'reset', courses: [], targetEmail: 'prof@uw.edu' });
    acceptMock.mockResolvedValue({
      user: { userId: 'u', email: 'prof@uw.edu', displayName: 'Prof', role: 'admin', courseIds: [] },
      participant: null,
    });
    renderInvite();
    expect(await screen.findByText(/For the account prof@uw.edu/)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Email address/)).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/^Password/), { target: { value: 'a brand new passphrase' } });
    fireEvent.change(screen.getByLabelText(/Confirm password/), { target: { value: 'a brand new passphrase' } });
    fireEvent.click(screen.getByRole('button', { name: /Set password and sign in/ }));

    await waitFor(() => expect(acceptMock).toHaveBeenCalledWith(TOKEN, { password: 'a brand new passphrase' }));
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/admin'));
  });
});
