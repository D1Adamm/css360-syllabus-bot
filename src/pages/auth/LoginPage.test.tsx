/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const loginMock = vi.hoisted(() => vi.fn());
const fetchSessionMock = vi.hoisted(() => vi.fn());

vi.mock('../../lib/authApi', async () => {
  const actual = await vi.importActual<typeof import('../../lib/authApi')>('../../lib/authApi');
  return { ...actual, login: loginMock, fetchSession: fetchSessionMock };
});

import { ApiError } from '../../lib/httpClient';
import { SessionProvider, type Session } from '../../context/SessionContext';
import { LoginPage } from './LoginPage';

const PROFESSOR: Session = {
  user: { userId: 'u', email: 'prof@uw.edu', displayName: 'Prof', role: 'professor', courseIds: [] },
  participant: null,
};

function Probe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderLogin(session: Session = { user: null, participant: null }, state?: unknown) {
  return render(
    <MemoryRouter initialEntries={[{ pathname: '/login', state }]}>
      <SessionProvider initialSession={session}>
        <Probe />
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<p>Landing</p>} />
          <Route path="/professor/courses" element={<p>Courses</p>} />
          <Route path="/professor/course/:courseId/results" element={<p>Results</p>} />
        </Routes>
      </SessionProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  loginMock.mockReset();
  fetchSessionMock.mockReset();
  fetchSessionMock.mockResolvedValue(PROFESSOR);
});

afterEach(() => {
  cleanup();
});

describe('LoginPage', () => {
  it('points students at the class code instead of a password', () => {
    renderLogin();
    expect(screen.getByRole('link', { name: /Enter a class code/ })).toHaveAttribute('href', '/join');
  });

  it('signs in and returns to where the visitor was going', async () => {
    loginMock.mockResolvedValue(undefined);
    renderLogin(undefined, { from: '/professor/course/css-360-winter-2026-a7rp/results' });

    fireEvent.change(screen.getByLabelText(/Email address/), { target: { value: 'Prof@UW.edu ' } });
    fireEvent.change(screen.getByLabelText(/Password/), { target: { value: 'correct horse battery' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));

    await waitFor(() => expect(loginMock).toHaveBeenCalledWith('Prof@UW.edu', 'correct horse battery'));
    await waitFor(() =>
      expect(screen.getByTestId('location')).toHaveTextContent(
        '/professor/course/css-360-winter-2026-a7rp/results',
      ),
    );
  });

  it('shows the backend message for a refused sign-in', async () => {
    loginMock.mockRejectedValue(new ApiError('The email address or password is not correct.', 401));
    renderLogin();

    fireEvent.change(screen.getByLabelText(/Email address/), { target: { value: 'x@uw.edu' } });
    fireEvent.change(screen.getByLabelText(/Password/), { target: { value: 'wrong' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));

    expect(await screen.findByText(/not correct/)).toBeInTheDocument();
    expect(screen.getByTestId('location')).toHaveTextContent('/login');
  });

  it('sends someone already signed in to their home', async () => {
    renderLogin(PROFESSOR);
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/professor/courses'));
    expect(loginMock).not.toHaveBeenCalled();
  });
});
