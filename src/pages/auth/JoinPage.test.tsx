/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const joinCourseMock = vi.hoisted(() => vi.fn());
const fetchSessionMock = vi.hoisted(() => vi.fn());

vi.mock('../../lib/authApi', async () => {
  const actual = await vi.importActual<typeof import('../../lib/authApi')>('../../lib/authApi');
  return { ...actual, joinCourse: joinCourseMock, fetchSession: fetchSessionMock };
});

import { ApiError } from '../../lib/httpClient';
import { SessionProvider } from '../../context/SessionContext';
import { JoinPage } from './JoinPage';

const COURSE = 'css-360-winter-2026-a7rp';

function Probe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <SessionProvider initialSession={{ user: null, participant: null }}>
        <Probe />
        <Routes>
          <Route path="/join" element={<JoinPage />} />
          <Route path="/join/:code" element={<JoinPage />} />
          <Route path="/student/course/:courseId" element={<p>Course home</p>} />
        </Routes>
      </SessionProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  joinCourseMock.mockReset();
  fetchSessionMock.mockReset();
  fetchSessionMock.mockResolvedValue({ user: null, participant: { courseId: COURSE } });
});

afterEach(() => {
  cleanup();
});

describe('JoinPage', () => {
  it('asks for a class code and nothing personal', () => {
    renderAt('/join');
    expect(screen.getByLabelText(/Class code/)).toBeInTheDocument();
    expect(screen.queryByLabelText(/name/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/email/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/password/i)).not.toBeInTheDocument();
  });

  it('upper-cases what is typed, joins, and lands in the course', async () => {
    joinCourseMock.mockResolvedValue({ courseId: COURSE, courseName: 'CSS 360', alreadyJoined: false });
    renderAt('/join');

    const input = screen.getByLabelText(/Class code/);
    fireEvent.change(input, { target: { value: '7k4p9x' } });
    expect(input).toHaveValue('7K4P9X');
    fireEvent.click(screen.getByRole('button', { name: 'Join' }));

    await waitFor(() => expect(joinCourseMock).toHaveBeenCalledWith('7K4P9X'));
    await waitFor(() =>
      expect(screen.getByTestId('location')).toHaveTextContent(`/student/course/${COURSE}`),
    );
    expect(fetchSessionMock).toHaveBeenCalled();
  });

  it('a direct link joins on arrival without a click', async () => {
    joinCourseMock.mockResolvedValue({ courseId: COURSE, alreadyJoined: true });
    renderAt('/join/7K4P9X');

    await waitFor(() => expect(joinCourseMock).toHaveBeenCalledWith('7K4P9X'));
    await waitFor(() =>
      expect(screen.getByTestId('location')).toHaveTextContent(`/student/course/${COURSE}`),
    );
  });

  it('shows the neutral message for a code the backend refuses', async () => {
    joinCourseMock.mockRejectedValue(
      new ApiError("That code isn't valid right now. Check it with your instructor and try again.", 404),
    );
    renderAt('/join');
    fireEvent.change(screen.getByLabelText(/Class code/), { target: { value: 'ZZZZZZ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Join' }));

    expect(await screen.findByText(/isn't valid right now/)).toBeInTheDocument();
    expect(screen.getByTestId('location')).toHaveTextContent('/join');
  });

  it('explains a throttle rather than a refusal', async () => {
    joinCourseMock.mockRejectedValue(new ApiError('Too many attempts.', 429));
    renderAt('/join');
    fireEvent.change(screen.getByLabelText(/Class code/), { target: { value: 'ZZZZZZ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Join' }));

    expect(await screen.findByText(/Too many attempts. Wait a minute/)).toBeInTheDocument();
  });

  it('does not submit an empty code', () => {
    renderAt('/join');
    fireEvent.click(screen.getByRole('button', { name: 'Join' }));
    expect(within(screen.getByRole('alert')).getByText(/Enter the class code/)).toBeInTheDocument();
    expect(joinCourseMock).not.toHaveBeenCalled();
  });
});
