/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const { createCourseMock } = vi.hoisted(() => ({
  createCourseMock: vi.fn(),
}));

vi.mock('../lib/firebase', () => ({
  app: {},
  database: { name: 'mock-db' },
}));

vi.mock('../lib/createCourse', () => ({
  createCourse: createCourseMock,
}));

import { CreateCoursePage } from '../pages/CreateCoursePage';

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderCreateCoursePage() {
  return render(
    <MemoryRouter initialEntries={['/create-course']}>
      <Routes>
        <Route path="/create-course" element={<CreateCoursePage />} />
        <Route path="/course/:courseId/home" element={<div>Course home</div>} />
      </Routes>
      <LocationProbe />
    </MemoryRouter>,
  );
}

describe('CreateCoursePage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('requires name, title, and term before saving', async () => {
    renderCreateCoursePage();

    fireEvent.click(screen.getByRole('button', { name: 'Create course' }));

    expect(await screen.findByText('Course name or code is required.')).toBeInTheDocument();
    expect(screen.getByText('Course title is required.')).toBeInTheDocument();
    expect(screen.getByText('Term is required.')).toBeInTheDocument();
    expect(createCourseMock).not.toHaveBeenCalled();
  });

  it('saves course metadata and redirects to /course/{courseId}/home', async () => {
    createCourseMock.mockResolvedValue({
      courseId: 'css-430-summer-2026-a82f',
      metadata: {
        name: 'CSS 430',
        title: 'Operating Systems',
        term: 'Summer 2026',
        instructorName: '',
        createdAt: '2026-01-01T00:00:00.000Z',
        syllabusStatus: 'not_uploaded',
        syllabusFileName: '',
        syllabusType: '',
        chunkCount: 0,
      },
    });

    const view = renderCreateCoursePage();

    fireEvent.change(screen.getByLabelText(/Course name or code/), {
      target: { value: 'CSS 430' },
    });
    fireEvent.change(screen.getByLabelText(/Course title/), {
      target: { value: 'Operating Systems' },
    });
    fireEvent.change(screen.getByLabelText(/^Term/), {
      target: { value: 'Summer 2026' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Create course' }));

    await waitFor(() => {
      expect(createCourseMock).toHaveBeenCalledWith({
        name: 'CSS 430',
        title: 'Operating Systems',
        term: 'Summer 2026',
        instructorName: '',
      });
    });

    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        '/course/css-430-summer-2026-a82f/home',
      );
    });
  });

  it('displays Firebase save errors', async () => {
    createCourseMock.mockRejectedValue(new Error('Firebase permission denied'));

    renderCreateCoursePage();

    fireEvent.change(screen.getByLabelText(/Course name or code/), {
      target: { value: 'CSS 430' },
    });
    fireEvent.change(screen.getByLabelText(/Course title/), {
      target: { value: 'Operating Systems' },
    });
    fireEvent.change(screen.getByLabelText(/^Term/), {
      target: { value: 'Summer 2026' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Create course' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Firebase permission denied',
    );
    expect(screen.getByTestId('location')).toHaveTextContent('/create-course');
  });
});
