/** @vitest-environment jsdom */
import { cleanup, render, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

vi.mock('./lib/firebase', () => ({
  app: {},
  database: { name: 'mock-db' },
}));

vi.mock('firebase/app', () => ({
  initializeApp: () => ({}),
}));

vi.mock('firebase/database', () => ({
  getDatabase: () => ({}),
  ref: (_db: unknown, path: string) => ({ path }),
  onValue: vi.fn(() => () => undefined),
  push: vi.fn(() => ({ key: 'generated-id' })),
  set: vi.fn(async () => undefined),
  remove: vi.fn(async () => undefined),
  get: vi.fn(async () => ({ exists: () => false, val: () => null })),
  update: vi.fn(async () => undefined),
}));

vi.mock('./lib/api', () => ({
  ApiError: class ApiError extends Error {},
  generateBaseModel: vi.fn(),
  generateFineTuned: vi.fn(),
  generateFineTunedRag: vi.fn(),
  generateRag: vi.fn(),
  fetchCourseSyllabusText: vi.fn(async () => ({
    courseId: 'css360-default',
    text: 'Mock syllabus text for route tests.',
    characterCount: 34,
  })),
  listCourseSeeds: vi.fn(async () => ({
    courseId: 'css360-default',
    count: 0,
    firebasePath: 'courses/css360-default/seedExamples',
    seeds: [],
  })),
  reviewCourseSeed: vi.fn(),
  exportApprovedCourseSeeds: vi.fn(),
  getApprovedExportStatus: vi.fn(async () => ({
    courseId: 'css360-default',
    exists: false,
    exportPath: 'data/exports/css360-default/approved-finetune.jsonl',
    exampleCount: 0,
    sourceFile: 'approved-finetune.jsonl',
  })),
  prepareTrainingSplit: vi.fn(),
}));

const subscribeToCoursesMock = vi.hoisted(() =>
  vi.fn((onData: (courses: unknown[]) => void) => {
    onData([]);
    return () => undefined;
  }),
);

vi.mock('./lib/coursesDb', async () => {
  const actual = await vi.importActual<typeof import('./lib/coursesDb')>(
    './lib/coursesDb',
  );
  return {
    ...actual,
    subscribeToCourses: subscribeToCoursesMock,
  };
});

import { AppRoutes } from './App';
import { DEFAULT_COURSE_ID } from './lib/courseId';

function LocationProbe() {
  const location = useLocation();
  return (
    <div data-testid="location">
      {location.pathname}
      {location.search}
    </div>
  );
}

function renderApp(initialEntry: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <AppRoutes />
      <LocationProbe />
    </MemoryRouter>,
  );
}

describe('course-specific routes', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    subscribeToCoursesMock.mockImplementation((onData: (courses: unknown[]) => void) => {
      onData([]);
      return () => undefined;
    });
  });

  afterEach(() => {
    cleanup();
  });

  it('renders the course picker on the root route instead of redirecting', async () => {
    const view = renderApp('/');

    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent('/');
    });
    expect(view.getByRole('heading', { name: 'Courses' })).toBeInTheDocument();
    const main = view.container.querySelector('main');
    expect(main).not.toBeNull();
    expect(
      within(main as HTMLElement).getByRole('link', { name: 'Create Course' }),
    ).toHaveAttribute('href', '/create-course');
    expect(view.getByTestId('location').textContent).toBe('/');
  });

  it('redirects /course/css360-default to /course/css360-default/home', async () => {
    const view = renderApp(`/course/${DEFAULT_COURSE_ID}`);

    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        `/course/${DEFAULT_COURSE_ID}/home`,
      );
    });
    expect(view.getByRole('heading', { name: 'Syllabus Model Lab' })).toBeInTheDocument();
  });

  it('renders existing page content on a valid course route', async () => {
    const view = renderApp(`/course/${DEFAULT_COURSE_ID}/syllabus`);

    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        `/course/${DEFAULT_COURSE_ID}/syllabus`,
      );
    });
    expect(view.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });

  it('preserves courseId in course navigation links', async () => {
    const view = renderApp(`/course/${DEFAULT_COURSE_ID}/home`);

    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        `/course/${DEFAULT_COURSE_ID}/home`,
      );
    });

    const nav = view.getByRole('navigation', { name: 'Main navigation' });
    expect(within(nav).getByRole('link', { name: 'Compare' })).toHaveAttribute(
      'href',
      `/course/${DEFAULT_COURSE_ID}/compare`,
    );
    expect(within(nav).getByRole('link', { name: 'Build Seeds' })).toHaveAttribute(
      'href',
      `/course/${DEFAULT_COURSE_ID}/seeds`,
    );
    expect(within(nav).getByRole('link', { name: 'Architecture' })).toHaveAttribute(
      'href',
      '/architecture',
    );
  });

  it('redirects legacy routes to the default course', async () => {
    const view = renderApp('/compare');

    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        `/course/${DEFAULT_COURSE_ID}/compare`,
      );
    });
  });

  it('redirects legacy /seed-builder to /course/css360-default/seeds', async () => {
    const view = renderApp('/seed-builder');

    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        `/course/${DEFAULT_COURSE_ID}/seeds`,
      );
    });
  });

  it('preserves query strings when redirecting legacy evaluate', async () => {
    const view = renderApp('/evaluate?comparison=comparison-001');

    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        `/course/${DEFAULT_COURSE_ID}/evaluate?comparison=comparison-001`,
      );
    });
  });

  it('redirects /home to the default course home', async () => {
    const view = renderApp('/home');

    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        `/course/${DEFAULT_COURSE_ID}/home`,
      );
    });
  });

  it('shows an invalid course error for unsafe course ids', async () => {
    const view = renderApp('/course/Bad_Id/home');

    await waitFor(() => {
      expect(view.getByRole('heading', { name: 'Invalid Course' })).toBeInTheDocument();
    });
    expect(
      view.getByText(/The course id "Bad_Id" is not valid/),
    ).toBeInTheDocument();
    expect(view.getByRole('link', { name: 'Back to Courses' })).toHaveAttribute(
      'href',
      '/',
    );
  });

  it('keeps Architecture outside the course route tree', async () => {
    const view = renderApp('/architecture');

    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent('/architecture');
    });
    expect(view.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });
});
