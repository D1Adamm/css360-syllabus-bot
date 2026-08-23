/** @vitest-environment jsdom */
import { cleanup, render, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const subscribeToCoursesMock = vi.fn();
const subscribeToCourseMetadataMock = vi.fn();

vi.mock('./lib/coursesDb', async () => {
  const actual = await vi.importActual<typeof import('./lib/coursesDb')>(
    './lib/coursesDb',
  );
  return {
    ...actual,
    subscribeToCourses: (...args: unknown[]) => subscribeToCoursesMock(...args),
    subscribeToCourseMetadata: (...args: unknown[]) =>
      subscribeToCourseMetadataMock(...args),
  };
});

vi.mock('./lib/api', async () => {
  const actual = await vi.importActual<typeof import('./lib/api')>('./lib/api');
  return {
    ...actual,
    fetchCourseSyllabusText: vi.fn().mockResolvedValue({
      courseId: 'css360-default',
      text: 'Syllabus body',
      characterCount: 13,
    }),
    listCourseSeeds: vi.fn().mockResolvedValue({
      courseId: 'css360-default',
      count: 0,
      seeds: [],
    }),
    getApprovedExportStatus: vi.fn().mockResolvedValue({
      courseId: 'css360-default',
      exists: false,
      exportPath: '',
      exampleCount: 0,
      sourceFile: '',
    }),
  };
});

vi.mock('./hooks/useSeedExamples', () => ({
  useSeedExamples: () => ({
    seeds: [],
    loading: false,
    error: null,
    saving: false,
    saveError: null,
    addSeed: vi.fn(),
    deleteSeed: vi.fn(),
    deleteAllSeeds: vi.fn(),
    clearSaveError: vi.fn(),
  }),
}));

vi.mock('./hooks/useEvaluations', () => ({
  useEvaluations: () => ({
    evaluations: [],
    loading: false,
    error: null,
    saving: false,
    saveError: null,
    addEvaluation: vi.fn(),
    deleteEvaluation: vi.fn(),
    deleteAllEvaluations: vi.fn(),
    clearSaveError: vi.fn(),
  }),
}));

import { AppRoutes } from './App';
import { ComparisonRunProvider } from './context/ComparisonRunContext';
import { RoleProvider, type Role } from './context/RoleContext';

function LocationProbe() {
  const location = useLocation();
  return (
    <div data-testid="location">{`${location.pathname}${location.search}`}</div>
  );
}

function renderAt(path: string, role: Role = 'student') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <RoleProvider initialRole={role}>
        <ComparisonRunProvider>
          <LocationProbe />
          <AppRoutes />
        </ComparisonRunProvider>
      </RoleProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  subscribeToCoursesMock.mockReset();
  subscribeToCoursesMock.mockImplementation((onData: (value: unknown[]) => void) => {
    onData([]);
    return () => {};
  });

  subscribeToCourseMetadataMock.mockReset();
  subscribeToCourseMetadataMock.mockImplementation(
    (_courseId: string, onData: (value: unknown) => void) => {
      onData({
        name: 'CSS 360',
        title: 'Software Engineering',
        term: 'Winter 2026',
        instructorName: '',
        createdAt: '2026-01-01T00:00:00.000Z',
        syllabusStatus: 'indexed',
        syllabusFileName: 'syllabus.pdf',
        syllabusType: 'pdf',
        chunkCount: 12,
      });
      return () => {};
    },
  );
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('role landing', () => {
  it('sends each role to its own home from the root route', async () => {
    const student = renderAt('/', 'student');
    await waitFor(() => {
      expect(student.getByTestId('location')).toHaveTextContent('/student');
    });
    cleanup();

    const professor = renderAt('/', 'professor');
    await waitFor(() => {
      expect(professor.getByTestId('location')).toHaveTextContent(
        '/professor/courses',
      );
    });
    cleanup();

    const admin = renderAt('/', 'admin');
    await waitFor(() => {
      expect(admin.getByTestId('location')).toHaveTextContent('/admin');
    });
  });
});

describe('role navigation', () => {
  it('shows only the student sections in the student area', async () => {
    const view = renderAt('/student/course/css360-default', 'student');

    const nav = await view.findByRole('navigation', { name: 'Main navigation' });
    for (const label of ['Home', 'Contribute', 'Compare', 'Evaluate']) {
      expect(within(nav).getByRole('link', { name: label })).toBeInTheDocument();
    }
    expect(within(nav).queryByRole('link', { name: 'Courses' })).toBeNull();
    expect(within(nav).queryByRole('link', { name: 'System' })).toBeNull();
  });

  it('keeps course links scoped to the active course', async () => {
    const view = renderAt('/student/course/other-course', 'student');

    const nav = await view.findByRole('navigation', { name: 'Main navigation' });
    expect(within(nav).getByRole('link', { name: 'Compare' })).toHaveAttribute(
      'href',
      '/student/course/other-course/compare',
    );
    expect(within(nav).getByRole('link', { name: 'Contribute' })).toHaveAttribute(
      'href',
      '/student/course/other-course/contribute',
    );
  });

  it('shows only the professor sections in the professor area', async () => {
    const view = renderAt('/professor/courses', 'professor');

    const nav = await view.findByRole('navigation', { name: 'Main navigation' });
    expect(within(nav).getByRole('link', { name: 'Courses' })).toBeInTheDocument();
    // Everything a professor does is scoped to a course, so Courses is the only
    // top-level destination; the old cross-course hubs are gone.
    expect(within(nav).queryByRole('link', { name: 'Reviews' })).toBeNull();
    expect(within(nav).queryByRole('link', { name: 'Models' })).toBeNull();
    expect(within(nav).queryByRole('link', { name: 'Contribute' })).toBeNull();
  });

  it('uses the admin sidebar and exposes the technical sections there', async () => {
    const view = renderAt('/admin', 'admin');

    const nav = await view.findByRole('navigation', { name: 'Admin navigation' });
    for (const label of ['Overview', 'Courses', 'Training', 'Models', 'System']) {
      expect(within(nav).getByRole('link', { name: label })).toBeInTheDocument();
    }
  });

  it('derives the shell from the URL, not the remembered development role', async () => {
    // A professor deep link opened while the switcher says "student".
    const view = renderAt('/professor/courses', 'student');

    const nav = await view.findByRole('navigation', { name: 'Main navigation' });
    expect(within(nav).getByRole('link', { name: 'Courses' })).toBeInTheDocument();
    expect(within(nav).queryByRole('link', { name: 'Evaluate' })).toBeNull();
  });
});

describe('legacy URL redirects', () => {
  const cases: { from: string; to: string; role?: Role }[] = [
    { from: '/architecture', to: '/admin/system' },
    { from: '/create-course', to: '/professor/courses/new' },
    { from: '/course/css360-default/seeds', to: '/student/course/css360-default/contribute' },
    { from: '/course/css360-default/compare', to: '/student/course/css360-default/compare' },
    { from: '/course/css360-default/review', to: '/professor/course/css360-default/examples' },
    { from: '/course/css360-default/results', to: '/professor/course/css360-default/results' },
    { from: '/course/css360-default/dataset', to: '/admin/courses/css360-default/examples' },
    { from: '/professor/reviews', to: '/professor/courses', role: 'professor' },
    { from: '/professor/models', to: '/professor/courses', role: 'professor' },
    { from: '/seed-builder', to: '/student/course/css360-default/contribute' },
    { from: '/compare', to: '/student/course/css360-default/compare' },
    { from: '/review', to: '/professor/course/css360-default/examples' },
    { from: '/dataset', to: '/admin/courses/css360-default/examples' },
    { from: '/home', to: '/student/course/css360-default', role: 'student' },
    { from: '/home', to: '/professor/course/css360-default', role: 'professor' },
    { from: '/course/css360-default', to: '/student/course/css360-default', role: 'student' },
  ];

  for (const { from, to, role } of cases) {
    it(`redirects ${from} to ${to}${role ? ` as ${role}` : ''}`, async () => {
      const view = renderAt(from, role ?? 'student');
      await waitFor(() => {
        expect(view.getByTestId('location')).toHaveTextContent(to);
      });
    });
  }

  it('preserves the query string when redirecting evaluate', async () => {
    const view = renderAt('/evaluate?comparison=comparison-2');
    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent(
        '/student/course/css360-default/evaluate?comparison=comparison-2',
      );
    });
  });
});

describe('course id validation', () => {
  it('rejects an unsafe course id instead of rendering a course page', async () => {
    const view = renderAt('/student/course/Bad_Id/compare');
    expect(
      await view.findByText(/We couldn't find that course/i),
    ).toBeInTheDocument();
  });

  it('sends a legacy URL with an unsafe course id to the not-found page', async () => {
    const view = renderAt('/course/Bad_Id/compare');
    await waitFor(() => {
      expect(view.getByTestId('location')).toHaveTextContent('/not-found');
    });
  });
});

describe('technical surfaces', () => {
  it('keeps the architecture reference inside the admin area', async () => {
    const view = renderAt('/admin/system', 'admin');
    expect(
      await view.findByRole('heading', { name: 'Architecture' }),
    ).toBeInTheDocument();
  });

  it('does not offer the architecture page in student navigation', async () => {
    const view = renderAt('/student/course/css360-default', 'student');
    const nav = await view.findByRole('navigation', { name: 'Main navigation' });
    expect(within(nav).queryByRole('link', { name: /architecture/i })).toBeNull();
  });
});
