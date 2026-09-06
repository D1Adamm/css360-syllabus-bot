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

vi.mock('./hooks/useCourseActivity', () => ({
  useCourseActivity: () => ({
    status: 'ready',
    activity: { courseId: 'css360-default', contributedQuestions: 0, evaluations: 0 },
  }),
}));

vi.mock('./lib/authApi', async () => {
  const actual = await vi.importActual<typeof import('./lib/authApi')>('./lib/authApi');
  return {
    ...actual,
    fetchSession: vi.fn().mockResolvedValue({ user: null, participant: null }),
    logout: vi.fn().mockResolvedValue(undefined),
  };
});

vi.mock('./lib/inviteApi', async () => {
  const actual = await vi.importActual<typeof import('./lib/inviteApi')>('./lib/inviteApi');
  return {
    ...actual,
    listStudentInvites: vi.fn().mockResolvedValue({ courseId: 'x', count: 0, invites: [] }),
  };
});

vi.mock('./lib/adminPeopleApi', async () => {
  const actual = await vi.importActual<typeof import('./lib/adminPeopleApi')>(
    './lib/adminPeopleApi',
  );
  return {
    ...actual,
    listUsers: vi.fn().mockResolvedValue({ count: 0, users: [] }),
    listInvitations: vi.fn().mockResolvedValue({ count: 0, invitations: [] }),
    listAudit: vi.fn().mockResolvedValue({ count: 0, actions: [] }),
  };
});

import { AppRoutes } from './App';
import { ComparisonRunProvider } from './context/ComparisonRunContext';
import { SessionProvider, type Session } from './context/SessionContext';

const COURSE = 'css360-default';
const OTHER = 'other-course';

/**
 * The sessions the backend could report. `student` is an anonymous
 * participant of the default course; `professor` holds a membership in it;
 * `admin` holds none and needs none.
 */
const SESSIONS: Record<'anonymous' | 'student' | 'professor' | 'admin', Session> = {
  anonymous: { user: null, participant: null },
  student: { user: null, participant: { courseId: COURSE } },
  professor: {
    user: {
      userId: 'u-prof',
      email: 'prof@uw.edu',
      displayName: 'Prof Example',
      role: 'professor',
      courseIds: [COURSE],
    },
    participant: null,
  },
  admin: {
    user: {
      userId: 'u-admin',
      email: 'admin@uw.edu',
      displayName: 'Admin Example',
      role: 'admin',
      courseIds: [],
    },
    participant: null,
  },
};

type Who = keyof typeof SESSIONS;

function LocationProbe() {
  const location = useLocation();
  return (
    <div data-testid="location">{`${location.pathname}${location.search}`}</div>
  );
}

function renderAt(path: string, who: Who = 'student') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <SessionProvider initialSession={SESSIONS[who]}>
        <ComparisonRunProvider>
          <LocationProbe />
          <AppRoutes />
        </ComparisonRunProvider>
      </SessionProvider>
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

async function expectLocation(view: ReturnType<typeof render>, expected: string) {
  await waitFor(() => {
    expect(view.getByTestId('location')).toHaveTextContent(expected);
  });
}

describe('landing', () => {
  it('sends each session to its own home from the root route', async () => {
    const student = renderAt('/', 'student');
    await expectLocation(student, '/student');
    cleanup();

    const professor = renderAt('/', 'professor');
    await expectLocation(professor, '/professor/courses');
    cleanup();

    const admin = renderAt('/', 'admin');
    await expectLocation(admin, '/admin');
  });

  it('sends nobody-in-particular to sign in, which points students at the join page', async () => {
    const view = renderAt('/', 'anonymous');
    await expectLocation(view, '/login');
    expect(await view.findByRole('link', { name: /Enter a class code/ })).toHaveAttribute(
      'href',
      '/join',
    );
  });
});

describe('route guards', () => {
  it('sends an anonymous visitor on a student course page to the join page', async () => {
    const view = renderAt(`/student/course/${COURSE}/compare`, 'anonymous');
    await expectLocation(view, '/join');
  });

  it('lets a participant into the course they joined and not another', async () => {
    const own = renderAt(`/student/course/${COURSE}`, 'student');
    await expectLocation(own, `/student/course/${COURSE}`);
    expect(await own.findByRole('heading', { name: /CSS 360/ })).toBeInTheDocument();
    cleanup();

    const other = renderAt(`/student/course/${OTHER}`, 'student');
    await expectLocation(other, '/forbidden');
  });

  it('sends a participant deep-linking into the professor area to sign in', async () => {
    const view = renderAt('/professor/courses', 'student');
    await expectLocation(view, '/login');
  });

  it('lets a professor into an assigned course and refuses an unassigned one', async () => {
    const assigned = renderAt(`/professor/course/${COURSE}`, 'professor');
    await expectLocation(assigned, `/professor/course/${COURSE}`);
    cleanup();

    const unassigned = renderAt(`/professor/course/${OTHER}`, 'professor');
    await expectLocation(unassigned, '/forbidden');
  });

  it('refuses a professor in the admin area and admits an administrator', async () => {
    const professor = renderAt('/admin', 'professor');
    await expectLocation(professor, '/forbidden');
    cleanup();

    const admin = renderAt('/admin', 'admin');
    await expectLocation(admin, '/admin');
    cleanup();

    const anyCourse = renderAt(`/admin/courses/${OTHER}`, 'admin');
    await expectLocation(anyCourse, `/admin/courses/${OTHER}`);
  });

  it('sends an anonymous visitor in the admin area to sign in', async () => {
    const view = renderAt('/admin/people', 'anonymous');
    await expectLocation(view, '/login');
  });

  it('lets a professor walk the student flow of their own course', async () => {
    const view = renderAt(`/student/course/${COURSE}/compare`, 'professor');
    await expectLocation(view, `/student/course/${COURSE}/compare`);
  });
});

describe('role navigation', () => {
  it('shows only the student sections in the student area', async () => {
    const view = renderAt(`/student/course/${COURSE}`, 'student');

    const nav = await view.findByRole('navigation', { name: 'Main navigation' });
    for (const label of ['Home', 'Contribute', 'Compare', 'Evaluate']) {
      expect(within(nav).getByRole('link', { name: label })).toBeInTheDocument();
    }
    expect(within(nav).queryByRole('link', { name: 'Courses' })).toBeNull();
    expect(within(nav).queryByRole('link', { name: 'System' })).toBeNull();
  });

  it('keeps course links scoped to the active course', async () => {
    const view = renderAt(`/student/course/${COURSE}`, 'student');

    const nav = await view.findByRole('navigation', { name: 'Main navigation' });
    expect(within(nav).getByRole('link', { name: 'Compare' })).toHaveAttribute(
      'href',
      `/student/course/${COURSE}/compare`,
    );
    expect(within(nav).getByRole('link', { name: 'Contribute' })).toHaveAttribute(
      'href',
      `/student/course/${COURSE}/contribute`,
    );
  });

  it('shows only the professor sections in the professor area', async () => {
    const view = renderAt('/professor/courses', 'professor');

    const nav = await view.findByRole('navigation', { name: 'Main navigation' });
    expect(within(nav).getByRole('link', { name: 'Courses' })).toBeInTheDocument();
    expect(within(nav).queryByRole('link', { name: 'Reviews' })).toBeNull();
    expect(within(nav).queryByRole('link', { name: 'Models' })).toBeNull();
    expect(within(nav).queryByRole('link', { name: 'Contribute' })).toBeNull();
  });

  it('uses the admin sidebar and exposes the technical and people sections there', async () => {
    const view = renderAt('/admin', 'admin');

    const nav = await view.findByRole('navigation', { name: 'Admin navigation' });
    for (const label of ['Overview', 'Courses', 'People', 'Training', 'Models', 'Audit', 'System']) {
      expect(within(nav).getByRole('link', { name: label })).toBeInTheDocument();
    }
  });

  it('shows who is signed in and a way out, and no navigation to nobody', async () => {
    const professor = renderAt('/professor/courses', 'professor');
    expect(await professor.findByText('Prof Example')).toBeInTheDocument();
    expect(professor.getByRole('button', { name: 'Sign out' })).toBeInTheDocument();
    cleanup();

    const nobody = renderAt('/login', 'anonymous');
    expect(await nobody.findByRole('link', { name: 'Sign in' })).toBeInTheDocument();
    expect(nobody.queryByRole('navigation', { name: 'Main navigation' })).toBeNull();
  });

  it('never renders a development role switcher', async () => {
    const view = renderAt(`/student/course/${COURSE}`, 'student');
    await view.findByRole('navigation', { name: 'Main navigation' });
    expect(view.queryByLabelText(/Development role/)).toBeNull();
    expect(view.queryByText('DEV')).toBeNull();
  });
});

describe('legacy URL redirects', () => {
  const cases: { from: string; to: string; who?: Who }[] = [
    { from: '/architecture', to: '/admin/system', who: 'admin' },
    { from: '/create-course', to: '/professor/courses/new', who: 'professor' },
    { from: `/course/${COURSE}/seeds`, to: `/student/course/${COURSE}/contribute` },
    { from: `/course/${COURSE}/compare`, to: `/student/course/${COURSE}/compare` },
    { from: `/course/${COURSE}/review`, to: `/professor/course/${COURSE}/examples`, who: 'professor' },
    { from: `/course/${COURSE}/results`, to: `/professor/course/${COURSE}/results`, who: 'professor' },
    { from: `/course/${COURSE}/dataset`, to: `/admin/courses/${COURSE}/examples`, who: 'admin' },
    { from: '/professor/reviews', to: '/professor/courses', who: 'professor' },
    { from: '/professor/models', to: '/professor/courses', who: 'professor' },
    { from: '/seed-builder', to: `/student/course/${COURSE}/contribute` },
    { from: '/compare', to: `/student/course/${COURSE}/compare` },
    { from: '/review', to: `/professor/course/${COURSE}/examples`, who: 'professor' },
    { from: '/dataset', to: `/admin/courses/${COURSE}/examples`, who: 'admin' },
    { from: '/home', to: `/student/course/${COURSE}`, who: 'student' },
    { from: '/home', to: `/professor/course/${COURSE}`, who: 'professor' },
    { from: `/course/${COURSE}`, to: `/student/course/${COURSE}`, who: 'student' },
  ];

  for (const { from, to, who } of cases) {
    it(`redirects ${from} to ${to}${who ? ` as ${who}` : ''}`, async () => {
      const view = renderAt(from, who ?? 'student');
      await expectLocation(view, to);
    });
  }

  it('preserves the query string when redirecting evaluate', async () => {
    const view = renderAt('/evaluate?comparison=comparison-2');
    await expectLocation(view, `/student/course/${COURSE}/evaluate?comparison=comparison-2`);
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
    await expectLocation(view, '/not-found');
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
    const view = renderAt(`/student/course/${COURSE}`, 'student');
    const nav = await view.findByRole('navigation', { name: 'Main navigation' });
    expect(within(nav).queryByRole('link', { name: /architecture/i })).toBeNull();
  });
});
