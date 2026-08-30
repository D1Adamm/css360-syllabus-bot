/** @vitest-environment jsdom */
import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * The admin detail page's Model state section.
 *
 * It used to be a fixed sentence — "No per-course model registry exists, so
 * this cannot be answered" — beside an `unknown` pill. The registry has existed
 * since the `course_models` tables landed, and Admin Models, Admin Training and
 * the professor Model page were all reading it, so this one page reported
 * `unknown` for a course whose current version was ready and online. The
 * regression these tests hold is that a course with a current registered
 * version never displays `unknown` here.
 */

let courseId = 'css-350-spring-2026-n3h9';
vi.mock('../../context/CourseContext', () => ({
  useCourseId: () => courseId,
}));

vi.mock('../../hooks/useCourseMetadata', () => ({
  useCourseMetadata: () => ({
    state: { status: 'ready', metadata: METADATA } as const,
    metadata: METADATA,
    retry: vi.fn(),
  }),
}));

vi.mock('../../hooks/useCourseExampleCounts', () => ({
  useCourseExampleCounts: () => ({
    status: 'ready',
    counts: { total: 42, approved: 42, pending: 0, rejected: 0, edited: 0 },
  }),
}));

const modelStateByCourse = new Map<string, CourseModelState>();
const requestStateByCourse = new Map<string, CourseModelRequestState>();

vi.mock('../../hooks/useCourseModel', () => ({
  useCourseModel: (id: string | null) => ({
    state: (id && modelStateByCourse.get(id)) || ({ status: 'none' } as const),
    retry: vi.fn(),
  }),
}));

vi.mock('../../hooks/useCourseModelRequest', () => ({
  useCourseModelRequest: (id: string | null) => ({
    state: (id && requestStateByCourse.get(id)) || ({ status: 'none' } as const),
    submitting: false,
    submitError: null,
    submit: vi.fn(),
    clearSubmitError: vi.fn(),
  }),
}));

vi.mock('../../lib/adminApi', () => ({
  ApiError: class ApiError extends Error {},
  fetchCourseChunks: vi.fn(),
  fetchFactInventory: vi.fn(),
  runSeedQualityCheck: vi.fn(),
}));

import type { CourseModelState } from '../../hooks/useCourseModel';
import type { CourseModelRequestState } from '../../hooks/useCourseModelRequest';
import type {
  CourseMetadata,
  CourseModelRegistry,
  CourseModelVersion,
} from '../../types';
import { AdminCourseDetailPage } from './AdminCourseDetailPage';

const METADATA: CourseMetadata = {
  name: 'CSS 350',
  title: 'Management Principles',
  term: 'Spring 2026',
  instructorName: '',
  createdAt: '2026-02-01T09:00:00.000Z',
  syllabusStatus: 'indexed',
  syllabusFileName: 'syllabus.pdf',
  syllabusType: 'pdf',
  chunkCount: 12,
};

/** CSS 350 as the deployed smoke test found it: v2, ready, online. */
const V2: CourseModelVersion = {
  version: 'v2',
  baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
  trainingExampleCount: 37,
  status: 'ready',
  deployment: 'online',
  artifactRef: 'serving/css-350-spring-2026-n3h9/v2/adapter',
  createdAt: '2026-08-27T20:53:10.000Z',
  runId: 'run-20260827t205310z-8c3cdb',
};

const V1: CourseModelVersion = {
  ...V2,
  version: 'v1',
  deployment: 'offline',
  trainingExampleCount: 42,
  artifactRef: 'serving/css-350-spring-2026-n3h9/v1/adapter',
  createdAt: '2026-08-09T23:45:00.000Z',
  runId: undefined,
};

function registry(
  currentVersion: string,
  versions: CourseModelVersion[],
): CourseModelRegistry {
  return {
    currentVersion,
    versions: Object.fromEntries(versions.map((v) => [v.version, v])),
  };
}

function renderPage() {
  return render(
    <MemoryRouter>
      <AdminCourseDetailPage />
    </MemoryRouter>,
  );
}

function modelSection(): HTMLElement {
  return screen.getByRole('list', { name: 'Model state' });
}

describe('AdminCourseDetailPage model state', () => {
  beforeEach(() => {
    courseId = 'css-350-spring-2026-n3h9';
    modelStateByCourse.clear();
    requestStateByCourse.clear();
  });

  afterEach(() => {
    cleanup();
  });

  it('does not display unknown for a course with a current registered version', () => {
    modelStateByCourse.set(courseId, {
      status: 'ready',
      registry: registry('v2', [V1, V2]),
    });

    renderPage();

    const section = modelSection();
    expect(section).not.toHaveTextContent('unknown');
    expect(section).not.toHaveTextContent('No per-course model registry exists');
    expect(section).toHaveTextContent('Ready · published');
    expect(section).toHaveTextContent('v2');
  });

  it('reports the same state the registry records for a registered but unpublished model', () => {
    modelStateByCourse.set(courseId, {
      status: 'ready',
      registry: registry('v1', [V1]),
    });

    renderPage();

    const section = modelSection();
    expect(section).toHaveTextContent('Ready · not published');
    expect(section).not.toHaveTextContent('unknown');
  });

  it('uses the same logic for any other course, with no special case', () => {
    courseId = 'css-360-winter-2026-a7rp';
    modelStateByCourse.set(courseId, {
      status: 'ready',
      registry: registry('v2', [{ ...V2, deployment: 'offline' }]),
    });

    renderPage();

    expect(modelSection()).toHaveTextContent('Ready · not published');
    expect(modelSection()).not.toHaveTextContent('unknown');
  });

  it('says a course has no registered version rather than inventing one', () => {
    modelStateByCourse.set(courseId, { status: 'none' });

    renderPage();

    const section = modelSection();
    expect(section).toHaveTextContent('No model version is registered');
    expect(section).toHaveTextContent('Not created yet');
  });

  it('does not claim a course has no model while the registry is still loading', () => {
    modelStateByCourse.set(courseId, { status: 'loading' });

    renderPage();

    expect(modelSection()).toHaveTextContent('Checking…');
    expect(modelSection()).not.toHaveTextContent('Not created yet');
  });

  it('separates a registry that could not be read from a course with no model', () => {
    modelStateByCourse.set(courseId, { status: 'unavailable', message: 'network' });

    renderPage();

    const section = modelSection();
    expect(section).toHaveTextContent('Temporarily unavailable');
    expect(section).not.toHaveTextContent('Not created yet');
  });

  it('shows the version history a course has accumulated', () => {
    modelStateByCourse.set(courseId, {
      status: 'ready',
      registry: registry('v2', [V1, V2]),
    });

    renderPage();

    const section = modelSection();
    expect(section).toHaveTextContent('v1');
    expect(section).toHaveTextContent('current');
  });
});
