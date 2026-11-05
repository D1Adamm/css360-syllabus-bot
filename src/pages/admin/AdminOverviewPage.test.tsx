/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/*
 * Course model readiness on the admin Overview (audit finding B7, 2026-11-03).
 *
 * The service probes say whether the fine-tuned service answers; they cannot
 * say whether the version a course has activated is loaded on it. That is the
 * failure students actually meet — "temporarily unavailable" for one course
 * while another works — and until now nothing in the UI showed it.
 */

const fetchBackendHealth = vi.fn();
const fetchFineTunedHealth = vi.fn();
const fetchStarterGenerationStatus = vi.fn();
const fetchModelReadiness = vi.fn();

vi.mock('../../lib/adminApi', () => ({
  fetchBackendHealth: (...args: unknown[]) => fetchBackendHealth(...args),
  fetchFineTunedHealth: (...args: unknown[]) => fetchFineTunedHealth(...args),
  fetchStarterGenerationStatus: (...args: unknown[]) => fetchStarterGenerationStatus(...args),
  fetchModelReadiness: (...args: unknown[]) => fetchModelReadiness(...args),
  getConfiguredApiBaseUrl: () => 'http://localhost:8000/api',
}));

vi.mock('../../lib/coursesDb', () => ({
  subscribeToCourses: (onData: (courses: unknown[]) => void) => {
    onData([]);
    return () => undefined;
  },
}));

import { AdminOverviewPage } from './AdminOverviewPage';

const READINESS = {
  mode: 'vm',
  service: { reachable: true, status: 'ok', ok: true, error: null, errorCode: null },
  courses: [
    {
      courseId: 'css-360-winter-2026-a7rp',
      courseName: 'CSS 360',
      activatedVersion: 'v2',
      newestVersion: 'v3',
      newestStatus: 'ready',
      servableVersions: ['v2', 'v3'],
      problem: null,
      message: null,
    },
    {
      courseId: 'css-350-winter-2026-drlb',
      courseName: 'CSS 350',
      activatedVersion: 'v1',
      newestVersion: 'v1',
      newestStatus: 'ready',
      servableVersions: [],
      problem: 'activated_version_not_servable',
      message: 'Activated version v1 is not loaded by the fine-tuned service.',
    },
    {
      courseId: 'css-490-spring-2026-cgvl',
      courseName: 'CSS 490',
      activatedVersion: null,
      newestVersion: 'v1',
      newestStatus: 'ready',
      servableVersions: ['v1'],
      problem: 'no_activated_version',
      message: 'No version is activated; students get "not available yet".',
    },
  ],
  problems: [
    {
      courseId: 'css-350-winter-2026-drlb',
      problem: 'activated_version_not_servable',
      message: 'Activated version v1 is not loaded by the fine-tuned service.',
    },
  ],
  ok: false,
};

beforeEach(() => {
  vi.clearAllMocks();
  fetchBackendHealth.mockResolvedValue({ status: 'ok', service: 'syllabus-bot' });
  fetchFineTunedHealth.mockResolvedValue({ status: 'ok', adapterLoaded: true, model: 'm' });
  fetchStarterGenerationStatus.mockResolvedValue({ active: false });
  fetchModelReadiness.mockResolvedValue(READINESS);
});

afterEach(() => {
  cleanup();
});

describe('AdminOverviewPage course model readiness', () => {
  it('shows, per course, the activated version and whether the service can serve it', async () => {
    render(<AdminOverviewPage />);

    const list = await screen.findByRole('list', { name: 'Course model readiness' });
    const rows = within(list).getAllByRole('listitem');
    expect(rows).toHaveLength(4);

    expect(rows[1]).toHaveTextContent('CSS 360');
    expect(rows[1]).toHaveTextContent('active v2');
    expect(rows[1]).toHaveTextContent('newest v3 (ready)');
    expect(rows[1]).toHaveTextContent('serving');

    expect(rows[2]).toHaveTextContent('CSS 350');
    expect(rows[2]).toHaveTextContent('not loaded');
    expect(rows[2]).toHaveTextContent('Activated version v1 is not loaded');

    expect(rows[3]).toHaveTextContent('CSS 490');
    expect(rows[3]).toHaveTextContent('no activated version');
    expect(rows[3]).toHaveTextContent('not activated');
  });

  it('summarises how many courses cannot get Fine-Tuned answers', async () => {
    render(<AdminOverviewPage />);

    expect(await screen.findByRole('alert')).toHaveTextContent(
      '1 course cannot get Fine-Tuned answers right now',
    );
  });

  it('says nothing is wrong when every activated version is loaded', async () => {
    fetchModelReadiness.mockResolvedValue({
      ...READINESS,
      courses: [READINESS.courses[0]],
      problems: [],
      ok: true,
    });
    render(<AdminOverviewPage />);

    await screen.findByRole('list', { name: 'Course model readiness' });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('reads the check as not performed when the backend is down, not as a model problem', async () => {
    fetchBackendHealth.mockRejectedValue(new Error('down'));
    fetchModelReadiness.mockRejectedValue(new Error('down'));
    render(<AdminOverviewPage />);

    await waitFor(() => {
      expect(screen.getAllByText('Not checked — backend unavailable').length).toBeGreaterThan(0);
    });
    expect(screen.queryByRole('list', { name: 'Course model readiness' })).not.toBeInTheDocument();
  });

  it('shows the readiness error when only that check failed', async () => {
    fetchModelReadiness.mockRejectedValue(new Error('No response after 30 seconds. The request was stopped.'));
    render(<AdminOverviewPage />);

    expect(await screen.findByText(/No response after 30 seconds/)).toBeInTheDocument();
  });
});
