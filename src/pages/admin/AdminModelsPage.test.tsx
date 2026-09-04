/** @vitest-environment jsdom */
import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * The admin Models page: registered course models beside the shared inference
 * service that serves them.
 *
 * Two things this page has to keep right, and the reasons each is pinned here:
 *
 *   - The service section used to render `Service URL` and `Host` straight out
 *     of the health response. Those were the Tillicum compute node and the SSH
 *     tunnel destination, on a page served over a route that needs no
 *     credential. The backend no longer sends them, and this page no longer
 *     has anywhere to put them — the second half is what these tests hold,
 *     including against a payload that still carries the old fields.
 *
 *   - Whether the service answers says nothing about whether a course has a
 *     model. A failed health check must leave the registry section intact.
 */

const fetchFineTunedHealth = vi.fn();
const fetchCourseModel = vi.fn();

vi.mock('../../lib/adminApi', () => ({
  fetchFineTunedHealth: () => fetchFineTunedHealth(),
}));

vi.mock('../../lib/courseModelDb', async () => {
  const actual = await vi.importActual<typeof import('../../lib/courseModelDb')>(
    '../../lib/courseModelDb',
  );
  return {
    ...actual,
    fetchCourseModel: (...args: unknown[]) => fetchCourseModel(...args),
  };
});

vi.mock('../../hooks/useCourses', () => ({
  useCourses: () => ({
    state: {
      status: 'ready',
      courses: [
        {
          courseId: CSS350,
          metadata: { name: 'CSS 350', title: 'Management Principles' },
        },
      ],
    },
    retry: vi.fn(),
  }),
}));

import type { CourseModelRegistry, CourseModelVersion } from '../../types';
import { AdminModelsPage } from './AdminModelsPage';

const CSS350 = 'css-350-spring-2026-n3h9';

/** CSS 350 as deployed: v2 published, v1 still registered. */
const V2: CourseModelVersion = {
  version: 'v2',
  baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
  trainingExampleCount: 37,
  status: 'ready',
  deployment: 'online',
  artifactRef: `serving/${CSS350}/v2/adapter`,
  createdAt: '2026-08-27T20:53:10.000Z',
  runId: 'run-20260827t205310z-8c3cdb',
};

const V1: CourseModelVersion = {
  ...V2,
  version: 'v1',
  deployment: 'offline',
  trainingExampleCount: 42,
  artifactRef: `serving/${CSS350}/v1/adapter`,
  createdAt: '2026-08-09T23:45:00.000Z',
  runId: undefined,
};

const REGISTRY: CourseModelRegistry = {
  currentVersion: 'v2',
  versions: { v1: V1, v2: V2 },
};

/** Exactly what `GET /api/fine-tuned/health` returns for a live session. */
const HEALTH = {
  status: 'ok',
  model: 'meta-llama/Llama-3.2-3B-Instruct',
  adapterLoaded: true,
  courses: [{ courseId: CSS350, versions: ['v1', 'v2'], currentVersion: 'v2' }],
  secondsRemaining: 4200,
};

/** Synthetic topology. None of it may appear in the document. */
const NODE_HOSTNAME = 'n3129.hyak.local';
const SERVICE_PORT = 8412;
const SERVICE_URL = 'http://localhost:8412';

function renderPage() {
  return render(
    <MemoryRouter>
      <AdminModelsPage />
    </MemoryRouter>,
  );
}

afterEach(cleanup);

beforeEach(() => {
  fetchFineTunedHealth.mockReset();
  fetchCourseModel.mockReset();
  fetchFineTunedHealth.mockResolvedValue(HEALTH);
  fetchCourseModel.mockResolvedValue(REGISTRY);
});

describe('AdminModelsPage inference service', () => {
  it('shows which courses the service can answer for and the allocation left', async () => {
    renderPage();

    const service = await screen.findByRole('list', { name: 'Fine-tuned service' });
    expect(service).toHaveTextContent(`${CSS350} v2`);
    expect(service).toHaveTextContent('70 min');
    expect(service).toHaveTextContent('meta-llama/Llama-3.2-3B-Instruct');
  });

  it('says so when the service is up but serves no published adapter', async () => {
    fetchFineTunedHealth.mockResolvedValue({ ...HEALTH, courses: [], secondsRemaining: null });

    renderPage();

    const service = await screen.findByRole('list', { name: 'Fine-tuned service' });
    expect(service).toHaveTextContent('No published adapters');
  });

  it('renders no compute node, port or tunnel URL, even when a payload carries them', async () => {
    // An older backend build forwarded these three fields. The page must have
    // nowhere to put them rather than merely not receive them.
    fetchFineTunedHealth.mockResolvedValue({
      ...HEALTH,
      hostname: NODE_HOSTNAME,
      port: SERVICE_PORT,
      serviceUrl: SERVICE_URL,
    });

    renderPage();

    await screen.findByRole('list', { name: 'Fine-tuned service' });

    const text = document.body.textContent ?? '';
    expect(text).not.toContain(NODE_HOSTNAME);
    expect(text).not.toContain('n3129');
    expect(text).not.toContain(String(SERVICE_PORT));
    expect(text).not.toContain(SERVICE_URL);
    expect(text).not.toContain('localhost');
    expect(screen.queryByText('Service URL')).not.toBeInTheDocument();
    expect(screen.queryByText('Host')).not.toBeInTheDocument();
  });
});

describe('AdminModelsPage registry', () => {
  it('keeps the registered models when the service check fails', async () => {
    fetchFineTunedHealth.mockRejectedValue(new Error('backend unreachable'));

    renderPage();

    expect(await screen.findByText('Service did not respond')).toBeInTheDocument();

    const registered = await screen.findByRole('list', {
      name: 'Registered course models',
    });
    expect(registered).toHaveTextContent(CSS350);
    expect(registered).toHaveTextContent('v2');
    expect(registered).toHaveTextContent('online');
  });

  it('lists a course with no registry separately from a registered one', async () => {
    fetchCourseModel.mockResolvedValue(null);

    renderPage();

    expect(await screen.findByText('No course models registered')).toBeInTheDocument();
    expect(screen.getByText(new RegExp(`No model registered for: ${CSS350}`))).toBeInTheDocument();
  });
});
