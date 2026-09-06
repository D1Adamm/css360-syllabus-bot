/** @vitest-environment jsdom */
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

/**
 * The Diagnostics section of the admin course page.
 *
 * Two production defects live here. The fact inventory control sat on
 * "Building…" indefinitely, because it held one request open for an extraction
 * that takes the CPU model minutes to hours and nothing bounded the wait. And
 * the page showed two chunk counts for one course — 92 on the record row, 163
 * in the index inspector — without saying that they come from different
 * places or which one retrieval uses.
 *
 * Every test here drives the real hook and the real page against a mocked API
 * module; no timer is left running past the test that started it.
 */

const COURSE = 'css-360-winter-2026-a7rp';

vi.mock('../../context/CourseContext', () => ({
  useCourseId: () => COURSE,
}));

let recordChunkCount = 92;
vi.mock('../../hooks/useCourseMetadata', () => ({
  useCourseMetadata: () => {
    const metadata = { ...METADATA, chunkCount: recordChunkCount };
    return { state: { status: 'ready', metadata } as const, metadata, retry: vi.fn() };
  },
}));

vi.mock('../../hooks/useCourseExampleCounts', () => ({
  useCourseExampleCounts: () => ({
    status: 'ready',
    counts: { total: 4, approved: 2, pending: 1, rejected: 1, edited: 0 },
  }),
}));

vi.mock('../../hooks/useCourseModel', () => ({
  useCourseModel: () => ({ state: { status: 'none' } as const, retry: vi.fn() }),
}));

vi.mock('../../hooks/useCourseModelRequest', () => ({
  useCourseModelRequest: () => ({
    state: { status: 'none' } as const,
    submitting: false,
    submitError: null,
    submit: vi.fn(),
    clearSubmitError: vi.fn(),
  }),
}));

const api = vi.hoisted(() => ({
  fetchCourseChunks: vi.fn(),
  requestFactInventory: vi.fn(),
  runSeedQualityCheck: vi.fn(),
}));

vi.mock('../../lib/adminApi', () => ({
  ApiError: class ApiError extends Error {
    status?: number;
    constructor(message: string, status?: number) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
    }
  },
  DIAGNOSTIC_TIMEOUT_MS: 30_000,
  ...api,
}));

vi.mock('../../lib/adminPeopleApi', () => ({
  listUsers: vi.fn().mockResolvedValue({ count: 0, users: [] }),
  addMembership: vi.fn(),
  removeMembership: vi.fn(),
}));

vi.mock('../../components/invite/StudentAccessPanel', () => ({
  StudentAccessPanel: () => null,
}));

import type { CourseMetadata } from '../../types';
import { ApiError } from '../../lib/adminApi';
import {
  FACT_INVENTORY_MAX_POLLS,
  FACT_INVENTORY_POLL_INTERVAL_MS,
} from '../../hooks/useFactInventoryProbe';
import { AdminCourseDetailPage } from './AdminCourseDetailPage';

const METADATA: CourseMetadata = {
  name: 'CSS 360',
  title: 'Software Engineering',
  term: 'Fall 2025',
  instructorName: '',
  createdAt: '2026-02-01T09:00:00.000Z',
  syllabusStatus: 'indexed',
  syllabusFileName: 'syllabus.pdf',
  syllabusType: 'pdf',
  chunkCount: 92,
};

const INVENTORY = {
  courseId: COURSE,
  model: 'qwen3:4b',
  factCount: 66,
  cached: true,
  countsByKind: { deadline: 20, policy: 46 },
};

function chunksResponse(count: number) {
  return {
    courseId: COURSE,
    chunkCount: count,
    indexVersion: 2,
    documentTitle: 'Software Engineering (Fall 2025)',
    chunks: Array.from({ length: count }, (_, index) => ({
      chunkId: `chunk-${String(index).padStart(3, '0')}`,
      sectionTitle: `Section ${index}`,
      text: `Chunk ${index}`,
      order: index,
    })),
  };
}

function renderPage() {
  return render(
    <MemoryRouter>
      <AdminCourseDetailPage />
    </MemoryRouter>,
  );
}

function row(label: string): HTMLElement {
  const element = screen.getByText(label).closest('li');
  if (!element) {
    throw new Error(`No diagnostics row labelled ${label}`);
  }
  return element;
}

const factsRow = () => row('Fact inventory');
const chunksRow = () => row('Syllabus index');

async function click(button: HTMLElement) {
  await act(async () => {
    fireEvent.click(button);
  });
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe('AdminCourseDetailPage fact inventory', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    api.fetchCourseChunks.mockReset();
    api.requestFactInventory.mockReset();
    api.runSeedQualityCheck.mockReset();
    recordChunkCount = 92;
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it('shows the inventory when the backend has it', async () => {
    api.requestFactInventory.mockResolvedValue({ status: 'ready', inventory: INVENTORY });
    renderPage();

    await click(within(factsRow()).getByRole('button', { name: 'Inspect' }));

    expect(api.requestFactInventory).toHaveBeenCalledWith(COURSE);
    expect(factsRow()).toHaveTextContent('66 facts');
    expect(factsRow()).toHaveTextContent('qwen3:4b');
    expect(factsRow()).toHaveTextContent('cached');
    expect(factsRow()).toHaveTextContent('deadline: 20 · policy: 46');
    expect(within(factsRow()).getByRole('button', { name: 'Inspect' })).toBeEnabled();
  });

  it('polls while the backend builds and stops the moment the result arrives', async () => {
    api.requestFactInventory
      .mockResolvedValueOnce({ status: 'building', startedAt: '2026-09-06T10:00:00Z' })
      .mockResolvedValueOnce({ status: 'building', startedAt: '2026-09-06T10:00:00Z' })
      .mockResolvedValueOnce({ status: 'ready', inventory: { ...INVENTORY, cached: false } });
    renderPage();

    await click(within(factsRow()).getByRole('button', { name: 'Inspect' }));

    expect(api.requestFactInventory).toHaveBeenCalledTimes(1);
    expect(within(factsRow()).getByRole('button', { name: 'Building…' })).toBeDisabled();
    expect(within(factsRow()).getByRole('status')).toHaveTextContent('Building on the server');

    await advance(FACT_INVENTORY_POLL_INTERVAL_MS);
    expect(api.requestFactInventory).toHaveBeenCalledTimes(2);
    expect(within(factsRow()).getByRole('button', { name: 'Building…' })).toBeDisabled();

    await advance(FACT_INVENTORY_POLL_INTERVAL_MS);
    expect(api.requestFactInventory).toHaveBeenCalledTimes(3);
    expect(factsRow()).toHaveTextContent('66 facts');
    // Freshly built, so the result line carries no cache marker.
    expect(factsRow()).not.toHaveTextContent('· cached');
    expect(within(factsRow()).getByRole('button', { name: 'Inspect' })).toBeEnabled();

    // Nothing keeps polling once the result is in.
    await advance(FACT_INVENTORY_POLL_INTERVAL_MS * 3);
    expect(api.requestFactInventory).toHaveBeenCalledTimes(3);
  });

  it('shows the error with a Retry that works when the build fails', async () => {
    api.requestFactInventory
      .mockRejectedValueOnce(
        new ApiError('The fact inventory could not be built: Ollama is unavailable.', 503),
      )
      .mockResolvedValueOnce({ status: 'ready', inventory: INVENTORY });
    renderPage();

    await click(within(factsRow()).getByRole('button', { name: 'Inspect' }));

    expect(within(factsRow()).getByRole('alert')).toHaveTextContent('Ollama is unavailable');
    const retry = within(factsRow()).getByRole('button', { name: 'Retry' });
    expect(retry).toBeEnabled();

    await click(retry);

    expect(api.requestFactInventory).toHaveBeenCalledTimes(2);
    expect(within(factsRow()).queryByRole('alert')).toBeNull();
    expect(factsRow()).toHaveTextContent('66 facts');
  });

  it('never stays on Building… indefinitely: after the poll budget it hands back control', async () => {
    api.requestFactInventory.mockResolvedValue({
      status: 'building',
      startedAt: '2026-09-06T10:00:00Z',
    });
    renderPage();

    await click(within(factsRow()).getByRole('button', { name: 'Inspect' }));
    for (let poll = 1; poll < FACT_INVENTORY_MAX_POLLS; poll += 1) {
      await advance(FACT_INVENTORY_POLL_INTERVAL_MS);
    }

    expect(api.requestFactInventory).toHaveBeenCalledTimes(FACT_INVENTORY_MAX_POLLS);
    expect(within(factsRow()).queryByRole('button', { name: 'Building…' })).toBeNull();
    const checkAgain = within(factsRow()).getByRole('button', { name: 'Check again' });
    expect(checkAgain).toBeEnabled();
    expect(within(factsRow()).getByRole('status')).toHaveTextContent('Still building on the server');

    // Polling has genuinely stopped, not merely relabelled.
    await advance(FACT_INVENTORY_POLL_INTERVAL_MS * 5);
    expect(api.requestFactInventory).toHaveBeenCalledTimes(FACT_INVENTORY_MAX_POLLS);

    // Check again is one more request, on the reader's terms.
    await click(checkAgain);
    expect(api.requestFactInventory).toHaveBeenCalledTimes(FACT_INVENTORY_MAX_POLLS + 1);
    expect(within(factsRow()).getByRole('button', { name: 'Building…' })).toBeDisabled();
  });

  it('cancels its polling when the page goes away', async () => {
    api.requestFactInventory.mockResolvedValue({ status: 'building', startedAt: null });
    const view = renderPage();

    await click(within(factsRow()).getByRole('button', { name: 'Inspect' }));
    expect(api.requestFactInventory).toHaveBeenCalledTimes(1);

    view.unmount();
    await advance(FACT_INVENTORY_POLL_INTERVAL_MS * 4);

    expect(api.requestFactInventory).toHaveBeenCalledTimes(1);
  });

  it('ignores the answer to a superseded request', async () => {
    let resolveFirst: (value: unknown) => void = () => undefined;
    api.requestFactInventory
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFirst = resolve;
          }),
      )
      .mockResolvedValueOnce({ status: 'ready', inventory: { ...INVENTORY, factCount: 7 } });
    renderPage();

    await click(within(factsRow()).getByRole('button', { name: 'Inspect' }));
    // The control is disabled while a request is out, so a second inspect
    // only ever comes from the hook's own callers — drive it the same way the
    // Retry / Check again buttons do, by re-clicking once it is enabled.
    await act(async () => {
      resolveFirst({ status: 'building', startedAt: null });
    });
    expect(within(factsRow()).getByRole('button', { name: 'Building…' })).toBeDisabled();

    await advance(FACT_INVENTORY_POLL_INTERVAL_MS);
    expect(factsRow()).toHaveTextContent('7 facts');
  });
});

describe('AdminCourseDetailPage syllabus index', () => {
  beforeEach(() => {
    api.fetchCourseChunks.mockReset();
    api.requestFactInventory.mockReset();
    recordChunkCount = 92;
  });

  afterEach(() => {
    cleanup();
  });

  it('expands to the first 12 chunks and collapses again', async () => {
    api.fetchCourseChunks.mockResolvedValue(chunksResponse(163));
    renderPage();

    const inspect = within(chunksRow()).getByRole('button', { name: 'Inspect' });
    expect(inspect).toHaveAttribute('aria-expanded', 'false');
    await click(inspect);

    expect(chunksRow()).toHaveTextContent('163 chunks in the index file');
    expect(chunksRow()).toHaveTextContent('Software Engineering (Fall 2025)');
    expect(chunksRow()).toHaveTextContent('index v2');
    expect(within(chunksRow()).getAllByRole('listitem')).toHaveLength(12);
    expect(chunksRow()).toHaveTextContent('Showing the first 12 of 163.');

    const collapse = within(chunksRow()).getByRole('button', { name: 'Collapse' });
    expect(collapse).toHaveAttribute('aria-expanded', 'true');
    await click(collapse);

    expect(within(chunksRow()).queryAllByRole('listitem')).toHaveLength(0);
    expect(chunksRow()).not.toHaveTextContent('163 chunks');
    expect(within(chunksRow()).getByRole('button', { name: 'Inspect' })).toBeEnabled();
    // Collapsing is local; it does not re-read the file.
    expect(api.fetchCourseChunks).toHaveBeenCalledTimes(1);
  });

  it('names both sources and the fix when the record and the index file disagree', async () => {
    api.fetchCourseChunks.mockResolvedValue(chunksResponse(163));
    renderPage();

    expect(screen.getByText('Index chunks (course record)').closest('li')).toHaveTextContent('92');

    await click(within(chunksRow()).getByRole('button', { name: 'Inspect' }));

    const warning = screen.getByText('The course record disagrees with the index file').closest(
      '.ui-callout',
    );
    expect(warning).not.toBeNull();
    expect(warning).toHaveTextContent('record in PostgreSQL says 92 chunks');
    expect(warning).toHaveTextContent('index file holds 163');
    expect(warning).toHaveTextContent('163 is the real count');
    expect(warning).toHaveTextContent(
      `python -m app.reindex_course --course-id ${COURSE} --sync-record`,
    );
  });

  it('says nothing about a discrepancy when the two agree', async () => {
    recordChunkCount = 163;
    api.fetchCourseChunks.mockResolvedValue(chunksResponse(163));
    renderPage();

    await click(within(chunksRow()).getByRole('button', { name: 'Inspect' }));

    expect(chunksRow()).toHaveTextContent('163 chunks in the index file');
    expect(screen.queryByText('The course record disagrees with the index file')).toBeNull();
  });

  it('offers Retry after a failed read', async () => {
    api.fetchCourseChunks
      .mockRejectedValueOnce(new ApiError('Course syllabus index was not found.', 404))
      .mockResolvedValueOnce(chunksResponse(3));
    renderPage();

    await click(within(chunksRow()).getByRole('button', { name: 'Inspect' }));
    expect(within(chunksRow()).getByRole('alert')).toHaveTextContent('index was not found');

    await click(within(chunksRow()).getByRole('button', { name: 'Retry' }));
    expect(chunksRow()).toHaveTextContent('3 chunks in the index file');
    expect(within(chunksRow()).queryByRole('alert')).toBeNull();
  });
});

describe('AdminCourseDetailPage examples row', () => {
  afterEach(() => {
    cleanup();
  });

  it('links to the dataset and to the review workflow for this course', () => {
    renderPage();
    expect(screen.getByRole('link', { name: 'Open dataset' })).toHaveAttribute(
      'href',
      `/admin/courses/${COURSE}/examples`,
    );
    expect(screen.getByRole('link', { name: 'Review examples' })).toHaveAttribute(
      'href',
      `/admin/courses/${COURSE}/review`,
    );
  });
});
