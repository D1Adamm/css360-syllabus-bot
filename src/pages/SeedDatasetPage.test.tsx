/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import type { SeedExample } from '../types';
import { CourseProvider } from '../context/CourseContext';
import { SeedDatasetPage } from './SeedDatasetPage';

const subscribeToSeedExamplesMock = vi.hoisted(() => vi.fn());

vi.mock('../lib/firebase', () => ({
  app: {},
  database: { name: 'mock-db' },
}));

vi.mock('../lib/seedExamplesDb', () => ({
  subscribeToSeedExamples: subscribeToSeedExamplesMock,
  createSeedExample: vi.fn(),
  deleteSeedExample: vi.fn(),
  deleteAllSeedExamples: vi.fn(),
  updateSeedExample: vi.fn(),
}));

function makeSeed(courseLabel: string, id: string): SeedExample {
  return {
    id,
    instruction: `When does ${courseLabel} meet?`,
    response: `${courseLabel} meets on a course-specific schedule described in its syllabus.`,
    category: 'Course Basics',
    sourceSection: `${courseLabel} Meetings`,
    difficulty: 'Easy',
    directlyAnswered: true,
    origin: 'user',
    createdAt: '2026-07-01T00:00:00.000Z',
  };
}

function renderDataset(courseId: string) {
  return render(
    <MemoryRouter initialEntries={[`/course/${courseId}/dataset`]}>
      <Routes>
        <Route
          path="/course/:courseId/dataset"
          element={
            <CourseProvider courseId={courseId}>
              <SeedDatasetPage />
            </CourseProvider>
          }
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe('SeedDatasetPage course separation', () => {
  beforeEach(() => {
    subscribeToSeedExamplesMock.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('loads only the route course seedExamples path via the courseId subscription', async () => {
    subscribeToSeedExamplesMock.mockImplementation(
      (courseId: string, onData: (seeds: SeedExample[]) => void) => {
        onData([makeSeed('CSS 430', `${courseId}-seed-1`)]);
        return () => undefined;
      },
    );

    renderDataset('css-430-summer-2026-ibce');

    await waitFor(() => {
      expect(subscribeToSeedExamplesMock).toHaveBeenCalledWith(
        'css-430-summer-2026-ibce',
        expect.any(Function),
        expect.any(Function),
      );
    });

    expect(screen.getByText(/courses\/css-430-summer-2026-ibce\/seedExamples/)).toBeInTheDocument();
    expect(screen.getByText(/When does CSS 430 meet\?/)).toBeInTheDocument();
    expect(screen.queryByText(/CSS360|CSS 360/i)).not.toBeInTheDocument();
  });

  it('keeps CSS 350 and CSS 430 datasets fully separate', async () => {
    subscribeToSeedExamplesMock.mockImplementation(
      (courseId: string, onData: (seeds: SeedExample[]) => void) => {
        if (courseId === 'css-350-spring-2026-abcd') {
          onData([makeSeed('CSS 350', 'css350-only')]);
        } else {
          onData([makeSeed('CSS 430', 'css430-only')]);
        }
        return () => undefined;
      },
    );

    const first = renderDataset('css-350-spring-2026-abcd');
    expect(await screen.findByText(/When does CSS 350 meet\?/)).toBeInTheDocument();
    expect(screen.queryByText(/When does CSS 430 meet\?/)).not.toBeInTheDocument();
    first.unmount();

    renderDataset('css-430-summer-2026-ibce');
    expect(await screen.findByText(/When does CSS 430 meet\?/)).toBeInTheDocument();
    expect(screen.queryByText(/When does CSS 350 meet\?/)).not.toBeInTheDocument();
  });

  it('shows an empty state when the course has no seed examples', async () => {
    subscribeToSeedExamplesMock.mockImplementation(
      (_courseId: string, onData: (seeds: SeedExample[]) => void) => {
        onData([]);
        return () => undefined;
      },
    );

    renderDataset('css-430-summer-2026-ibce');

    expect(
      await screen.findByText('No seed examples have been created for this course yet.'),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText('Seed examples')).not.toBeInTheDocument();
  });

  it('does not import local CSS 360 prototype seed JSON into the dataset list', async () => {
    subscribeToSeedExamplesMock.mockImplementation(
      (_courseId: string, onData: (seeds: SeedExample[]) => void) => {
        onData([]);
        return () => undefined;
      },
    );

    renderDataset('css-430-summer-2026-ibce');

    await screen.findByText('No seed examples have been created for this course yet.');

    // Phrase commonly present in the legacy CSS 360 prototype seed export.
    expect(screen.queryByText(/UW1 room 302/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/prototype examples/i)).not.toBeInTheDocument();
  });
});
