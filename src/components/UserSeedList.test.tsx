/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import type { SeedExample, SeedOrigin } from '../types';
import { UserSeedList } from './UserSeedList';

function makeSeed(origin: SeedOrigin, id = `${origin}-1`): SeedExample {
  return {
    id,
    instruction: `${origin} question?`,
    response: `${origin} answer.`,
    category: 'Course Basics',
    sourceSection: 'Meetings',
    difficulty: 'Easy',
    directlyAnswered: true,
    origin,
  };
}

describe('UserSeedList manual-only list', () => {
  afterEach(() => {
    cleanup();
  });

  it('shows only user seeds and titles the list as Your seed examples', () => {
    const seeds = [
      makeSeed('user', 'user-1'),
      makeSeed('ai_generated', 'ai-1'),
      makeSeed('prototype', 'proto-1'),
    ];

    render(
      <UserSeedList seeds={seeds} onDelete={vi.fn()} onDeleteAll={vi.fn()} />,
    );

    expect(screen.getByRole('heading', { name: /Your seed examples \(1\)/ })).toBeInTheDocument();
    expect(screen.getByText('user question?')).toBeInTheDocument();
    expect(screen.queryByText('ai_generated question?')).not.toBeInTheDocument();
    expect(screen.queryByText('prototype question?')).not.toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'Delete example' })).toHaveLength(1);
  });

  it('hides AI-generated seeds from the Build Seeds list', () => {
    render(
      <UserSeedList
        seeds={[makeSeed('ai_generated')]}
        onDelete={vi.fn()}
        onDeleteAll={vi.fn()}
      />,
    );

    expect(screen.getByRole('heading', { name: /Your seed examples \(0\)/ })).toBeInTheDocument();
    expect(screen.queryByText('ai_generated question?')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Delete example' })).not.toBeInTheDocument();
    expect(
      screen.getByText(/AI-generated and reviewed seeds appear on Review Seeds and Dataset/),
    ).toBeInTheDocument();
  });

  it('hides prototype seeds from the Build Seeds list', () => {
    render(
      <UserSeedList
        seeds={[makeSeed('prototype')]}
        onDelete={vi.fn()}
        onDeleteAll={vi.fn()}
      />,
    );

    expect(screen.getByRole('heading', { name: /Your seed examples \(0\)/ })).toBeInTheDocument();
    expect(screen.queryByText('prototype question?')).not.toBeInTheDocument();
  });

  it('Delete all my examples only appears when user seeds exist and still calls user-only handler', async () => {
    const onDeleteAll = vi.fn();
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

    const { rerender } = render(
      <UserSeedList
        seeds={[makeSeed('ai_generated'), makeSeed('prototype')]}
        onDelete={vi.fn()}
        onDeleteAll={onDeleteAll}
      />,
    );

    expect(
      screen.queryByRole('button', { name: 'Delete all my examples' }),
    ).not.toBeInTheDocument();

    rerender(
      <UserSeedList
        seeds={[
          makeSeed('user', 'user-1'),
          makeSeed('ai_generated', 'ai-1'),
          makeSeed('prototype', 'proto-1'),
        ]}
        onDelete={vi.fn()}
        onDeleteAll={onDeleteAll}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Delete all my examples' }));
    expect(confirmSpy).toHaveBeenCalled();
    expect(onDeleteAll).toHaveBeenCalledTimes(1);

    confirmSpy.mockRestore();
  });
});
