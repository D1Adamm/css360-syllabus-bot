/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import type { SeedExample } from '../types';
import { SeedCard } from './SeedCard';

function makeSeed(overrides: Partial<SeedExample> = {}): SeedExample {
  return {
    id: 'seed-1',
    instruction: 'Can I submit late work?',
    response: 'Late work may be submitted within 24 hours for half credit.',
    category: 'Late Work',
    sourceSection: 'Late Policy',
    difficulty: 'Medium',
    directlyAnswered: true,
    origin: 'ai_generated',
    ...overrides,
  };
}

describe('SeedCard validation display', () => {
  afterEach(() => {
    cleanup();
  });

  it('shows overall percentage for legacy validation records', () => {
    render(
      <SeedCard
        seed={makeSeed({
          validation: {
            grounded: true,
            correct: true,
            clear: true,
            useful: true,
            score: 0.95,
            reason: 'Supported by the source chunk.',
          },
        })}
      />,
    );

    expect(screen.getByText('Validation 95%')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Expand example' }));
    expect(screen.getByText(/Supported by the source chunk/)).toBeInTheDocument();
    expect(screen.queryByText(/Component scores:/)).not.toBeInTheDocument();
  });

  it('shows questionType, component scores, and unsupported claims for new records', () => {
    render(
      <SeedCard
        seed={makeSeed({
          questionType: 'scenario',
          validation: {
            score: 0.86,
            reason: 'Mostly grounded with one weak claim.',
            unsupportedClaims: ['Suggested work-division strategy'],
            components: {
              grounded: 0.78,
              correct: 0.8,
              clear: 0.88,
              useful: 0.9,
              naturalStudentWording: 0.85,
              categoryCorrect: 0.84,
              notTrivialOrTemporary: 0.82,
            },
            grounded: 0.78,
            correct: 0.8,
            clear: 0.88,
            useful: 0.9,
          },
        })}
      />,
    );

    expect(screen.getByText('Validation 86%')).toBeInTheDocument();
    expect(screen.getByText('scenario')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Expand example' }));
    expect(screen.getByText(/Component scores:/)).toBeInTheDocument();
    expect(screen.getByText(/grounded 78%/)).toBeInTheDocument();
    expect(screen.getByText(/Unsupported claims:/)).toBeInTheDocument();
    expect(screen.getByText(/Suggested work-division strategy/)).toBeInTheDocument();
  });
});

describe('SeedCard delete permissions', () => {
  afterEach(() => {
    cleanup();
  });

  it('shows Delete example for user seeds', () => {
    render(<SeedCard seed={makeSeed({ origin: 'user' })} onDelete={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Delete example' })).toBeInTheDocument();
  });

  it('hides Delete example for ai_generated seeds', () => {
    render(<SeedCard seed={makeSeed({ origin: 'ai_generated' })} onDelete={vi.fn()} />);
    expect(screen.queryByRole('button', { name: 'Delete example' })).not.toBeInTheDocument();
  });

  it('hides Delete example for prototype seeds', () => {
    render(<SeedCard seed={makeSeed({ origin: 'prototype' })} onDelete={vi.fn()} />);
    expect(screen.queryByRole('button', { name: 'Delete example' })).not.toBeInTheDocument();
  });
});
