/** @vitest-environment jsdom */
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { QUESTION_MAX_LENGTH, QuestionAsk } from './QuestionAsk';

afterEach(() => {
  cleanup();
});

describe('QuestionAsk', () => {
  it('stops the question at the length the backend accepts and a rating can record', () => {
    render(
      <QuestionAsk
        examples={[]}
        examplesLabel="Try an example"
        examplesLoading={false}
        isRunning={false}
        onAsk={vi.fn()}
      />,
    );
    expect(QUESTION_MAX_LENGTH).toBe(2000);
    expect(
      screen.getByLabelText('What would you like to ask about this course?'),
    ).toHaveAttribute('maxlength', '2000');
  });
});
