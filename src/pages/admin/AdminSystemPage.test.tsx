/** @vitest-environment jsdom */
import { cleanup, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import '@testing-library/jest-dom/vitest';

import { AdminSystemPage } from './AdminSystemPage';

/**
 * The architecture page describes what is deployed.
 *
 * After authentication shipped it still said authentication, access control
 * and course join codes were not implemented, and described the data model
 * as Firebase-shaped paths. These tests hold the page to the system that is
 * actually running.
 */
describe('AdminSystemPage', () => {
  afterEach(() => {
    cleanup();
  });

  it('no longer lists implemented identity work as not yet implemented', () => {
    render(<AdminSystemPage />);

    const deferred = screen.getByText('Not yet implemented').closest('section');
    expect(deferred).not.toBeNull();
    expect(deferred).not.toHaveTextContent(/authentication and access control/i);
    expect(deferred).not.toHaveTextContent(/join codes/i);
    expect(deferred).not.toHaveTextContent(/student enrolment/i);
    // What is genuinely deferred stays.
    expect(deferred).toHaveTextContent(/Self-service password reset/);
    expect(deferred).toHaveTextContent(/Reproducible evaluation provenance/);
    expect(deferred).toHaveTextContent(/Retention, redaction, and alerting/);
  });

  it('describes the identity implementation that is in production', () => {
    render(<AdminSystemPage />);

    const identity = screen.getByText('Identity and access').closest('section');
    expect(identity).not.toBeNull();
    expect(identity).toHaveTextContent(/anonymous, course-specific participants/);
    expect(identity).toHaveTextContent(/class code/);
    expect(identity).toHaveTextContent(/email and password/);
    expect(identity).toHaveTextContent(/Two separate sessions/);
    expect(identity).toHaveTextContent(/Sessions live in PostgreSQL/);
    expect(identity).toHaveTextContent(/Authorization is enforced by the backend/);
    expect(identity).toHaveTextContent(/exactly the courses in/);
    expect(identity).toHaveTextContent(/administrator reaches every course/);
  });

  it('makes PostgreSQL the authoritative data model and names every identity table', () => {
    render(<AdminSystemPage />);

    const identityTables = screen.getByRole('list', { name: 'Identity tables' });
    for (const table of [
      'users',
      'participants',
      'auth_sessions',
      'course_memberships',
      'invitations',
      'admin_actions',
    ]) {
      expect(within(identityTables).getByText(table)).toBeInTheDocument();
    }

    const courseTables = screen.getByRole('list', { name: 'Course tables' });
    for (const table of ['courses', 'seed_examples', 'evaluations', 'training_runs']) {
      expect(within(courseTables).getByText(table)).toBeInTheDocument();
    }

    expect(screen.getByText(/PostgreSQL is the system of record/)).toBeInTheDocument();
    // The Firebase-shaped storage path is gone from the storage table.
    const storage = screen.getByRole('list', { name: 'Storage locations' });
    expect(storage).not.toHaveTextContent('metadata|seedExamples|evaluations');
  });
});
