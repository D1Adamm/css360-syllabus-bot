/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const listMock = vi.hoisted(() => vi.fn());
const createMock = vi.hoisted(() => vi.fn());
const revokeMock = vi.hoisted(() => vi.fn());
const toDataURLMock = vi.hoisted(() => vi.fn());

vi.mock('../../lib/inviteApi', async () => {
  const actual = await vi.importActual<typeof import('../../lib/inviteApi')>('../../lib/inviteApi');
  return {
    ...actual,
    listStudentInvites: listMock,
    createStudentInvite: createMock,
    revokeStudentInvite: revokeMock,
  };
});

vi.mock('qrcode', () => ({ toDataURL: toDataURLMock }));

import { StudentAccessPanel } from './StudentAccessPanel';
import type { StudentInvite } from '../../lib/inviteApi';

const COURSE = 'css-360-winter-2026-a7rp';

function invite(overrides: Partial<StudentInvite> = {}): StudentInvite {
  return {
    invitationId: 'inv-1',
    courseId: COURSE,
    code: '7K4P9X',
    status: 'active',
    createdAt: '2026-09-04T12:00:00+00:00',
    useCount: 12,
    participantCount: 9,
    createdByName: 'Prof',
    ...overrides,
  };
}

beforeEach(() => {
  listMock.mockReset();
  createMock.mockReset();
  revokeMock.mockReset();
  toDataURLMock.mockReset();
  toDataURLMock.mockResolvedValue('data:image/png;base64,QR');
  Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
});

afterEach(() => {
  cleanup();
});

describe('StudentAccessPanel', () => {
  it('shows the join page and the class code for the board, and the direct link', async () => {
    listMock.mockResolvedValue({ courseId: COURSE, count: 1, invites: [invite()] });
    render(<StudentAccessPanel courseId={COURSE} courseName="CSS 360" />);

    expect(await screen.findByText('7K4P9X')).toBeInTheDocument();
    expect(screen.getByText(`${window.location.host}/join`)).toBeInTheDocument();
    expect(screen.getByText(`${window.location.origin}/join/7K4P9X`)).toBeInTheDocument();
    expect(screen.getByText(/used 12 times · 9 students/)).toBeInTheDocument();
    expect(listMock).toHaveBeenCalledWith(COURSE);
  });

  it('offers to create a code when there is none', async () => {
    listMock.mockResolvedValueOnce({ courseId: COURSE, count: 0, invites: [] });
    listMock.mockResolvedValueOnce({ courseId: COURSE, count: 1, invites: [invite()] });
    createMock.mockResolvedValue(invite());
    render(<StudentAccessPanel courseId={COURSE} />);

    fireEvent.click(await screen.findByRole('button', { name: /Create class code/ }));
    await waitFor(() => expect(createMock).toHaveBeenCalledWith(COURSE));
    expect(await screen.findByText('7K4P9X')).toBeInTheDocument();
  });

  it('copies the direct link and the code', async () => {
    listMock.mockResolvedValue({ courseId: COURSE, count: 1, invites: [invite()] });
    render(<StudentAccessPanel courseId={COURSE} />);
    await screen.findByText('7K4P9X');

    fireEvent.click(screen.getByRole('button', { name: 'Copy link' }));
    await waitFor(() =>
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(`${window.location.origin}/join/7K4P9X`),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Copy code' }));
    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith('7K4P9X'));
  });

  it('draws a QR code for the direct link on request', async () => {
    listMock.mockResolvedValue({ courseId: COURSE, count: 1, invites: [invite()] });
    render(<StudentAccessPanel courseId={COURSE} />);
    await screen.findByText('7K4P9X');

    fireEvent.click(screen.getByRole('button', { name: 'Show QR' }));
    const image = await screen.findByRole('img', { name: /QR code for/ });
    expect(image).toHaveAttribute('src', 'data:image/png;base64,QR');
    expect(toDataURLMock).toHaveBeenCalledWith(
      `${window.location.origin}/join/7K4P9X`,
      expect.objectContaining({ width: 240 }),
    );
  });

  it('revokes only after confirmation', async () => {
    listMock.mockResolvedValueOnce({ courseId: COURSE, count: 1, invites: [invite()] });
    listMock.mockResolvedValueOnce({
      courseId: COURSE,
      count: 1,
      invites: [invite({ status: 'revoked', revokedAt: '2026-09-05T12:00:00+00:00' })],
    });
    revokeMock.mockResolvedValue(invite({ status: 'revoked' }));
    render(<StudentAccessPanel courseId={COURSE} />);
    await screen.findByText('7K4P9X');

    fireEvent.click(screen.getByRole('button', { name: 'Revoke' }));
    expect(revokeMock).not.toHaveBeenCalled();
    const dialog = await screen.findByRole('alertdialog');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Revoke' }));

    await waitFor(() => expect(revokeMock).toHaveBeenCalledWith(COURSE, 'inv-1'));
    expect(await screen.findByRole('button', { name: /Create class code/ })).toBeInTheDocument();
    expect(screen.getByText(/1 earlier code/)).toBeInTheDocument();
  });

  it('replaces the code through the backend, not by revoking and creating separately', async () => {
    listMock.mockResolvedValue({ courseId: COURSE, count: 1, invites: [invite()] });
    createMock.mockResolvedValue(invite({ invitationId: 'inv-2', code: 'M3N8QR' }));
    render(<StudentAccessPanel courseId={COURSE} />);
    await screen.findByText('7K4P9X');

    fireEvent.click(screen.getByRole('button', { name: 'New code' }));
    const dialog = await screen.findByRole('alertdialog');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Create new code' }));

    await waitFor(() => expect(createMock).toHaveBeenCalledWith(COURSE, { replaceExisting: true }));
    expect(revokeMock).not.toHaveBeenCalled();
  });

  it('reports a load failure with a retry', async () => {
    listMock.mockRejectedValueOnce(new Error('Could not load the class codes.'));
    listMock.mockResolvedValueOnce({ courseId: COURSE, count: 1, invites: [invite()] });
    render(<StudentAccessPanel courseId={COURSE} />);

    expect(await screen.findByText('Could not load class codes')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByText('7K4P9X')).toBeInTheDocument();
  });
});
