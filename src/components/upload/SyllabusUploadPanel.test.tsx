/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { ApiError } from '../../lib/httpClient';

const uploadCourseSyllabusMock = vi.hoisted(() => vi.fn());

vi.mock('../../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../../lib/api')>('../../lib/api');
  return { ...actual, uploadCourseSyllabus: uploadCourseSyllabusMock };
});

import { SyllabusUploadPanel } from './SyllabusUploadPanel';

const COURSE = 'css-430-fall-2026-k7q2';

const UPLOADED = {
  courseId: COURSE,
  syllabusFileName: 'fall.pdf',
  syllabusType: 'pdf',
  syllabusStatus: 'indexed',
  fileSize: 16,
  characterCount: 1800,
  chunkCount: 9,
};

function renderPanel(props: { hasSyllabus: boolean; lastUploadFailed?: boolean }) {
  const onUploaded = vi.fn();
  render(
    <SyllabusUploadPanel
      courseId={COURSE}
      hasSyllabus={props.hasSyllabus}
      lastUploadFailed={props.lastUploadFailed ?? false}
      onUploaded={onUploaded}
    />,
  );
  return onUploaded;
}

function choose(name = 'fall.pdf', type = 'application/pdf') {
  const file = new File(['%PDF-1.4 sample'], name, { type });
  fireEvent.change(screen.getByLabelText(/Syllabus file/), { target: { files: [file] } });
  return file;
}

describe('SyllabusUploadPanel', () => {
  beforeEach(() => {
    uploadCourseSyllabusMock.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('after a failed first upload, says so and retries into the same course', async () => {
    uploadCourseSyllabusMock.mockResolvedValue({ ...UPLOADED, replaced: false });
    const onUploaded = renderPanel({ hasSyllabus: false, lastUploadFailed: true });

    expect(screen.getByText('The last syllabus upload could not be processed')).toBeInTheDocument();
    const file = choose();
    fireEvent.click(screen.getByRole('button', { name: 'Upload syllabus' }));

    expect(await screen.findByText(/Syllabus uploaded/)).toBeInTheDocument();
    expect(uploadCourseSyllabusMock).toHaveBeenCalledWith(COURSE, file);
    expect(onUploaded).toHaveBeenCalledWith(expect.objectContaining({ chunkCount: 9 }));
    expect(
      screen.queryByText('The last syllabus upload could not be processed'),
    ).not.toBeInTheDocument();
  });

  it('replaces a working syllabus and says the new version is in use', async () => {
    uploadCourseSyllabusMock.mockResolvedValue({ ...UPLOADED, replaced: true });
    const onUploaded = renderPanel({ hasSyllabus: true });

    choose();
    fireEvent.click(screen.getByRole('button', { name: 'Replace syllabus' }));

    expect(await screen.findByText(/Syllabus replaced/)).toBeInTheDocument();
    expect(onUploaded).toHaveBeenCalledTimes(1);
  });

  it('a failed replacement tells the professor the current syllabus still serves', async () => {
    uploadCourseSyllabusMock.mockRejectedValue(
      new ApiError('Ollama is unavailable for embeddings.', 503),
    );
    const onUploaded = renderPanel({ hasSyllabus: true });

    choose();
    fireEvent.click(screen.getByRole('button', { name: 'Replace syllabus' }));

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Your current syllabus is still in use.');
    expect(alert).not.toHaveTextContent(/ollama/i);
    expect(onUploaded).not.toHaveBeenCalled();
  });

  it('checks the file before sending anything', async () => {
    renderPanel({ hasSyllabus: true });

    choose('syllabus.docx', 'application/octet-stream');
    fireEvent.click(screen.getByRole('button', { name: 'Replace syllabus' }));

    await waitFor(() =>
      expect(
        screen.getByText('Only .pdf and .txt syllabus files are supported.'),
      ).toBeInTheDocument(),
    );
    expect(uploadCourseSyllabusMock).not.toHaveBeenCalled();
  });
});
