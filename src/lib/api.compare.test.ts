import { afterEach, describe, expect, it, vi } from 'vitest';
import { generateBaseModel, generateFineTuned, generateFineTunedRag, generateRag } from './api';

describe('compare live API client courseId', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('includes courseId in the Base Model request body', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:8001');

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        answer: 'Base answer',
        model: 'llama3.2:3b',
        responseType: 'base',
        courseId: 'css-430-summer-2026-ibce',
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await generateBaseModel('css-430-summer-2026-ibce', 'Can I submit late work?');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://127.0.0.1:8001/base-model/generate');
    expect(options.method).toBe('POST');
    expect(JSON.parse(String(options.body))).toEqual({
      courseId: 'css-430-summer-2026-ibce',
      question: 'Can I submit late work?',
    });
  });

  it('includes courseId in the RAG request body', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:8001');

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        courseId: 'css-430-summer-2026-ibce',
        answer: 'RAG answer',
        model: 'llama3.2:3b',
        responseType: 'rag',
        sources: [
          {
            chunkId: 'css430-late-1',
            sectionTitle: 'CSS 430 Late Policy',
            text: 'Late work policy text',
            score: 0.91,
          },
        ],
        retrievedChunks: [],
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await generateRag('css-430-summer-2026-ibce', 'Can I submit late work?');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://127.0.0.1:8001/rag/generate');
    expect(JSON.parse(String(options.body))).toEqual({
      courseId: 'css-430-summer-2026-ibce',
      question: 'Can I submit late work?',
      topK: 3,
    });
  });

  it('includes courseId in the Fine-Tuned request body', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:8001');

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        answer: 'Fine-tuned answer',
        model: 'meta-llama/Llama-3.2-3B-Instruct',
        responseType: 'fineTuned',
        courseId: 'css-430-summer-2026-ibce',
        adapterLoaded: true,
        generationSeconds: 1.1,
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await generateFineTuned('css-430-summer-2026-ibce', 'Can I submit late work?');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://127.0.0.1:8001/fine-tuned/generate');
    expect(options.method).toBe('POST');
    expect(JSON.parse(String(options.body))).toEqual({
      courseId: 'css-430-summer-2026-ibce',
      question: 'Can I submit late work?',
    });
  });

  it('includes courseId in the Fine-Tuned + RAG request body', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:8001');

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        courseId: 'css-430-summer-2026-ibce',
        answer: 'Fine-tuned RAG answer',
        model: 'meta-llama/Llama-3.2-3B-Instruct',
        responseType: 'fineTunedRag',
        adapterLoaded: true,
        generationSeconds: 1.2,
        sources: [
          {
            chunkId: 'css430-late-1',
            sectionTitle: 'CSS 430 Late Policy',
            text: 'Late work policy text',
            score: 0.91,
          },
        ],
        retrievedChunks: [],
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await generateFineTunedRag('css-430-summer-2026-ibce', 'Can I submit late work?');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://127.0.0.1:8001/fine-tuned-rag/generate');
    expect(options.method).toBe('POST');
    expect(JSON.parse(String(options.body))).toEqual({
      courseId: 'css-430-summer-2026-ibce',
      question: 'Can I submit late work?',
      topK: 3,
    });
  });
});
