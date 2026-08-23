

/**
 * Suite-wide guarantee that no frontend test performs real network I/O.
 *
 * The backend suite learned this the hard way: `backend/.env` held a live
 * database URL, so an unstubbed path did not fail — it succeeded, against
 * production, and recreated a course record on every run. The frontend is one
 * `VITE_API_BASE_URL` away from the same class of mistake: the paths that would
 * leak point at a real FastAPI host rather than a mock.
 *
 * So `fetch` is replaced with something that fails loudly and names the URL.
 * Every test that needs a response mocks `./dbApi`, `./api`, or `fetch` itself;
 * anything that does not is asserting against a request it never meant to make.
 */

class RealNetworkRequestBlocked extends Error {
  constructor(url: string, method: string) {
    super(
      `A test tried to reach ${method} ${url}.\n` +
        'Frontend tests must not perform real network requests. Mock the module ' +
        'under test (usually ./dbApi or ./api), or stub global.fetch for this case.',
    );
    this.name = 'RealNetworkRequestBlocked';
  }
}

/*
 * Installed once, at setup scope, rather than per test.
 *
 * A file that stubs `fetch` itself — `vi.stubGlobal('fetch', ...)` at module
 * scope — loads after this and replaces the guard, which is exactly right: it
 * has said what it expects the request to be. Reinstalling before every test
 * would instead overwrite that stub and break the tests doing the correct
 * thing. Vitest restores stubbed globals between files, so the guard is back
 * for the next one.
 */
globalThis.fetch = ((input: unknown, init?: { method?: string }) => {
  const url =
    typeof input === 'string'
      ? input
      : input instanceof URL
        ? input.toString()
        : ((input as { url?: string })?.url ?? String(input));

  throw new RealNetworkRequestBlocked(url, init?.method ?? 'GET');
}) as typeof globalThis.fetch;

// Exported so a test can assert the guard itself is live rather than trusting
// that an absence of failures means it is doing anything.
export { RealNetworkRequestBlocked };
