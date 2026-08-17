import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
    // Fails any test that performs real network I/O. See the file for why the
    // frontend needs this as much as the backend did.
    setupFiles: ['./src/test/noNetwork.ts'],
  },
})
