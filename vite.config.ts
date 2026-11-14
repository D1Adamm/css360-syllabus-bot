import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/*
 * Where `/api` is proxied. The backend's usual local port by default; set
 * `DEV_API_PROXY_TARGET` to point this dev or preview server at another
 * backend, e.g. an isolated test instance on its own port
 * (docs/answer-cache-vm-test.md). `vite preview` uses the same proxy.
 */
const apiProxyTarget = process.env.DEV_API_PROXY_TARGET || 'http://127.0.0.1:8001'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    /*
     * Same-origin development.
     *
     * Production serves the React build and `/api/` from one origin behind
     * Nginx, and the session cookies are scoped to that origin. Proxying `/api`
     * here gives development the same shape — `VITE_API_BASE_URL=/api` — so
     * cookies, CORS and the CSRF origin check behave exactly as they will on
     * the VM instead of in a cross-origin configuration nobody deploys.
     */
    proxy: {
      '/api': {
        target: apiProxyTarget,
        changeOrigin: false,
      },
    },
  },
})
