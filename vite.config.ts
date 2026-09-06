import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

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
        target: 'http://127.0.0.1:8001',
        changeOrigin: false,
      },
    },
  },
})
