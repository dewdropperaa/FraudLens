import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const backendTarget = process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000'
let hasLoggedProxyDown = false

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: (proxy) => {
          proxy.on('error', (error, req, res) => {
            if (!hasLoggedProxyDown) {
              hasLoggedProxyDown = true
              console.warn(
                `[vite-proxy] Backend unreachable at ${backendTarget}. Start the API server or update VITE_PROXY_TARGET.`,
              )
            }

            if (!res.headersSent) {
              res.writeHead(503, { 'Content-Type': 'application/json' })
            }
            res.end(
              JSON.stringify({
                detail: 'API backend is not reachable. Start the backend server and retry.',
                code: error.code || 'PROXY_ERROR',
              }),
            )
          })
          proxy.on('proxyRes', () => {
            hasLoggedProxyDown = false
          })
        },
      },
    },
  },
})
