import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // The OceanEmbed backend (`uvicorn api.server:app --port 8000`); the frontend falls back to mock data if it's down.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
