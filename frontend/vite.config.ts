import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'bundle-size-check',
      closeBundle() {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const fs = require('node:fs');
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const zlib = require('node:zlib');
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const path = require('node:path');
        const dist = path.join(import.meta.dirname, 'dist/assets');
        if (!fs.existsSync(dist)) return;
        // Budget what a visitor downloads first (entry chunk); lazy pages and admin load on demand.
        let initial = 0;
        let total = 0;
        for (const f of fs.readdirSync(dist)) {
          if (!f.endsWith('.js')) continue;
          const size = zlib.gzipSync(fs.readFileSync(path.join(dist, f))).length;
          total += size;
          if (f.startsWith('index-')) initial += size;
        }
        const initialBudget = 120 * 1024;
        const totalBudget = 200 * 1024;
        if (initial > initialBudget) {
          throw new Error(`Bundle size check failed: initial JS ${initial} bytes gzipped exceeds ${initialBudget} bytes`);
        }
        if (total > totalBudget) {
          throw new Error(`Bundle size check failed: all JS ${total} bytes gzipped exceeds ${totalBudget} bytes`);
        }
      },
    },
  ] as never,
  server: {
    proxy: {
      '/api': { target: process.env.API_PROXY_TARGET || 'http://localhost:8000', changeOrigin: true },
      '/query': { target: process.env.API_PROXY_TARGET || 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: { rollupOptions: { output: {} } },
});
