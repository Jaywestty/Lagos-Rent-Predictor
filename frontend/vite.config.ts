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
        let total = 0;
        for (const f of fs.readdirSync(dist)) {
          if (!f.endsWith('.js')) continue;
          const buf = fs.readFileSync(path.join(dist, f));
          total += zlib.gzipSync(buf).length;
        }
        const budget = 120 * 1024;
        if (total > budget) {
          throw new Error(`Bundle size check failed: ${total} bytes gzipped exceeds ${budget} bytes`);
        }
      },
    },
  ] as never,
  server: {
    proxy: {
      '/api': { target: process.env.VITE_API_BASE_URL || 'http://localhost:8000', changeOrigin: true },
      '/query': { target: process.env.VITE_API_BASE_URL || 'http://localhost:8000', changeOrigin: true },
      '/admin': { target: process.env.VITE_API_BASE_URL || 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: { rollupOptions: { output: {} } },
});
