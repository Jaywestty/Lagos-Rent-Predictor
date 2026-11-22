# Adugbo AI web app

React 19 + TypeScript + Vite front end for Adugbo AI. See the [root README](../README.md) for setup.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api to http://localhost:8000
npm test
npm run lint
npm run build    # type-checks, builds and enforces the bundle size budget
```

Set `VITE_API_BASE_URL` to point the app at an API on another origin.

- `src/pages/` public pages (area guides, compare, move-in cost, saved)
- `src/components/` chat, answer card, listing cards and charts
- `src/admin/` admin console, loaded only on `/admin`
- `src/services/api.ts` API client
