Route testing workflow

1. Generate filesystem route list

Run from `sveltekit-frontend`:

```bash
node scripts/generate-routes.mjs
```

This writes `routes.txt` in the `sveltekit-frontend` root.

2. Run Playwright smoke tests

Install Playwright if not installed:

```bash
npm i -D @playwright/test
npx playwright install
```

Run tests (from `sveltekit-frontend`):

```bash
npx playwright test
```

3. DB debug endpoint (dev-only)

A lightweight endpoint is provided at `/api/debug/db?email=...` that will attempt to query your Drizzle `db` if present and report whether a user exists (dev only).

4. CI / dev integration

You can run your app and tests together using a concurrently-based task or the provided VS Code task `Dev: Full Stack (dev:full, tee logs)` which starts the full stack and writes logs to `./logs/dev-full.log`.

Notes
- Playwright will fail on any console.error or uncaught exception during navigation which helps catch hydration failures and missing components.
- For routes that require authentication or specific DB state, add route-specific Playwright tests that perform setup (create user via API, seed data) before navigation.
- If Playwright reports failures, check `./logs/vite.log`, `./logs/dev-full.log`, and `/api/debug/logs` for merged server logs.
