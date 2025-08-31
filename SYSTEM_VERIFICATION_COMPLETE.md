# System Verification Assets

This directory contains lightweight verification assets created for quick local validation of the development environment.

Files created:

- `test-complete-crud-system.js` — Minimal Node-based CRUD smoke test that exercises example API endpoints (replace endpoints to match your app).
- `verify-system-architecture.cjs` — CommonJS script that checks critical service ports (SvelteKit, Ollama, Postgres, Redis).
- `run-system-tests.bat` — Windows batch runner that executes the above scripts and pauses for inspection.

How to run (Windows):

1. Open PowerShell or Command Prompt in the project root: `C:\Users\james\Desktop\deeds-web\deeds-web-app`
2. Run `run-system-tests.bat` and inspect the console output.

Notes and assumptions:

- These are placeholder/test stubs. Replace endpoints, ports, and payloads with production/test fixtures as needed.
- Node 18+ is recommended (uses global fetch in `test-complete-crud-system.js`).
- Running network checks assumes services are bound to localhost. Adjust hosts if needed.

Next steps:

- If you'd like, I can:
  - Commit these files in separate, auditable commits.
  - Expand the test harness to use a test runner (Jest/Mocha) and include assertions.
  - Wire tests into CI (GitHub Actions) with matrixed service spin-ups.

