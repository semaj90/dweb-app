Playwright E2E: Full CRUD flow (register, login, create case, upload evidence, verify Postgres)

Setup

1. Install dev dependencies (from `sveltekit-frontend`):

   npm install -D @playwright/test pg
   npx playwright install

2. Create and configure your Postgres database. By default the test uses:

   postgresql://postgres:postgres@localhost:5432/legal_ai_db

   Ensure a `cases` table exists with at least `id` (text or uuid) and `title` columns, or adapt the query in `tests/e2e/helpers/db.ts`.

3. Copy `.env.example` to `.env` and set `PG_CONNECTION_STRING` and `BASE_URL` as needed.

Run

From the `sveltekit-frontend` folder:

   # Start the dev server (Playwright can reuse it)
   npm run dev

   # In another shell run the tests
   npx playwright test tests/e2e --project=.

Notes & assumptions

- This test assumes the app exposes standard routes and selectors (Register, Login, /user/dashboard, New Case, form inputs named `email`, `password`, `title`, `description`, an `input[type=file]` for evidence upload, and buttons with text labels used in the test). You may need to adjust selectors to match the actual UI components in the project.

- The DB verification queries a `cases` table. If your backend uses a different schema (drizzle-orm, pgvector, etc.), adapt `tests/e2e/helpers/db.ts` accordingly.

- For CI, prefer a dedicated test database and manage migrations before running the tests.
