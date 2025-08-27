Native Postgres Test Harness - Quickstart

Run prerequisites:
- Ensure Postgres installed and running on Windows
- Ensure Node.js installed (v18+ recommended)

Quick run (PowerShell):
$env:PGHOST='localhost'; $env:PGPORT='5432'; $env:PGUSER='postgres'; $env:PGPASSWORD='postgres'; $env:PGDATABASE='legal_ai_test'; npm run test:pg:setup

Alternative (if ts-node not installed):
- Install ts-node: npm i -D ts-node typescript @types/node
- Then run: npx ts-node tests/setup-db.ts

If you prefer compiled JS, transpile tests/setup-db.ts to tests/setup-db.js using tsc before running with node.
