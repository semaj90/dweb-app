# Secure Environment Variables (Frontend vs Server)

Purpose

This document explains how to safely move secret and sensitive environment variables out of the frontend `.env` (which is client-exposed in SvelteKit) and into server-only environment storage. It includes recommended workflows for local development, CI, and deployment.

Why this matters

- Any variable prefixed with `PUBLIC_` or that is present in the frontend `.env` can be embedded into client-side bundles and viewed by users. Credentials such as `DATABASE_URL`, `PG_CONN_STRING`, `MINIO_SECRET_KEY`, or `JWT_SECRET` must never be stored in or shipped from frontend environment files.

Quick recommendations

1. Move secrets to server-only environment files (example: `server.env`), systemd unit files, container secrets, or a managed secret store (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault, GitHub Actions secrets, GitLab CI variables).

2. Keep a checked-in `server.env.example` with placeholder keys (no real credentials) so other developers know what keys to set.

3. In local development, source `server.env` in your shell before starting backend services. Keep `server.env` out of version control (add to `.gitignore`).

4. The frontend should only receive sanitized, non-sensitive values (API base URLs, flags, public feature toggles). Use `PUBLIC_` prefix intentionally for values that are safe to expose.

Loading secrets in different runtimes

- Node / Go / Other servers
  - Export or load environment variables from a local file (only for local dev):

    ```powershell
    # PowerShell example (dev only)
    $env:DATABASE_URL = "postgresql://user:pass@localhost:5432/db"
    npm run dev
    ```

  - For Go, read from `os.Getenv("DATABASE_URL")` and do not rely on client-provided values.

- SvelteKit server-only variables
  - Use non-`PUBLIC_` env vars and access them from server routes, hooks, and +server.ts files only. Do not expose these via server-to-client assignments.

Parsing numeric env values

Environment variables are strings. Convert typed values explicitly:

```ts
const port = Number(process.env.PUBLIC_POSTGRES_PORT) || 5432;
```

Local dev tips

- Add `server.env` to `.gitignore` and store a `server.env.example` with placeholders.
- Use docker-compose secrets or bind mounts when spinning up local DB services.
- Use `dotenv` for local dev (server-side only); do not include `dotenv` in client bundles.

CI and deployment

- Configure secrets in your CI provider and your cloud provider’s secret manager. Use environment variables injected at build/runtime.
- For ephemeral credentials (e.g. temporary DB users), rotate and limit privileges.

Example workflow

1. Add `server.env` locally with your `DATABASE_URL`, `PG_CONN_STRING`, `JWT_SECRET`, etc.
2. Start backend services and ensure they read secrets from the environment.
3. Keep frontend `.env` with only public values and feature flags.

Where to find help

If you want, I can:
- Create `server.env.example` with placeholders (I will not populate real secrets).
- Add a checked-in `sveltekit-frontend/.env.example` with only public, non-sensitive keys.
- Add a small automation script to validate that no known secret keys are present in frontend `.env` files.

