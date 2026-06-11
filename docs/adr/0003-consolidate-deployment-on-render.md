# ADR-0003 — Consolidate full deployment on Render

- **Status**: Accepted
- **Date**: June 2026
- **Source**: SRS v1.3 (requirements document revision)

## Context

The initial deployment design split the application across two platforms: the React
frontend on Vercel (free tier, automatic deploys from Git) and the FastAPI backend
on Render (free tier, accepts `.pt` model files). This split implied:

- Two hosting platforms to configure, monitor and document.
- Cross-domain communication between frontend and backend, requiring a stricter
  CORS setup and two different production domains.
- Duplicated deployment pipelines and environment configuration.

Render's free tier can also serve static sites, so a single platform can host both
parts of the application without additional cost.

## Decision

Remove the Vercel dependency and **consolidate the full deployment (frontend and
backend) on Render** (free tier). Both parts are publicly accessible over HTTPS with
automatic TLS certificates issued by Render.

## Consequences

- A single hosting platform: one dashboard, one deployment configuration and a
  simpler operations story for a one-person team.
- CORS remains configured restrictively, but frontend and backend now live under the
  same platform domains.
- Render free tier limitations now apply to the whole application: the backend
  service sleeps after 15 minutes of inactivity (cold start of 30-60 s), so the
  warm-up request strategy before the project demo remains necessary.
- Automatic deploys from Git are kept, now centralized in Render.
- The SRS is updated accordingly: product perspective (§2.1), dependencies table
  (§2.5), software interfaces (§3.1.3) and system constraints (§3.4).
- Project documentation (`CLAUDE.md`, `README.md`) must replace every reference to
  Vercel with Render.
