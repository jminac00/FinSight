# ADR-0002 — Cookie and granular consent management

- **Status**: Accepted
- **Date**: June 2026
- **Source**: SRS v1.2 (requirements document revision)

## Context

FinSight is a public web application operated from Spain, so it must comply with
article 22 of the LSSI-CE and the guidelines of the AEPD (Spanish data protection
authority) regarding cookies and equivalent client-side storage technologies.

The first iteration has no user accounts, but the interface plans client-side
persistence features: remembering the cookie banner decision, recalling the last
analyzed ticker, storing the acknowledgement of the financial disclaimer (MiFID II)
and, optionally, usage analytics. The AEPD doctrine treats `localStorage`,
`sessionStorage` and `IndexedDB` as legally equivalent to cookies, so any of these
mechanisms triggers the same consent obligations.

## Decision

Introduce explicit cookie and consent management requirements (RNF-35 to RNF-39):

- **Cookie categories**: strictly necessary, functional and analytics.
- **Granular consent banner**: the user can accept or reject each category
  independently; the decision persists across visits.
- **Consent preference cookie**: the banner decision itself is stored in a strictly
  necessary technical cookie, which does not require prior consent under article 22
  of the LSSI-CE, so the banner does not reappear on every page load.
- **Functional cookies** (last analyzed ticker, disclaimer acknowledgement, visual
  theme) and **analytics cookies** (page views, most queried tickers, session times)
  are only activated after explicit consent for their category.
- **Legal equivalence of client-side storage**: consent obligations apply to any
  storage mechanism in the user's terminal (cookies, `localStorage`,
  `sessionStorage`, `IndexedDB`), regardless of the technology used.

## Consequences

- The frontend must implement the consent banner before introducing any
  non-essential client-side storage; functional and analytics features are gated by
  the user's granular consent.
- The privacy policy page (RGPD) must describe the cookie categories and their
  purposes.
- Analytics remain a low-priority, consent-conditioned option (RNF-38); the system
  works fully when every optional category is rejected.
- The SRS adds section 3.3.7 (Cookie and Consent Management) with requirements
  RNF-35 to RNF-39.
