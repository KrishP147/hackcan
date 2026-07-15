# Auth0 guest access and project history

FrameShift uses three separate responsibilities:

- Auth0 identifies optional signed-in users.
- Supabase Postgres stores small project metadata rows for the dashboard.
- The `frameshift-projects` Modal Volume stores videos, frames, masks, flows,
  edits, and rendered files.

Guests can upload and edit without authentication. Their project URLs and local
history remain in that browser. Signed-in users additionally get durable project
metadata and can resume their own projects from `/dashboard` on another device.

## Auth0

Create these two Auth0 resources:

1. A **Regular Web Application** for the Next.js frontend.
2. An **API** using an identifier such as `https://api.frameshift.app` and the
   RS256 signing algorithm.

For local development, configure the Regular Web Application with:

```text
Allowed Callback URLs: http://localhost:3000/auth/callback
Allowed Logout URLs:   http://localhost:3000
Allowed Web Origins:   http://localhost:3000
```

Add the equivalent HTTPS URLs for the production frontend. Do not add the Modal
URL as an Auth0 callback; Modal only validates API bearer tokens.

Copy `frontend/.env.example` to `frontend/.env.local` and set:

```dotenv
AUTH0_DOMAIN=your-tenant.eu.auth0.com
AUTH0_CLIENT_ID=...
AUTH0_CLIENT_SECRET=...
AUTH0_SECRET=64-hex-characters
APP_BASE_URL=http://localhost:3000
AUTH0_AUDIENCE=https://api.frameshift.app
```

Generate the cookie secret with:

```bash
openssl rand -hex 32
```

The same `AUTH0_DOMAIN` and `AUTH0_AUDIENCE` must be present in the
`frameshift-secrets` Modal secret. The frontend obtains an API access token from
Auth0 and attaches it to authenticated uploads; anonymous uploads omit it.

Auth0 v4 mounts `/auth/login`, `/auth/logout`, `/auth/callback`,
`/auth/profile`, and `/auth/access-token` through `frontend/src/proxy.ts`.

## Supabase

Create a Supabase project and run:

```bash
supabase db push
```

Alternatively, paste
`supabase/migrations/202607150001_create_projects.sql` into the Supabase SQL
Editor and run it once.

Set these only on the Next.js server/Vercel deployment:

```dotenv
NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SECRET_KEY=sb_secret_...
```

The secret key must never use a `NEXT_PUBLIC_` prefix. The legacy
`SUPABASE_SERVICE_ROLE_KEY` is also accepted, but new Supabase projects should
use an `sb_secret_...` key. The migration
revokes browser roles from the projects table; Next.js checks the Auth0 session
and filters every query by the session's `user.sub`.

## Modal secret

The backend secret should contain:

```dotenv
GEMINI_API_KEY=...
FRONTEND_ORIGIN=https://your-frontend.example
AUTH0_DOMAIN=your-tenant.eu.auth0.com
AUTH0_AUDIENCE=https://api.frameshift.app
```

Update and redeploy:

```bash
cd backend
venv/bin/modal secret create frameshift-secrets --from-dotenv .env.modal --force
FRAMESHIFT_MODAL_GPU=L40S venv/bin/modal deploy modal_app.py
```

Check `GET /health`: `auth0_configured` should be `true` on Modal.
