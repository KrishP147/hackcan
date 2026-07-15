import { Auth0Client } from "@auth0/nextjs-auth0/server";

const REQUIRED_AUTH0_ENV = [
  "AUTH0_DOMAIN",
  "AUTH0_CLIENT_ID",
  "AUTH0_CLIENT_SECRET",
  "AUTH0_SECRET",
] as const;

/** Auth0 is optional: guest upload/edit remains available without credentials. */
export function isAuth0Configured() {
  return REQUIRED_AUTH0_ENV.every((key) => Boolean(process.env[key]));
}

const audience = process.env.AUTH0_AUDIENCE;
const configured = isAuth0Configured();

// Auth0 v4 reads AUTH0_DOMAIN, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET,
// AUTH0_SECRET, and APP_BASE_URL directly. Supplying the API audience here
// makes /auth/access-token return a JWT the FastAPI backend can validate.
export const auth0 = new Auth0Client({
  // Valid inert defaults keep guest-only local builds quiet. Every route that
  // can invoke Auth0 is guarded by isAuth0Configured().
  domain: configured ? process.env.AUTH0_DOMAIN : "guest.invalid",
  clientId: configured ? process.env.AUTH0_CLIENT_ID : "guest-mode",
  clientSecret: configured ? process.env.AUTH0_CLIENT_SECRET : "guest-mode",
  secret: configured ? process.env.AUTH0_SECRET : "0".repeat(64),
  appBaseUrl: process.env.APP_BASE_URL || "http://localhost:3000",
  authorizationParameters: audience
    ? { audience, scope: "openid profile email" }
    : { scope: "openid profile email" },
});
