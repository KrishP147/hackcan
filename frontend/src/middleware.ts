import { NextRequest, NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";

// Only dashboard requires auth; editor is public (upload → edit without logging in)
const PROTECTED = ["/dashboard"];

export async function middleware(req: NextRequest) {
  // Local demo mode: Auth0 is optional until production credentials exist.
  if (!process.env.AUTH0_DOMAIN || !process.env.AUTH0_CLIENT_ID || !process.env.AUTH0_SECRET) {
    return NextResponse.next();
  }

  // Let Auth0 handle its own routes first
  const authResponse = await auth0.middleware(req);

  const { pathname } = req.nextUrl;
  const isProtected = PROTECTED.some((p) => pathname.startsWith(p));

  if (isProtected) {
    const session = await auth0.getSession(req);
    if (!session) {
      const loginUrl = new URL("/api/auth/login", req.url);
      loginUrl.searchParams.set("returnTo", pathname);
      return NextResponse.redirect(loginUrl);
    }
  }

  return authResponse;
}

export const config = {
  matcher: [
    "/dashboard",
    "/dashboard/:path*",
    "/api/auth/:path*",
    "/api/projects/:path*",
  ],
};
