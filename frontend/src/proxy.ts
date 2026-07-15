import { NextRequest, NextResponse } from "next/server";
import { auth0, isAuth0Configured } from "@/lib/auth0";

const PROTECTED_PREFIXES = ["/dashboard"];

/**
 * Auth0 v4 mounts /auth/* through the Next.js proxy. Everything except the
 * dashboard remains public so guests can upload and edit without an account.
 */
export async function proxy(request: NextRequest) {
  if (!isAuth0Configured()) return NextResponse.next();

  const authResponse = await auth0.middleware(request);
  const isProtected = PROTECTED_PREFIXES.some((prefix) =>
    request.nextUrl.pathname.startsWith(prefix)
  );

  if (isProtected) {
    const session = await auth0.getSession(request);
    if (!session) {
      const loginUrl = new URL("/auth/login", request.url);
      loginUrl.searchParams.set(
        "returnTo",
        `${request.nextUrl.pathname}${request.nextUrl.search}`
      );
      return NextResponse.redirect(loginUrl);
    }
  }

  return authResponse;
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt).*)",
  ],
};
