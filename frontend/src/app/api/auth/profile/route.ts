import { auth0 } from "@/lib/auth0";
import { NextResponse } from "next/server";

export async function GET() {
  if (!process.env.AUTH0_DOMAIN || !process.env.AUTH0_CLIENT_ID || !process.env.AUTH0_SECRET) {
    return new NextResponse(null, { status: 204 });
  }
  const session = await auth0.getSession();
  if (!session) return new NextResponse(null, { status: 204 });
  return NextResponse.json(session.user);
}
