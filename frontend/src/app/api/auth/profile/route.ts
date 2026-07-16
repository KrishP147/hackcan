import { auth0, isAuth0Configured } from "@/lib/auth0";
import { NextResponse } from "next/server";

export async function GET() {
  if (!isAuth0Configured()) {
    return new NextResponse(null, { status: 204 });
  }
  const session = await auth0.getSession();
  if (!session) return new NextResponse(null, { status: 204 });
  return NextResponse.json(session.user);
}
