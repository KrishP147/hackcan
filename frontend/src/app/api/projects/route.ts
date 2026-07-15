import { NextRequest, NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";
import { supabaseAdmin } from "@/lib/supabase";

const PROJECT_ID = /^[a-f0-9-]{8,36}$/i;

export async function GET() {
  if (!supabaseAdmin) {
    return NextResponse.json({ error: "Project sync is not configured" }, { status: 503 });
  }
  const session = await auth0.getSession();
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { data, error } = await supabaseAdmin
    .from("projects")
    .select("*")
    .eq("user_id", session.user.sub)
    .order("updated_at", { ascending: false });

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json(data);
}

export async function POST(req: NextRequest) {
  if (!supabaseAdmin) {
    return NextResponse.json({ error: "Project sync is not configured" }, { status: 503 });
  }
  const session = await auth0.getSession();
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json();
  const { project_id, name, thumbnail_url } = body;

  if (typeof project_id !== "string" || !PROJECT_ID.test(project_id)) {
    return NextResponse.json({ error: "Valid project_id required" }, { status: 400 });
  }

  const userId = session.user.sub;
  const safeName = typeof name === "string" && name.trim()
    ? name.trim().slice(0, 255)
    : "Untitled Project";
  const safeThumbnail = typeof thumbnail_url === "string"
    ? thumbnail_url.slice(0, 2048)
    : null;

  // Registration is idempotent. It lets a guest-created project be attached
  // after the user signs in, while preventing an existing project from being
  // reassigned to a different Auth0 identity.
  const { data: existing, error: existingError } = await supabaseAdmin
    .from("projects")
    .select("*")
    .eq("project_id", project_id)
    .maybeSingle();

  if (existingError) {
    return NextResponse.json({ error: existingError.message }, { status: 500 });
  }
  if (existing && existing.user_id !== userId) {
    return NextResponse.json({ error: "Project already belongs to another user" }, { status: 409 });
  }
  if (existing) {
    const { data, error } = await supabaseAdmin
      .from("projects")
      .update({ name: safeName, thumbnail_url: safeThumbnail ?? existing.thumbnail_url })
      .eq("project_id", project_id)
      .eq("user_id", userId)
      .select()
      .single();
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json(data);
  }

  const { data, error } = await supabaseAdmin
    .from("projects")
    .insert({
      project_id,
      user_id: userId,
      name: safeName,
      thumbnail_url: safeThumbnail,
      status: "created",
      last_frame: 0,
    })
    .select()
    .single();

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json(data);
}
