import { NextRequest, NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";
import { supabaseAdmin } from "@/lib/supabase";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ projectId: string }> }
) {
  if (!supabaseAdmin) {
    return NextResponse.json({ error: "Project sync is not configured" }, { status: 503 });
  }
  const session = await auth0.getSession();
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { projectId } = await params;
  const { data, error } = await supabaseAdmin
    .from("projects")
    .select("*")
    .eq("project_id", projectId)
    .eq("user_id", session.user.sub)
    .maybeSingle();

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  if (!data) return NextResponse.json({ error: "Project not found" }, { status: 404 });
  return NextResponse.json(data);
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ projectId: string }> }
) {
  if (!supabaseAdmin) {
    return NextResponse.json({ error: "Project sync is not configured" }, { status: 503 });
  }
  const session = await auth0.getSession();
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { projectId } = await params;
  const body = await req.json();

  // Only allow updating these fields
  const allowed = ["last_frame", "status", "name", "thumbnail_url"];
  const updates: Record<string, unknown> = {};
  for (const key of allowed) {
    if (key in body) updates[key] = body[key];
  }

  const { data, error } = await supabaseAdmin
    .from("projects")
    .update(updates)
    .eq("project_id", projectId)
    .eq("user_id", session.user.sub)
    .select()
    .single();

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json(data);
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ projectId: string }> }
) {
  if (!supabaseAdmin) {
    return NextResponse.json({ error: "Project sync is not configured" }, { status: 503 });
  }
  const session = await auth0.getSession();
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { projectId } = await params;

  const { error } = await supabaseAdmin
    .from("projects")
    .delete()
    .eq("project_id", projectId)
    .eq("user_id", session.user.sub);

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ success: true });
}
