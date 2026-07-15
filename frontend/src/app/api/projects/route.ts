import { NextRequest, NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";
import { supabaseAdmin } from "@/lib/supabase";

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

  if (!project_id) return NextResponse.json({ error: "project_id required" }, { status: 400 });

  const { data, error } = await supabaseAdmin
    .from("projects")
    .insert({
      project_id,
      user_id: session.user.sub,
      name: name || "Untitled Project",
      thumbnail_url: thumbnail_url || null,
      status: "created",
      last_frame: 0,
    })
    .select()
    .single();

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json(data);
}
