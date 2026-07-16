import { NextRequest, NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";
import { supabaseAdmin } from "@/lib/supabase";
import {
  MAX_PROJECT_BYTES,
  PROJECT_MEDIA_BUCKET,
  originalPath,
} from "@/lib/project-storage";

const PROJECT_ID = /^[a-f0-9]{32}$/;

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ projectId: string }> },
) {
  if (!supabaseAdmin) {
    return NextResponse.json({ error: "Supabase Storage is not configured" }, { status: 503 });
  }
  const { projectId } = await params;
  if (!PROJECT_ID.test(projectId)) {
    return NextResponse.json({ error: "Invalid project id" }, { status: 400 });
  }

  const contentType = req.headers.get("content-type") || "";
  const contentLength = Number(req.headers.get("content-length") || 0);
  if (!contentType.startsWith("video/")) {
    return NextResponse.json({ error: "A video content type is required" }, { status: 400 });
  }
  if (contentLength > MAX_PROJECT_BYTES) {
    return NextResponse.json({ error: "Video must be no larger than 50 MB" }, { status: 413 });
  }

  const session = await auth0.getSession();
  const { data: project } = await supabaseAdmin
    .from("projects")
    .select("user_id")
    .eq("project_id", projectId)
    .maybeSingle();
  if (project && (!session || project.user_id !== session.user.sub)) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const bytes = await req.arrayBuffer();
  if (bytes.byteLength <= 0 || bytes.byteLength > MAX_PROJECT_BYTES) {
    return NextResponse.json({ error: "Invalid video size" }, { status: 413 });
  }
  const { error } = await supabaseAdmin.storage
    .from(PROJECT_MEDIA_BUCKET)
    .upload(originalPath(projectId), bytes, {
      contentType,
      cacheControl: "3600",
      upsert: true,
    });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ stored: true });
}
