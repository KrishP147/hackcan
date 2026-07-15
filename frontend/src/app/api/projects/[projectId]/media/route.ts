import { NextRequest, NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";
import { supabaseAdmin } from "@/lib/supabase";
import {
  checkpointPath,
  createDownloadUrl,
  currentPath,
  objectExists,
  originalPath,
  thumbnailPath,
} from "@/lib/project-storage";

const PROJECT_ID = /^[a-f0-9]{32}$/;

export async function GET(
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

  const session = await auth0.getSession();
  const { data: project } = await supabaseAdmin
    .from("projects")
    .select("*")
    .eq("project_id", projectId)
    .maybeSingle();
  if (project && (!session || project.user_id !== session.user.sub)) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const kind = req.nextUrl.searchParams.get("kind") || "current";
  let path: string | null;
  if (kind === "original") {
    path = project?.original_path || originalPath(projectId);
  } else if (kind === "thumbnail") {
    path = project?.thumbnail_path || thumbnailPath(projectId);
  } else if (kind === "checkpoint") {
    path = project?.checkpoint_path || checkpointPath(projectId);
  } else if (kind === "export") {
    path = project?.export_path || `${currentPath(projectId).replace(/current\.mp4$/, "exports/final.mp4")}`;
  } else {
    const candidate = project?.current_path || currentPath(projectId);
    path = await objectExists(candidate)
      ? candidate
      : project?.original_path || originalPath(projectId);
  }

  if (!path || !(await objectExists(path))) {
    return NextResponse.json({ error: "Media not found" }, { status: 404 });
  }
  const signedUrl = await createDownloadUrl(path);
  return NextResponse.redirect(signedUrl, 307);
}
