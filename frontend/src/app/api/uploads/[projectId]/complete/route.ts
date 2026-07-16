import { NextRequest, NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";
import { supabaseAdmin } from "@/lib/supabase";
import {
  hydrateModalCache,
  objectExists,
  originalPath,
} from "@/lib/project-storage";

const PROJECT_ID = /^[a-f0-9]{32}$/;

export async function POST(
  _req: NextRequest,
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
  const { data: project, error: projectError } = await supabaseAdmin
    .from("projects")
    .select("*")
    .eq("project_id", projectId)
    .maybeSingle();
  if (projectError) return NextResponse.json({ error: projectError.message }, { status: 500 });
  if (project && (!session || project.user_id !== session.user.sub)) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const storedOriginal = project?.original_path || originalPath(projectId);
  if (!(await objectExists(storedOriginal))) {
    return NextResponse.json({ error: "Uploaded video was not found" }, { status: 409 });
  }

  if (project) {
    const { error } = await supabaseAdmin
      .from("projects")
      .update({
        original_path: storedOriginal,
        storage_status: "stored",
        status: "stored",
      })
      .eq("project_id", projectId)
      .eq("user_id", project.user_id);
    if (error?.code === "PGRST204") {
      await supabaseAdmin
        .from("projects")
        .update({ status: "stored" })
        .eq("project_id", projectId)
        .eq("user_id", project.user_id);
    }
  }

  const hydration = await hydrateModalCache({
    projectId,
    sourcePath: storedOriginal,
    userId: project?.user_id || null,
    originalStoragePath: storedOriginal,
  });

  if (project) {
    const { error } = await supabaseAdmin
      .from("projects")
      .update({ status: hydration.available ? "processing" : "stored" })
      .eq("project_id", projectId)
      .eq("user_id", project.user_id);
    if (error) console.warn("Could not update project compute status", error.message);
  }

  return NextResponse.json({
    project_id: projectId,
    stored: true,
    compute_available: hydration.available,
    compute_error: hydration.available ? null : hydration.error,
  });
}
