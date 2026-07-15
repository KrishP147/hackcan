import { NextRequest, NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";
import { supabaseAdmin } from "@/lib/supabase";
import {
  checkpointPath,
  currentPath,
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
  const { data: project } = await supabaseAdmin
    .from("projects")
    .select("*")
    .eq("project_id", projectId)
    .maybeSingle();
  if (project && (!session || project.user_id !== session.user.sub)) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const durableCurrent = project?.current_path || currentPath(projectId);
  const durableOriginal = project?.original_path || originalPath(projectId);
  const sourcePath = await objectExists(durableCurrent) ? durableCurrent : durableOriginal;
  if (!(await objectExists(sourcePath))) {
    return NextResponse.json({ error: "No durable video is available" }, { status: 404 });
  }
  const durableCheckpoint = project?.checkpoint_path || checkpointPath(projectId);
  const hydration = await hydrateModalCache({
    projectId,
    sourcePath,
    savedCheckpointPath: durableCheckpoint,
    userId: project?.user_id || null,
    originalStoragePath: durableOriginal,
    currentStoragePath: sourcePath === durableCurrent ? durableCurrent : null,
  });

  return NextResponse.json({
    project_id: projectId,
    stored: true,
    compute_available: hydration.available,
    compute_error: hydration.available ? null : hydration.error,
  });
}

