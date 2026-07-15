import { randomBytes } from "crypto";
import { NextRequest, NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";
import { supabaseAdmin } from "@/lib/supabase";
import {
  MAX_PROJECT_BYTES,
  PROJECT_MEDIA_BUCKET,
  originalPath,
} from "@/lib/project-storage";

export async function POST(req: NextRequest) {
  if (!supabaseAdmin) {
    return NextResponse.json({ error: "Supabase Storage is not configured" }, { status: 503 });
  }

  const body = await req.json();
  const fileName = typeof body.file_name === "string" ? body.file_name.trim().slice(0, 255) : "";
  const contentType = typeof body.content_type === "string" ? body.content_type : "";
  const size = Number(body.size);

  if (!fileName || !contentType.startsWith("video/")) {
    return NextResponse.json({ error: "A valid video file is required" }, { status: 400 });
  }
  if (!Number.isFinite(size) || size <= 0 || size > MAX_PROJECT_BYTES) {
    return NextResponse.json({ error: "Video must be no larger than 50 MB" }, { status: 400 });
  }

  const projectId = randomBytes(16).toString("hex");
  const path = originalPath(projectId);
  const { data: upload, error: uploadError } = await supabaseAdmin.storage
    .from(PROJECT_MEDIA_BUCKET)
    .createSignedUploadUrl(path, { upsert: false });

  if (uploadError || !upload) {
    return NextResponse.json(
      { error: uploadError?.message || "Could not prepare video storage" },
      { status: 500 },
    );
  }

  const session = await auth0.getSession();
  if (session) {
    const extendedProject = {
      project_id: projectId,
      user_id: session.user.sub,
      name: fileName,
      original_path: path,
      storage_status: "uploading",
      status: "created",
      last_frame: 0,
    };
    let { error } = await supabaseAdmin.from("projects").insert(extendedProject);
    // Keep uploads operational before the optional metadata migration is
    // applied. Media paths are deterministic and can still be recovered.
    if (error?.code === "PGRST204") {
      ({ error } = await supabaseAdmin.from("projects").insert({
        project_id: projectId,
        user_id: session.user.sub,
        name: fileName,
        status: "created",
        last_frame: 0,
      }));
    }
    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
  }

  return NextResponse.json({
    project_id: projectId,
    storage_path: path,
    signed_url: upload.signedUrl,
    upload_token: upload.token,
  });
}
