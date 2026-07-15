import { validateVideoUpload } from "@/lib/video-upload";

type UploadResult = {
  project_id: string;
  storage_path: string;
  compute_available: boolean;
  compute_error?: string | null;
};

export async function uploadProjectVideo(
  file: File,
  onStatus?: (status: string) => void,
): Promise<UploadResult> {
  onStatus?.("Checking video length...");
  await validateVideoUpload(file);

  onStatus?.("Preparing secure storage...");
  const intentResponse = await fetch("/api/uploads/intent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_name: file.name,
      content_type: file.type,
      size: file.size,
    }),
  });
  const intent = await intentResponse.json();
  if (!intentResponse.ok) {
    throw new Error(intent.error || "Could not prepare video storage");
  }

  onStatus?.("Saving video to your project...");
  const directBody = new FormData();
  directBody.append("cacheControl", "3600");
  directBody.append("", file);

  let directUpload: Response | null = null;
  try {
    directUpload = await fetch(intent.signed_url, {
      method: "PUT",
      headers: { "x-upsert": "false" },
      body: directBody,
    });
  } catch {
    // The same-origin fallback below handles restrictive browser/network setups.
  }

  if (!directUpload?.ok) {
    const fallback = await fetch(`/api/uploads/${intent.project_id}/content`, {
      method: "PUT",
      headers: { "Content-Type": file.type },
      body: file,
    });
    if (!fallback.ok) {
      const failure = await fallback.json().catch(() => ({}));
      throw new Error(failure.error || "Video storage upload failed");
    }
  }

  onStatus?.("Starting the GPU editor...");
  const completeResponse = await fetch(`/api/uploads/${intent.project_id}/complete`, {
    method: "POST",
  });
  const complete = await completeResponse.json();
  if (!completeResponse.ok) {
    throw new Error(complete.error || "Video was saved, but project setup failed");
  }

  return {
    project_id: intent.project_id,
    storage_path: intent.storage_path,
    compute_available: Boolean(complete.compute_available),
    compute_error: complete.compute_error,
  };
}

