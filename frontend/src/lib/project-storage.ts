import { supabaseAdmin } from "@/lib/supabase";

export const PROJECT_MEDIA_BUCKET = "project-media";
export const MAX_PROJECT_BYTES = 100 * 1024 * 1024;

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function projectPrefix(projectId: string) {
  return `projects/${projectId}`;
}

export function originalPath(projectId: string) {
  return `${projectPrefix(projectId)}/original.mp4`;
}

export function currentPath(projectId: string) {
  return `${projectPrefix(projectId)}/current.mp4`;
}

export function thumbnailPath(projectId: string) {
  return `${projectPrefix(projectId)}/thumbnail.jpg`;
}

export function checkpointPath(projectId: string) {
  return `${projectPrefix(projectId)}/checkpoint.tar.gz`;
}

export async function objectExists(path: string) {
  if (!supabaseAdmin) return false;
  const slash = path.lastIndexOf("/");
  const folder = path.slice(0, slash);
  const name = path.slice(slash + 1);
  const { data, error } = await supabaseAdmin.storage
    .from(PROJECT_MEDIA_BUCKET)
    .list(folder, { limit: 100, search: name });
  return !error && Boolean(data?.some((entry) => entry.name === name));
}

export async function createDownloadUrl(path: string, expiresIn = 3600) {
  if (!supabaseAdmin) throw new Error("Supabase Storage is not configured");
  const { data, error } = await supabaseAdmin.storage
    .from(PROJECT_MEDIA_BUCKET)
    .createSignedUrl(path, expiresIn);
  if (error || !data?.signedUrl) {
    throw new Error(error?.message || "Could not create a media URL");
  }
  return data.signedUrl;
}

export async function hydrateModalCache({
  projectId,
  sourcePath,
  savedCheckpointPath,
  userId,
  originalStoragePath,
  currentStoragePath,
}: {
  projectId: string;
  sourcePath: string;
  savedCheckpointPath?: string | null;
  userId?: string | null;
  originalStoragePath?: string | null;
  currentStoragePath?: string | null;
}) {
  const importSecret = process.env.FRAMESHIFT_IMPORT_SECRET;
  if (!importSecret) {
    return { available: false, error: "Modal storage hydration is not configured" };
  }

  const sourceUrl = await createDownloadUrl(sourcePath, 15 * 60);
  const savedCheckpointExists = savedCheckpointPath
    ? await objectExists(savedCheckpointPath)
    : false;
  const checkpointUrl = savedCheckpointExists && savedCheckpointPath
    ? await createDownloadUrl(savedCheckpointPath, 15 * 60)
    : null;

  try {
    const response = await fetch(`${API_URL}/project/import`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Frameshift-Import-Secret": importSecret,
      },
      body: JSON.stringify({
        project_id: projectId,
        source_url: sourceUrl,
        checkpoint_url: checkpointUrl,
        user_id: userId || null,
        original_path: originalStoragePath || null,
        current_path: currentStoragePath || null,
        checkpoint_path: savedCheckpointPath || null,
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) {
      return { available: false, error: await response.text() };
    }
    return { available: true, data: await response.json() };
  } catch (error) {
    return {
      available: false,
      error: error instanceof Error ? error.message : "Modal is unavailable",
    };
  }
}

export async function removeProjectObjects(projectId: string) {
  if (!supabaseAdmin) return;
  const paths = [
    originalPath(projectId),
    currentPath(projectId),
    thumbnailPath(projectId),
    checkpointPath(projectId),
    `${projectPrefix(projectId)}/exports/final.mp4`,
  ];
  await supabaseAdmin.storage.from(PROJECT_MEDIA_BUCKET).remove(paths);
}
