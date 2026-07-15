export const MAX_VIDEO_DURATION_SECONDS = 6;

function readVideoDuration(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    const objectUrl = URL.createObjectURL(file);

    const cleanup = () => {
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(objectUrl);
    };

    video.preload = "metadata";
    video.onloadedmetadata = () => {
      const duration = video.duration;
      cleanup();

      if (!Number.isFinite(duration) || duration <= 0) {
        reject(new Error("Could not read this video's duration."));
        return;
      }

      resolve(duration);
    };
    video.onerror = () => {
      cleanup();
      reject(new Error("Please choose a valid video file."));
    };
    video.src = objectUrl;
  });
}

export async function validateVideoUpload(file: File): Promise<number> {
  if (!file.type.startsWith("video/")) {
    throw new Error("Please choose a video file.");
  }

  const duration = await readVideoDuration(file);
  if (duration >= MAX_VIDEO_DURATION_SECONDS) {
    throw new Error(
      `Video must be under ${MAX_VIDEO_DURATION_SECONDS} seconds. This video is ${duration.toFixed(1)} seconds.`
    );
  }

  return duration;
}
