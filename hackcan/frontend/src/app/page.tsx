'use client';

import { useState } from 'react';
import UploadWidget, { CloudinaryUploadResult } from '@/components/UploadWidget';

export default function Home() {
  const [uploadedVideo, setUploadedVideo] = useState<CloudinaryUploadResult | null>(null);

  return (
    <div className="flex min-h-screen flex-col bg-black text-white">
      <header className="flex items-center justify-between border-b border-white/10 px-8 py-4">
        <h1 className="text-xl font-semibold tracking-tight">FrameShift AI</h1>
        <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-white/60">
          Beta
        </span>
      </header>

      <main className="flex flex-1 flex-col items-center justify-center gap-8 px-8 py-20">
        {!uploadedVideo ? (
          <>
            <div className="flex flex-col items-center gap-4 text-center">
              <h2 className="text-4xl font-semibold tracking-tight">
                Edit objects in video with AI
              </h2>
              <p className="max-w-md text-lg text-white/50">
                Upload a video. Select an object. Edits propagate across every frame automatically.
              </p>
            </div>

            <UploadWidget onUpload={setUploadedVideo} resourceType="video">
              {(open) => (
                <button
                  onClick={open}
                  className="rounded-full bg-white px-8 py-3 text-sm font-medium text-black transition-opacity hover:opacity-80"
                >
                  Upload Video
                </button>
              )}
            </UploadWidget>
          </>
        ) : (
          <div className="flex w-full max-w-4xl flex-col gap-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-medium">Video uploaded</h2>
              <button
                onClick={() => setUploadedVideo(null)}
                className="text-sm text-white/40 hover:text-white"
              >
                Upload another
              </button>
            </div>

            <div className="overflow-hidden rounded-xl border border-white/10 bg-white/5">
              <video
                src={uploadedVideo.secure_url}
                controls
                className="w-full"
              />
            </div>

            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-white/40">Public ID</p>
              <p className="font-mono text-sm text-white/80">{uploadedVideo.public_id}</p>
            </div>

            <div className="rounded-xl border border-dashed border-white/10 p-8 text-center text-white/30">
              Frame editor — coming soon
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
