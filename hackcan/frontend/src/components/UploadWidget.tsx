'use client';

import Script from 'next/script';
import { useCallback, useRef } from 'react';

export interface CloudinaryUploadResult {
  public_id: string;
  secure_url: string;
  format: string;
  resource_type: string;
  duration?: number;
  width?: number;
  height?: number;
}

interface UploadWidgetProps {
  onUpload: (result: CloudinaryUploadResult) => void;
  children: (open: () => void) => React.ReactNode;
  resourceType?: 'video' | 'image' | 'auto';
}

declare global {
  interface Window {
    cloudinary?: {
      createUploadWidget: (
        options: Record<string, unknown>,
        callback: (error: unknown, result: { event: string; info: CloudinaryUploadResult }) => void
      ) => { open: () => void };
    };
  }
}

export default function UploadWidget({
  onUpload,
  children,
  resourceType = 'video',
}: UploadWidgetProps) {
  const widgetRef = useRef<{ open: () => void } | null>(null);

  const initWidget = useCallback(() => {
    if (!window.cloudinary) return;
    if (widgetRef.current) {
      widgetRef.current.open();
      return;
    }

    widgetRef.current = window.cloudinary.createUploadWidget(
      {
        cloudName: process.env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME,
        uploadPreset: process.env.NEXT_PUBLIC_CLOUDINARY_UPLOAD_PRESET,
        resourceType,
        multiple: false,
        sources: ['local', 'url'],
      },
      (error, result) => {
        if (error) {
          console.error('Upload error:', error);
          return;
        }
        if (result.event === 'success') {
          onUpload(result.info);
        }
      }
    );

    widgetRef.current.open();
  }, [onUpload, resourceType]);

  return (
    <>
      <Script
        src="https://upload-widget.cloudinary.com/global/all.js"
        strategy="lazyOnload"
      />
      {children(initWidget)}
    </>
  );
}
