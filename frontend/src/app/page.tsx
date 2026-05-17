"use client";

import { useCallback, useState } from "react";
import Header from "@/components/layout/Header";
import VideoUploader from "@/components/video/VideoUploader";
import VideoPlayer from "@/components/video/VideoPlayer";
import DashboardPanel from "@/components/dashboard/DashboardPanel";

export default function Home() {
  const [video, setVideo] = useState<{
    file: File;
    previewUrl: string;
  } | null>(null);

  const handleVideoSelect = useCallback((file: File, previewUrl: string) => {
    setVideo({ file, previewUrl });
  }, []);

  const handleReset = useCallback(() => {
    if (video?.previewUrl) {
      URL.revokeObjectURL(video.previewUrl);
    }
    setVideo(null);
  }, [video]);

  return (
    <div className="flex flex-col h-full min-h-screen">
      <Header />
      <main className="flex flex-1 flex-col lg:flex-row gap-4 p-4">
        {video ? (
          <VideoPlayer src={video.previewUrl} onReset={handleReset} />
        ) : (
          <VideoUploader onVideoSelect={handleVideoSelect} />
        )}
        <DashboardPanel />
      </main>
    </div>
  );
}
