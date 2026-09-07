"use client";

import Hls from "hls.js";
import { useEffect, useRef } from "react";

export default function HlsPlayer({ source, name, autoPlay = true }) {
  const ref = useRef(null);

  useEffect(() => {
    const video = ref.current;
    if (!video || !source) return undefined;
    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = source;
      return undefined;
    }
    if (!Hls.isSupported()) return undefined;
    const player = new Hls({
      liveSyncDurationCount: 2,
      maxLiveSyncPlaybackRate: 1.5,
    });
    player.loadSource(source);
    player.attachMedia(video);
    return () => player.destroy();
  }, [source]);

  return (
    <video
      ref={ref}
      aria-label={`${name} live stream`}
      autoPlay={autoPlay}
      muted
      playsInline
      controls
    />
  );
}
