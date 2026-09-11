"use client";

import Hls from "hls.js";
import { useEffect, useRef } from "react";

export default function HlsPlayer({ source, name, autoPlay = true }) {
  const ref = useRef(null);

  useEffect(() => {
    const video = ref.current;
    if (!video || !source) return undefined;

    const seekNativeToLiveEdge = () => {
      if (!video.seekable.length) return;
      const liveEdge = video.seekable.end(video.seekable.length - 1) - 0.25;
      if (liveEdge - video.currentTime > 1.5) video.currentTime = liveEdge;
    };
    const handleNativeVisibility = () => {
      if (!document.hidden && !video.paused) seekNativeToLiveEdge();
    };

    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = source;
      video.addEventListener("loadedmetadata", seekNativeToLiveEdge);
      video.addEventListener("play", seekNativeToLiveEdge);
      document.addEventListener("visibilitychange", handleNativeVisibility);
      return () => {
        video.removeEventListener("loadedmetadata", seekNativeToLiveEdge);
        video.removeEventListener("play", seekNativeToLiveEdge);
        document.removeEventListener(
          "visibilitychange",
          handleNativeVisibility,
        );
        video.removeAttribute("src");
        video.load();
      };
    }

    if (!Hls.isSupported()) return undefined;
    const player = new Hls({
      lowLatencyMode: true,
      liveSyncDurationCount: 1,
      liveMaxLatencyDurationCount: 3,
      maxLiveSyncPlaybackRate: 1.5,
      maxBufferLength: 4,
      backBufferLength: 0,
    });
    const seekToLiveEdge = () => {
      const livePosition = player.liveSyncPosition;
      if (
        !video.paused &&
        video.readyState > 0 &&
        Number.isFinite(livePosition) &&
        livePosition - video.currentTime > 1.5
      ) {
        video.currentTime = livePosition;
      }
    };
    const handleVisibility = () => {
      if (!document.hidden) seekToLiveEdge();
    };

    player.on(Hls.Events.LEVEL_UPDATED, seekToLiveEdge);
    video.addEventListener("play", seekToLiveEdge);
    document.addEventListener("visibilitychange", handleVisibility);
    player.loadSource(source);
    player.attachMedia(video);

    return () => {
      player.off(Hls.Events.LEVEL_UPDATED, seekToLiveEdge);
      video.removeEventListener("play", seekToLiveEdge);
      document.removeEventListener("visibilitychange", handleVisibility);
      player.destroy();
    };
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
