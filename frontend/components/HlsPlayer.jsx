"use client";

import Hls from "hls.js";
import { useEffect, useRef, useState } from "react";

export default function HlsPlayer({ source, name, autoPlay = true }) {
  const ref = useRef(null);
  const [detecting, setDetecting] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState(null);
  const [message, setMessage] = useState("Detection stopped");

  useEffect(() => {
    setResult(null);
    if (!detecting || !source) return;
    let disposed = false;
    let timer;
    let expiry;
    let controller;
    let lastCapturedAt = null;
    const canvas = document.createElement("canvas");
    const video = ref.current;
    let generation = 0;
    const clear = () => setResult(null);
    const invalidate = () => {
      generation += 1;
      clear();
    };
    video.addEventListener("pause", invalidate);
    video.addEventListener("seeking", invalidate);
    document.addEventListener("visibilitychange", invalidate);
    const tick = async () => {
      let delay = 500;
      try {
        if (
          document.hidden ||
          video.paused ||
          video.seeking ||
          video.readyState < 2 ||
          !video.videoWidth ||
          !video.videoHeight
        ) {
          clear();
          return;
        }
        if (video.currentTime === lastCapturedAt) {
          clear();
          setMessage("Waiting for a new video frame");
          return;
        }
        // Capture only when ready to submit: no queued frames or overlapping requests.
        const scale = Math.min(
          1,
          1280 / Math.max(video.videoWidth, video.videoHeight),
        );
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        const capturedAt = video.currentTime;
        lastCapturedAt = capturedAt;
        const started = performance.now();
        const capturedGeneration = generation;
        canvas
          .getContext("2d")
          .drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise((resolve) =>
          canvas.toBlob(resolve, "image/jpeg", 0.85),
        );
        if (
          disposed ||
          !blob ||
          capturedGeneration !== generation ||
          performance.now() - started > 1000
        )
          return;
        controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);
        try {
          const form = new FormData();
          form.append("image", blob, "frame.jpg");
          const response = await fetch(
            `/api/inference/detect?threshold=${threshold}`,
            {
              method: "POST",
              body: form,
              signal: controller.signal,
            },
          );
          if (disposed) return;
          if (response.status === 429) {
            clear();
            setMessage("Busy — dropping frame");
            delay = 1000;
            return;
          }
          if (!response.ok)
            throw new Error(`Detection unavailable (${response.status})`);
          const payload = await response.json();
          if (disposed) return;
          if (
            document.hidden ||
            video.paused ||
            video.seeking ||
            Math.abs(video.currentTime - capturedAt) > 1 ||
            capturedGeneration !== generation ||
            performance.now() - started > 1000
          ) {
            clear();
            setMessage("Dropping delayed result");
            return;
          }
          setResult(payload);
          setMessage(
            `${payload.detections.length} ${payload.detections.length === 1 ? "person" : "people"} · ${Math.round(performance.now() - started)} ms capture-to-response · ${Math.round(payload.inferenceMs)} ms inference`,
          );
          clearTimeout(expiry);
          expiry = setTimeout(
            clear,
            Math.max(0, 1000 - (performance.now() - started)),
          );
        } finally {
          clearTimeout(timeout);
        }
      } catch (error) {
        if (!disposed) {
          clear();
          setMessage(
            error.name === "AbortError" ? "Detection timed out" : error.message,
          );
          delay = 2000;
        }
      } finally {
        if (!disposed) timer = setTimeout(tick, delay);
      }
    };
    setMessage("Waiting for video");
    tick();
    return () => {
      disposed = true;
      clearTimeout(timer);
      clearTimeout(expiry);
      controller?.abort();
      video.removeEventListener("pause", invalidate);
      video.removeEventListener("seeking", invalidate);
      document.removeEventListener("visibilitychange", invalidate);
    };
  }, [detecting, source, threshold]);

  useEffect(() => {
    const video = ref.current;
    if (!video || !source) return undefined;

    if (Hls.isSupported()) {
      const player = new Hls({
        preferManagedMediaSource: true,
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
          livePosition - video.currentTime > 0.75
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
    }

    if (!video.canPlayType("application/vnd.apple.mpegurl")) return undefined;

    const seekNativeToLiveEdge = () => {
      if (!video.seekable.length) return;
      const liveEdge = video.seekable.end(video.seekable.length - 1) - 0.25;
      if (liveEdge - video.currentTime > 1.25) video.currentTime = liveEdge;
    };
    const handleNativeVisibility = () => {
      if (!document.hidden && !video.paused) seekNativeToLiveEdge();
    };
    const nativeSyncEvents = [
      "loadedmetadata",
      "durationchange",
      "canplay",
      "progress",
      "play",
    ];

    video.src = source;
    nativeSyncEvents.forEach((eventName) =>
      video.addEventListener(eventName, seekNativeToLiveEdge),
    );
    document.addEventListener("visibilitychange", handleNativeVisibility);
    const nativeSyncInterval = window.setInterval(() => {
      if (!document.hidden && !video.paused) seekNativeToLiveEdge();
    }, 1000);

    return () => {
      window.clearInterval(nativeSyncInterval);
      nativeSyncEvents.forEach((eventName) =>
        video.removeEventListener(eventName, seekNativeToLiveEdge),
      );
      document.removeEventListener("visibilitychange", handleNativeVisibility);
      video.removeAttribute("src");
      video.load();
    };
  }, [source]);

  return (
    <div className="detection-player">
      <video
        ref={ref}
        aria-label={`${name} live stream`}
        autoPlay={autoPlay}
        muted
        playsInline
        controls
        crossOrigin="anonymous"
      />
      {detecting && result && (
        <svg
          className="detection-boxes"
          viewBox={`0 0 ${result.image.width} ${result.image.height}`}
          preserveAspectRatio="xMidYMid meet"
          aria-label="Person detections"
        >
          {result.detections.map((detection, index) => {
            const [x1, y1, x2, y2] = detection.box;
            return (
              <g key={index}>
                <rect
                  x={x1}
                  y={y1}
                  width={Math.max(0, x2 - x1)}
                  height={Math.max(0, y2 - y1)}
                />
                <text x={x1 + 3} y={Math.max(16, y1 - 5)}>
                  {Math.round(detection.confidence * 100)}%
                </text>
              </g>
            );
          })}
        </svg>
      )}
      <div className="detection-controls">
        <button type="button" onClick={() => setDetecting(!detecting)}>
          {detecting ? "Stop detection" : "Start detection"}
        </button>
        <label>
          Confidence {Math.round(threshold * 100)}%{" "}
          <input
            aria-label={`${name} confidence`}
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
          />
        </label>
        <span role="status">{detecting ? message : "Detection stopped"}</span>
      </div>
    </div>
  );
}
