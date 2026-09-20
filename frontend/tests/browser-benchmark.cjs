/* Measure the real HLS -> canvas/JPEG -> frontend proxy -> inference path.
 * Point CAMERA_BENCH_STREAM_URL at a same-origin HLS stream or finite replay.
 */
const fs = require("node:fs");
const { chromium } = require("playwright");
const count = Number(process.env.CAMERA_BENCH_CAMERAS || 4);
const seconds = Number(process.env.CAMERA_BENCH_SECONDS || 30);
const stream = process.env.CAMERA_BENCH_STREAM_URL;
if (
  !stream ||
  !Number.isInteger(count) ||
  count < 1 ||
  count > 32 ||
  !Number.isFinite(seconds) ||
  seconds < 5 ||
  seconds > 600
) {
  throw new Error(
    "Set CAMERA_BENCH_STREAM_URL, 1–32 cameras and 5–600 seconds",
  );
}
function summary(values) {
  const sorted = values.sort((a, b) => a - b);
  return sorted.length
    ? {
        samples: sorted.length,
        p50Ms: sorted[Math.floor(sorted.length / 2)],
        p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1],
        meanMs: sorted.reduce((a, b) => a + b, 0) / sorted.length,
      }
    : { samples: 0 };
}
(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    args: ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1600, height: 1200 },
    });
    await page.addInitScript(() => {
      window.benchmarkSamples = [];
      const canvases = new WeakMap(),
        blobs = new WeakMap(),
        forms = new WeakMap();
      const draw = CanvasRenderingContext2D.prototype.drawImage;
      CanvasRenderingContext2D.prototype.drawImage = function (video, ...args) {
        if (video instanceof HTMLVideoElement)
          canvases.set(this.canvas, {
            camera: video.getAttribute("aria-label"),
            start: performance.now(),
          });
        return draw.call(this, video, ...args);
      };
      const encode = HTMLCanvasElement.prototype.toBlob;
      HTMLCanvasElement.prototype.toBlob = function (callback, ...args) {
        const sample = canvases.get(this);
        return encode.call(
          this,
          (blob) => {
            if (blob && sample)
              blobs.set(blob, { ...sample, encoded: performance.now() });
            callback(blob);
          },
          ...args,
        );
      };
      const append = FormData.prototype.append;
      FormData.prototype.append = function (name, blob, ...args) {
        if (name === "image" && blobs.has(blob))
          forms.set(this, blobs.get(blob));
        return append.call(this, name, blob, ...args);
      };
      const fetchOriginal = window.fetch;
      window.fetch = async function (url, options) {
        const sample = forms.get(options?.body);
        const response = await fetchOriginal.call(this, url, options);
        if (sample) {
          const payload = response.ok ? await response.clone().json() : null;
          window.benchmarkSamples.push({
            camera: sample.camera,
            status: response.status,
            captureToResponseMs: performance.now() - sample.start,
            encodeMs: sample.encoded - sample.start,
            inferenceMs: payload?.inferenceMs,
            releaseId: payload?.releaseId,
            width: payload?.image?.width,
            height: payload?.image?.height,
            detections: payload?.detections?.length,
          });
        }
        return response;
      };
    });
    const cameras = Array.from({ length: count }, (_, i) => ({
      id: `replay-${i}`,
      name: `Replay ${i + 1}`,
      location: "Camera replay benchmark",
      enabled: true,
    }));
    await page.route("**/api/cameras", (route) =>
      route.fulfill({ json: cameras }),
    );
    await page.route("**/api/streams", (route) =>
      route.fulfill({
        json: cameras.map((camera) => ({
          cameraId: camera.id,
          status: "live",
          streamUrl: stream,
        })),
      }),
    );
    await page.goto(process.env.CAMERA_TEST_URL || "http://localhost:3101");
    await page.waitForFunction(
      (n) =>
        document.querySelectorAll("video").length === n &&
        Array.from(document.querySelectorAll("video")).every(
          (video) => video.readyState >= 2 && video.videoWidth > 0,
        ),
      count,
      { timeout: 30000 },
    );
    await page.locator("video").evaluateAll((videos) =>
      videos.forEach((video) => {
        video.loop = true;
        video.play();
      }),
    );
    const buttons = page.getByRole("button", {
      name: "Start detection",
      exact: true,
    });
    for (let i = 0; i < count; i++) await buttons.first().click();
    await page.waitForTimeout(seconds * 1000);
    const samples = await page.evaluate(() => window.benchmarkSamples);
    const perCamera = cameras.map((camera) => {
      const rows = samples
        .filter((row) => row.camera === `${camera.name} live stream`)
        .slice(5);
      const successful = rows.filter((row) => row.status === 200);
      return {
        camera: camera.name,
        warmupsExcluded: 5,
        responses: rows.reduce((out, row) => {
          out[row.status] = (out[row.status] || 0) + 1;
          return out;
        }, {}),
        captureToResponse: summary(
          successful.map((row) => row.captureToResponseMs),
        ),
        encoding: summary(successful.map((row) => row.encodeMs)),
        inference: summary(successful.map((row) => row.inferenceMs)),
      };
    });
    if (perCamera.some((camera) => !camera.captureToResponse.samples))
      throw new Error("Insufficient successful samples after warmup");
    const report = {
      createdAt: new Date().toISOString(),
      stream,
      cameras: count,
      seconds,
      scope:
        "Real browser HLS decode/capture and JPEG upload; timing begins at canvas capture and ends at parsed response. Excludes camera-to-HLS age and overlay paint.",
      hardwareNotes: process.env.CAMERA_BENCH_HARDWARE_NOTES || "not recorded",
      perCamera,
      samples,
    };
    fs.writeFileSync(
      process.env.CAMERA_BENCH_OUTPUT || "/tmp/camera-browser-benchmark.json",
      JSON.stringify(report, null, 2) + "\n",
    );
    console.log(JSON.stringify({ perCamera }, null, 2));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
