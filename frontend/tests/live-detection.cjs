/* Run with Playwright and Chromium available; see scripts/check-live-detection.sh.
 * Synthetic decoded video isolates capture scheduling from RTSP availability.
 */
const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1280, height: 900 },
    });
    page.on("pageerror", (error) => console.error("PAGE ERROR", error.message));
    page.on("console", (message) => {
      if (message.type() === "error") console.error("BROWSER", message.text());
    });
    await page.addInitScript(() => {
      window.captureCount = 0;
      window.testPaused = false;
      window.testHidden = false;
      Object.defineProperty(document, "hidden", {
        get: () => window.testHidden,
      });
      Object.defineProperties(HTMLVideoElement.prototype, {
        videoWidth: { get: () => 1920 },
        videoHeight: { get: () => 1080 },
        readyState: { get: () => 4 },
        paused: { get: () => window.testPaused },
        seeking: { get: () => false },
        currentTime: {
          get: () => window.testFixedTime ?? performance.now() / 1000,
          set: () => {},
        },
      });
      const original = CanvasRenderingContext2D.prototype.drawImage;
      CanvasRenderingContext2D.prototype.drawImage = function (
        source,
        ...args
      ) {
        if (source instanceof HTMLVideoElement) {
          window.captureCount++;
          this.fillStyle = `rgb(${window.captureCount % 255},40,60)`;
          this.fillRect(0, 0, this.canvas.width, this.canvas.height);
        } else original.call(this, source, ...args);
      };
    });
    await page.route("**/api/cameras", (route) =>
      route.fulfill({
        json: [
          {
            id: "test",
            name: "Test camera",
            location: "Fixture",
            enabled: true,
          },
        ],
      }),
    );
    await page.route("**/api/streams", (route) =>
      route.fulfill({
        json: [
          {
            cameraId: "test",
            status: "live",
            streamUrl: "/streams/test/index.m3u8",
          },
        ],
      }),
    );
    await page.route("**/streams/test/**", (route) => route.abort());
    let mode = "success",
      requests = 0,
      active = 0,
      maxActive = 0;
    const uploads = [];
    const thresholds = [];
    await page.route("**/api/inference/detect?*", async (route) => {
      requests++;
      active++;
      maxActive = Math.max(maxActive, active);
      const localMode = mode;
      uploads.push(route.request().postDataBuffer());
      thresholds.push(
        new URL(route.request().url()).searchParams.get("threshold"),
      );
      try {
        await delay(localMode === "slow" ? 1400 : 80);
        await route.fulfill(
          localMode === "busy"
            ? { status: 429, json: { detail: "busy" } }
            : {
                json: {
                  image: { width: 1280, height: 720 },
                  inferenceMs: 12,
                  detections: [
                    {
                      label: "person",
                      confidence: 0.91,
                      box: [128, 72, 640, 648],
                    },
                  ],
                },
              },
        );
      } catch {
        /* Cancellation is expected after stop/unmount. */
      } finally {
        active--;
      }
    });
    await page.goto(process.env.CAMERA_TEST_URL || "http://localhost:3101");
    await page
      .getByRole("button", { name: "Start detection", exact: true })
      .click();
    await page.locator(".detection-boxes rect").waitFor();
    assert.match(
      await page.locator(".detection-controls [role=status]").textContent(),
      /capture-to-response/,
    );
    assert.ok(uploads[0].includes(Buffer.from('name="image"')));
    assert.ok(uploads[0].includes(Buffer.from([0xff, 0xd8])));
    await page.locator("input[type=range]").fill("0.75");
    await page.waitForFunction(() =>
      document.querySelector(".detection-boxes rect"),
    );
    await delay(700);
    assert.equal(thresholds.at(-1), "0.75");
    // The video and SVG must share contain/meet letterboxing at any viewport ratio.
    await page.locator(".viewport").evaluate((el) => {
      el.style.aspectRatio = "1/1";
    });
    const geometry = await page.locator(".detection-boxes").evaluate((svg) => {
      const video = svg.parentElement.querySelector("video");
      const rect = video.getBoundingClientRect();
      const box = svg.querySelector("rect").getBoundingClientRect();
      const scale = Math.min(rect.width / 1280, rect.height / 720);
      return {
        fit: getComputedStyle(video).objectFit,
        actual: [box.x, box.y, box.width, box.height],
        expected: [
          rect.x + (rect.width - 1280 * scale) / 2 + 128 * scale,
          rect.y + (rect.height - 720 * scale) / 2 + 72 * scale,
          512 * scale,
          576 * scale,
        ],
      };
    });
    assert.equal(geometry.fit, "contain");
    geometry.actual.forEach((value, i) =>
      assert.ok(Math.abs(value - geometry.expected[i]) < 2),
    );
    mode = "busy";
    await page.getByText("Busy — dropping frame").waitFor();
    assert.equal(await page.locator(".detection-boxes").count(), 0);
    await delay(1200);
    assert.ok(
      !uploads.at(-1).equals(uploads.at(-2)),
      "429 must capture a fresh frame, not retry bytes",
    );
    mode = "slow";
    await page.getByText("Dropping delayed result").waitFor();
    assert.equal(await page.locator(".detection-boxes").count(), 0);
    assert.equal(maxActive, 1, "one request at a time per player");
    await page
      .getByRole("button", { name: "Stop detection", exact: true })
      .click();
    const stoppedCount = requests;
    await delay(1800);
    assert.equal(requests, stoppedCount);
    assert.equal(await page.locator(".detection-boxes").count(), 0);
    mode = "success";
    await page
      .getByRole("button", { name: "Start detection", exact: true })
      .click();
    await page.locator(".detection-boxes rect").waitFor();
    await page.evaluate(() => {
      window.testPaused = true;
      document.querySelector("video").dispatchEvent(new Event("pause"));
    });
    assert.equal(await page.locator(".detection-boxes").count(), 0);
    await delay(700);
    const pausedCount = requests;
    await delay(700);
    assert.equal(requests, pausedCount);
    await page.evaluate(() => {
      window.testPaused = false;
    });
    await page.locator(".detection-boxes rect").waitFor();
    await page.screenshot({ path: "/tmp/camera-live-detection.png" });
    await page.evaluate(() => {
      window.testHidden = true;
      document.dispatchEvent(new Event("visibilitychange"));
    });
    assert.equal(await page.locator(".detection-boxes").count(), 0);
    await delay(700);
    const hiddenCount = requests;
    await delay(700);
    assert.equal(requests, hiddenCount);
    await page.evaluate(() => {
      window.testHidden = false;
      document.dispatchEvent(new Event("visibilitychange"));
    });
    await page.locator(".detection-boxes rect").waitFor();
    const pending = page.waitForRequest((request) =>
      request.url().includes("/api/inference/detect"),
    );
    await pending;
    await page.evaluate(() =>
      document.querySelector("video").dispatchEvent(new Event("seeking")),
    );
    await page.getByText("Dropping delayed result").waitFor();
    assert.equal(await page.locator(".detection-boxes").count(), 0);
    await page.evaluate(() => {
      window.testFixedTime = 100;
    });
    await page.getByText("Waiting for a new video frame").waitFor();
    const stalledCount = requests;
    await delay(1100);
    assert.equal(
      requests,
      stalledCount,
      "stalled video must not resubmit the same frame",
    );
    await page.evaluate(() => {
      delete window.testFixedTime;
    });
    mode = "slow";
    await page.waitForRequest((request) =>
      request.url().includes("/api/inference/detect"),
    );
    await page.getByRole("tab", { name: "Settings", exact: true }).click();
    await page.locator("video").waitFor({ state: "detached" });
    const unmountedCount = requests;
    await delay(1800);
    assert.equal(requests, unmountedCount);
    console.log(
      "PASS: JPEG capture, serial requests, confidence, letterboxing, 429 fresh-frame drop, stale rejection, stop, pause, visibility, seek, stalled video and unmount",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
