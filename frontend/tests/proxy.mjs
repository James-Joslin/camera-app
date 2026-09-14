import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const source = await readFile(
  new URL("../lib/proxy.js", import.meta.url),
  "utf8",
);
const { proxy } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`
);
const request = new Request(
  "http://frontend/api/inference/detect?threshold=0.7",
  {
    method: "POST",
    body: "encoded image payload",
    headers: {
      "content-type": "application/octet-stream",
      authorization: "Bearer test",
    },
  },
);
globalThis.fetch = async (url, options) => {
  assert.equal(url, "http://api:8080/api/inference/detect?threshold=0.7");
  assert.equal(options.body, request.body);
  assert.equal(options.duplex, "half");
  assert.equal(options.headers.get("authorization"), "Bearer test");
  return Response.json({ detections: [] });
};
const response = await proxy(
  request,
  "/api/inference/detect",
  "http://api:8080",
);
assert.equal(response.status, 200);
assert.deepEqual(await response.json(), { detections: [] });
console.log("PASS: proxy streams request body and preserves query parameters");
