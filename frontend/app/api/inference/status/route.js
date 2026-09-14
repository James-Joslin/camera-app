import { proxy } from "@/lib/proxy";

export const dynamic = "force-dynamic";
export async function GET(request) {
  return proxy(
    request,
    "/api/inference/status",
    process.env.INFERENCE_API_URL || "http://api:8080",
  );
}
