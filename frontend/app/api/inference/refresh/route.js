import { proxy } from "@/lib/proxy";

export const dynamic = "force-dynamic";
export async function POST(request) {
  return proxy(
    request,
    "/api/inference/refresh",
    process.env.INFERENCE_API_URL || "http://api:8080",
  );
}
