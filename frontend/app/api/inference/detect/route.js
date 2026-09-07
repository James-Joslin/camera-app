import { proxy } from "@/lib/proxy";

export const dynamic = "force-dynamic";
export async function POST(request) {
  return proxy(
    request,
    "/api/inference/detect",
    process.env.INFERENCE_API_URL || "http://fastapi:8000",
  );
}
