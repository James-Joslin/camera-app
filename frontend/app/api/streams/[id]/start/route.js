import { proxy } from "@/lib/proxy";

export async function POST(request, { params }) {
  const { id } = await params;
  return proxy(request, `/api/streams/${encodeURIComponent(id)}/start`);
}
