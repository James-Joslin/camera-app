import { proxy } from "@/lib/proxy";

export async function DELETE(request, { params }) {
  const { id } = await params;
  return proxy(request, `/api/cameras/${encodeURIComponent(id)}`);
}
