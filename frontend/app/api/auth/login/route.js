import { proxy } from "@/lib/proxy";

export async function POST(request) {
  return proxy(request, "/api/users/login");
}
