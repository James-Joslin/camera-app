export async function proxy(
  request,
  path,
  baseUrl = process.env.CAMERA_API_URL || "http://api:8080",
) {
  try {
    const headers = new Headers();
    const authorization = request.headers.get("authorization");
    const contentType = request.headers.get("content-type");
    if (authorization) headers.set("authorization", authorization);
    if (contentType) headers.set("content-type", contentType);
    const hasBody = !["GET", "HEAD"].includes(request.method);
    const response = await fetch(`${baseUrl}${path}`, {
      method: request.method,
      headers,
      body: hasBody ? await request.arrayBuffer() : undefined,
      cache: "no-store",
    });
    return new Response(response.body, {
      status: response.status,
      headers: {
        "content-type":
          response.headers.get("content-type") || "application/json",
      },
    });
  } catch (error) {
    console.error(`Proxy request to ${path} failed`, error);
    return Response.json(
      { error: "The camera service is temporarily unavailable." },
      { status: 503 },
    );
  }
}
