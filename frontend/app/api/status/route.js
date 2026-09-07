export async function GET() {
  const baseUrl = process.env.CAMERA_API_URL || "http://api:8080";
  try {
    const response = await fetch(`${baseUrl}/status/live`, {
      cache: "no-store",
    });
    return Response.json(
      { api: response.ok ? "ok" : "unavailable" },
      { status: response.ok ? 200 : 503 },
    );
  } catch (error) {
    return Response.json(
      { api: "unavailable", error: error.message },
      { status: 503 },
    );
  }
}
