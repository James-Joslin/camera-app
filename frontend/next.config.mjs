/** @type {import('next').NextConfig} */
const allowedDevOrigins = (process.env.NEXT_ALLOWED_DEV_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

const nextConfig = {
  output: "standalone",
  ...(allowedDevOrigins.length ? { allowedDevOrigins } : {}),
  async rewrites() {
    const cameraApi = process.env.CAMERA_API_URL || "http://api:8080";
    return [
      { source: "/streams/:path*", destination: `${cameraApi}/streams/:path*` },
    ];
  },
};

export default nextConfig;
