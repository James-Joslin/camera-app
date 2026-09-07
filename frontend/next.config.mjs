/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    const cameraApi = process.env.CAMERA_API_URL || "http://api:8080";
    return [
      { source: "/streams/:path*", destination: `${cameraApi}/streams/:path*` },
    ];
  },
};

export default nextConfig;
