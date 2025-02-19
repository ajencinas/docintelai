/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone', // ✅ Enables standalone mode (reduces Docker image size)
  trailingSlash: true,
  experimental: {
    outputFileTracingRoot: "./", // Only needed if deploying outside the default root
  },
};

export default nextConfig;


