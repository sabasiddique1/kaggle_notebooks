/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Suppress the client-side rendering warning for login page (expected behavior)
  experimental: {
    optimizePackageImports: ['lucide-react'],
  },
  async rewrites() {
    // In production (Vercel), API routes are handled by vercel.json
    // In development, proxy to local backend
    if (process.env.NODE_ENV === 'production') {
      return []
    }
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  },
}

module.exports = nextConfig

