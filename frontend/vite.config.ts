import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// In development the Vite server proxies /api to the FastAPI backend, so the browser
// sees a single origin (no CORS setup needed). In production, set VITE_API_BASE_URL
// to the API's public URL at build time, or serve both behind one domain.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_DEV_API_PROXY ?? "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: { "/api": { target, changeOrigin: true } },
    },
    preview: {
      port: 4173,
      proxy: { "/api": { target, changeOrigin: true } },
    },
    test: {
      environment: "jsdom",
      include: ["src/**/*.test.{ts,tsx}"],
    },
  };
});
