import { readdirSync } from "node:fs";
import { relative, resolve } from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

const SRC_ROOT = resolve(import.meta.dirname, "src");

function collectUiEntries(dir: string): Record<string, string> {
  const entries: Record<string, string> = {};
  for (const item of readdirSync(dir, { withFileTypes: true })) {
    const full = resolve(dir, item.name);
    if (item.isDirectory()) {
      Object.assign(entries, collectUiEntries(full));
      continue;
    }
    if (!/\.(ts|tsx)$/.test(item.name)) {
      continue;
    }
    if (
      item.name === "cosmos.decorator.tsx" ||
      /\.(d\.ts|test\.tsx?|browser\.test\.tsx?|fixture\.tsx|bench\.ts)$/.test(item.name)
    ) {
      continue;
    }
    const key = relative(SRC_ROOT, full)
      .replace(/\.(ts|tsx)$/, "")
      .replaceAll("\\", "/");
    entries[key] = full;
  }
  return entries;
}

export default defineConfig(({ mode }) => ({
  plugins: [tailwindcss(), react()],
  ...(mode === "lib"
    ? {
        build: {
          emptyOutDir: true,
          outDir: "dist",
          target: "es2024",
          minify: false,
          sourcemap: true,
          lib: {
            entry: collectUiEntries(SRC_ROOT),
            formats: ["es"],
          },
          rollupOptions: {
            output: {
              preserveModules: true,
              preserveModulesRoot: SRC_ROOT,
              entryFileNames: "[name].js",
            },
            external: [
              "react",
              "react/jsx-runtime",
              "react-dom",
              "lucide-react",
              "clsx",
              "tailwind-merge",
              "class-variance-authority",
              /^@base-ui\/react/,
              /^@dnam-project\/dnam/,
            ],
          },
        },
      }
    : {}),
  test: {
    include: [
      "src/**/*.test.ts",
      "src/**/*.test.tsx",
      "tests/integration/**/*.test.ts",
      "tests/integration/**/*.test.tsx",
    ],
    benchmark: {
      include: ["src/**/*.bench.ts", "tests/integration/**/*.bench.ts"],
    },
    setupFiles: ["./tests/setup.ts"],
    browser: {
      enabled: true,
      headless: true,
      provider: playwright(),
      instances: [{ browser: "chromium" }],
    },
    coverage: {
      provider: "v8",
      include: ["src/**/*.ts", "src/**/*.tsx"],
      exclude: [
        "src/**/*.d.ts",
        "src/**/index.ts",
        "src/host/types.ts",
        "src/components/rack/models.ts",
        "src/components/workflows/models.ts",
        "src/cosmos.decorator.tsx",
        "src/**/*.fixture.tsx",
        "src/**/*.test.ts",
        "src/**/*.test.tsx",
        "src/**/*.bench.ts",
      ],
      thresholds: {
        lines: 95,
        functions: 95,
        statements: 95,
        branches: 85,
      },
    },
  },
}));
