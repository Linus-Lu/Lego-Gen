import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      include: ['src/api/**'],
      // Everything else stays out of scope — tested by integration / manual.
      exclude: [
        'src/components/**',
        'src/pages/**',
        'src/App.tsx',
        'src/main.tsx',
        '**/*.test.ts',
        '**/*.test.tsx',
      ],
      thresholds: {
        lines: 100,
        branches: 100,
        functions: 100,
        statements: 100,
      },
    },
  },
});
