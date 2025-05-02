import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  root: path.resolve(__dirname, 'src'),
  build: {
    outDir: '../static',
    emptyOutDir: true
  },
  server: {
    port: 1234,
    open: true
  }
}); 