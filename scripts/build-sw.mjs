/**
 * Workbox build script.
 * Runs after `astro build` and writes `dist/sw.js` with a precache manifest.
 * Pages are network-first so new chapters appear online; assets are cache-first.
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { generateSW } from 'workbox-build';

const root = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.join(root, '..');

const { count, size, warnings } = await generateSW({
	globDirectory: path.join(projectRoot, 'dist/'),
	globPatterns: ['**/*.{html,css,js,woff2,svg,webp,png,ico,json,xml}'],
	swDest: path.join(projectRoot, 'dist/sw.js'),
	skipWaiting: true,
	clientsClaim: true,
	navigateFallback: '/404',
	// Pagefind search index is large and versioned; do not precache it.
	navigateFallbackDenylist: [/^\/pagefind\//],
	runtimeCaching: [
		{
			urlPattern: ({ request }) => request.mode === 'navigate',
			handler: 'NetworkFirst',
			options: {
				cacheName: 'pages',
				networkTimeoutSeconds: 3,
				expiration: { maxEntries: 100, maxAgeSeconds: 60 * 60 * 24 * 30 },
			},
		},
		{
			urlPattern: /\/pagefind\/.*\.(?:js|json|wasm)$/,
			handler: 'StaleWhileRevalidate',
			options: {
				cacheName: 'pagefind',
				expiration: { maxEntries: 50, maxAgeSeconds: 60 * 60 * 24 * 30 },
			},
		},
		{
			urlPattern: /\/pagefind\/(?:index|fragment)\/[^?]+\.pf_[a-z_]+$/,
			handler: 'CacheFirst',
			options: {
				cacheName: 'pagefind-index',
				expiration: { maxEntries: 200, maxAgeSeconds: 60 * 60 * 24 * 30 },
			},
		},
		{
			urlPattern: /\.(?:woff2|css|js|svg|webp|png|ico)$/,
			handler: 'CacheFirst',
			options: {
				cacheName: 'assets',
				expiration: { maxEntries: 200, maxAgeSeconds: 60 * 60 * 24 * 365 },
			},
		},
	],
});

for (const warning of warnings) console.warn(warning);
console.log(`Workbox: precached ${count} files, ${(size / 1024).toFixed(1)} KiB.`);
