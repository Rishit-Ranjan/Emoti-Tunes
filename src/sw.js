const CACHE_NAME = 'emotitunes-cache-v1';
// This list includes the app shell, local assets, and critical external CDN resources.
const assetsToCache = [
    '/',
    '/index.html',
    '/main.jsx',
    '/App.jsx',
    '/constants.jsx',
    '/services/geminiService.js',
    '/components/EmotionSelector.jsx',
    '/components/PlaylistDisplay.jsx',
    '/components/Loader.jsx',
    '/components/icons/EmotionIcons.jsx',
    '/components/CameraView.jsx',
    '/components/AudioView.jsx',
    '/metadata.json',
    'https://cdn.tailwindcss.com',
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap',
    'https://aistudiocdn.com/react@^19.1.1',
    'https://aistudiocdn.com/react-dom@^19.1.1/client',
    'https://aistudiocdn.com/react@^19.1.1/jsx-runtime',
    'https://aistudiocdn.com/@google/genai@^1.16.0'
];
// Install event: open cache and add all core assets.
self.addEventListener('install', (event) => {
    event.waitUntil(caches.open(CACHE_NAME)
        .then((cache) => {
        console.log('EmotiTunes ServiceWorker: Caching core assets.');
        return cache.addAll(assetsToCache);
    })
        .catch(err => {
        console.error('EmotiTunes ServiceWorker: Failed to cache assets during install:', err);
    }));
});
// Activate event: clean up old caches.
self.addEventListener('activate', (event) => {
    const cacheWhitelist = [CACHE_NAME];
    event.waitUntil(caches.keys().then((cacheNames) => {
        return Promise.all(cacheNames.map((cacheName) => {
            if (cacheWhitelist.indexOf(cacheName) === -1) {
                console.log('EmotiTunes ServiceWorker: Deleting old cache:', cacheName);
                return caches.delete(cacheName);
            }
        }));
    }));
});
// Fetch event: implement a cache-first, then network strategy.
self.addEventListener('fetch', (event) => {
    // We only cache GET requests.
    if (event.request.method !== 'GET') {
        return;
    }
    event.respondWith(caches.match(event.request)
        .then((cachedResponse) => {
        // If we have a cached response, return it.
        if (cachedResponse) {
            return cachedResponse;
        }
        // If not in cache, fetch from the network.
        return fetch(event.request).then((networkResponse) => {
            // Check if we received a valid response
            if (!networkResponse || networkResponse.status !== 200 || networkResponse.type === 'error') {
                return networkResponse;
            }
            // Clone the response stream.
            const responseToCache = networkResponse.clone();
            caches.open(CACHE_NAME)
                .then((cache) => {
                // Don't cache chrome-extension:// requests
                if (!event.request.url.startsWith('chrome-extension')) {
                    cache.put(event.request, responseToCache);
                }
            });
            return networkResponse;
        }).catch(() => {
            // This catch block is triggered when the fetch fails, e.g., offline.
            // The request will fail and the app's error handling logic will take over.
            // No need to return a generic fallback page for assets.
        });
    }));
});
export {};
