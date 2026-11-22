/* Adugbo AI service worker: makes the app installable and quick to reopen.
 *
 * - App shell (index.html, manifest, icons) is cached on install, so the app opens offline.
 * - Built assets under /assets/ have content hashes, so they are cached forever once fetched.
 * - Page navigations go to the network first and fall back to the cached shell when offline.
 * - API calls (/api, /health) are never cached: prices and chats must always be live.
 */
const VERSION = "adugbo-v1";
const SHELL = ["/", "/index.html", "/manifest.webmanifest", "/icons/icon-192.png", "/icons/icon-512.png", "/icons/maskable-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);

  // Live data only: never answer API calls from a cache.
  if (url.origin === self.location.origin && (url.pathname.startsWith("/api/") || url.pathname === "/health")) return;

  // Page loads: network first, cached app shell when offline.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(VERSION).then((cache) => cache.put("/index.html", copy));
          return response;
        })
        .catch(() => caches.match("/index.html")),
    );
    return;
  }

  // Hashed build files: cache first, they never change.
  if (url.origin === self.location.origin && url.pathname.startsWith("/assets/")) {
    event.respondWith(
      caches.match(request).then((hit) => hit || fetch(request).then((response) => {
        const copy = response.clone();
        caches.open(VERSION).then((cache) => cache.put(request, copy));
        return response;
      })),
    );
    return;
  }

  // Fonts and icons: serve the cached copy, refresh it in the background.
  if (url.hostname.endsWith("fonts.googleapis.com") || url.hostname.endsWith("fonts.gstatic.com") || url.pathname.startsWith("/icons/")) {
    event.respondWith(
      caches.open(VERSION).then((cache) =>
        cache.match(request).then((hit) => {
          const refresh = fetch(request).then((response) => {
            if (response.ok || response.type === "opaque") cache.put(request, response.clone());
            return response;
          }).catch(() => hit);
          return hit || refresh;
        })),
    );
  }
});
