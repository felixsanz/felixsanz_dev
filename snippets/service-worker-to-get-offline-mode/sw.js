self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open('cache')

    await cache.addAll([
      '/offline',
      '/themes/default.css',
      '/fonts/Nunito.woff2',
    ])

    self.skipWaiting()
  })())
})

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim())
})

self.addEventListener('fetch', (event) => {
  if (
    event.request.method !== 'GET'
    || !event.request.url.includes('felixsanz.dev')
    || event.request.url.endsWith('.xml')
  ) {
    event.respondWith(fetch(event.request))
    return
  }

  event.respondWith((async () => {
    const cache = await caches.open('cache')

    const cached = await caches.match(event.request, { ignoreSearch: true })

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), (cached) ? 6000 : 12000)

    try {
      const response = await fetch(event.request, { signal: controller.signal })

      clearTimeout(timeout)

      if (response.status === 200) {
        await cache.put(event.request, response.clone())
      }

      return response
    } catch (error) {
      if (error.name !== 'AbortError') {
        clearTimeout(timeout)
      }

      return cached || cache.match('/offline')
    }
  })())
})
