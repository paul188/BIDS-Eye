<template>
  <div class="flex items-center justify-center h-full bg-surface">
    <div class="w-full max-w-sm bg-panel border border-border rounded-2xl shadow-sm p-8 space-y-6">
      <div class="text-center">
        <div class="flex items-center justify-center gap-2 mb-1">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="currentColor" class="text-ink">
            <path d="M12 3 1 8l11 5 9-4.09V14h2V8L12 3zM5 13.18v3L12 20l7-3.82v-3L12 17l-7-3.82z" />
          </svg>
          <h1 class="text-xl text-ink" style="font-weight: 500">BIDS-Eye</h1>
        </div>
        <p class="text-sm text-muted mt-1">Sign in to continue</p>
      </div>

      <a
        href="/api/auth/github"
        class="flex items-center justify-center gap-3 w-full bg-ink hover:opacity-90 text-white rounded-xl px-4 py-3 text-sm font-medium transition-opacity"
      >
        <!-- GitHub mark -->
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.84 1.236 1.84 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.418-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.605-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
        </svg>
        Sign in with GitHub
      </a>
    </div>
  </div>
</template>
