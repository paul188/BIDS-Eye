<script setup lang="ts">
import LogoMark from './LogoMark.vue'

// The backend sends the browser back to /?login=<reason> when a sign-in ended
// at GitHub without one (routers/auth.py: LOGIN_CANCELLED, LOGIN_REFUSED).
const LOGIN_NOTICES: Record<string, string> = {
  cancelled: 'Sign-in was cancelled. You can try again whenever you like.',
  refused: 'GitHub did not complete the sign-in. Please try again.',
}

function takeLoginNotice(): string {
  const params = new URLSearchParams(window.location.search)
  const reason = params.get('login')
  if (reason === null) return ''
  // Shown once: a reload or a shared link should not repeat it.
  params.delete('login')
  const query = params.toString()
  window.history.replaceState(null, '', window.location.pathname + (query ? `?${query}` : ''))
  return LOGIN_NOTICES[reason] ?? ''
}

const loginNotice = takeLoginNotice()
</script>

<template>
  <div class="flex items-center justify-center h-full bg-surface px-4">
    <div class="w-full max-w-sm">
      <div class="flex items-center gap-2.5 mb-8">
        <LogoMark :size="30" />
        <span class="text-xl font-semibold tracking-tight text-ink">BIDS-Eye</span>
      </div>

      <h1 class="font-serif text-[1.75rem] leading-tight text-ink">
        Find open neuroimaging datasets by describing them.
      </h1>
      <p class="mt-3 text-sm text-muted leading-relaxed">
        Sign in to search the indexed OpenNeuro datasets. Each account gets 20 searches a day.
      </p>

      <a
        href="/api/auth/github"
        class="mt-8 flex items-center justify-center gap-3 w-full bg-ink hover:bg-ink/90 text-white rounded-xl px-4 py-3 text-sm font-medium transition-colors"
      >
        <!-- GitHub mark -->
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path
            d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.84 1.236 1.84 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.418-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.605-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
          />
        </svg>
        Sign in with GitHub
      </a>
      <p v-if="loginNotice" role="status" class="mt-4 text-sm text-muted leading-relaxed">
        {{ loginNotice }}
      </p>
    </div>
  </div>
</template>
