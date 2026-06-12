<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{ loggedIn: [token: string] }>()

const password = ref('')
const error = ref('')
const loading = ref(false)

async function submit() {
  error.value = ''
  loading.value = true
  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: password.value }),
    })
    if (!res.ok) {
      error.value = 'Incorrect password.'
      return
    }
    const data = await res.json()
    emit('loggedIn', data.token)
  } catch {
    error.value = 'Could not reach the server.'
  } finally {
    loading.value = false
  }
}
</script>

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
        <p class="text-sm text-muted mt-1">Enter your password to continue</p>
      </div>

      <form class="space-y-4" @submit.prevent="submit">
        <input
          v-model="password"
          type="password"
          placeholder="Password"
          autocomplete="current-password"
          class="w-full bg-panel border border-border rounded-xl px-4 py-3 text-sm text-ink placeholder-muted-soft focus:outline-none focus:border-accent transition-colors"
        />
        <p v-if="error" class="text-red-500 text-xs">{{ error }}</p>
        <button
          type="submit"
          :disabled="!password || loading"
          class="w-full bg-accent hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl px-4 py-3 text-sm font-medium transition-colors"
        >
          {{ loading ? 'Signing in…' : 'Sign in' }}
        </button>
      </form>
    </div>
  </div>
</template>
