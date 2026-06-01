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
  <div class="flex items-center justify-center h-full bg-background">
    <div class="w-full max-w-sm bg-panel border border-border rounded-2xl p-8 space-y-6">
      <div class="text-center">
        <p class="text-3xl mb-1">🧠</p>
        <h1 class="text-xl font-bold text-white">BIDS-Eye</h1>
        <p class="text-sm text-muted mt-1">Enter your password to continue</p>
      </div>

      <form class="space-y-4" @submit.prevent="submit">
        <input
          v-model="password"
          type="password"
          placeholder="Password"
          autocomplete="current-password"
          class="w-full bg-surface border border-border rounded-xl px-4 py-3 text-sm text-white placeholder-muted focus:outline-none focus:border-accent transition-colors"
        />
        <p v-if="error" class="text-red-400 text-xs">{{ error }}</p>
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
