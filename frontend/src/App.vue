<script setup lang="ts">
import { ref, nextTick, onMounted, onUnmounted } from 'vue'
import { useChatStore } from './stores/chat'
import MessageBubble from './components/MessageBubble.vue'
import CrawlerStatusDot from './components/CrawlerStatusDot.vue'

const store = useChatStore()
const input = ref('')
const messagesEnd = ref<HTMLElement | null>(null)
const textarea = ref<HTMLTextAreaElement | null>(null)

const SUGGESTIONS = [
  'Give me all BIDS datasets from 3T MRI scanners with at least 50 subjects',
  'Show datasets with Alzheimer\'s disease participants',
  'List all resting-state fMRI datasets',
  'Find datasets with both structural and functional MRI',
]

async function submit() {
  const q = input.value.trim()
  if (!q) return
  input.value = ''
  autoResize()
  await store.sendQuestion(q)
  await nextTick()
  messagesEnd.value?.scrollIntoView({ behavior: 'smooth' })
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    submit()
  }
}

function autoResize() {
  if (!textarea.value) return
  textarea.value.style.height = 'auto'
  textarea.value.style.height = Math.min(textarea.value.scrollHeight, 160) + 'px'
}

function useSuggestion(s: string) {
  input.value = s
  nextTick(() => textarea.value?.focus())
}

// Poll crawler status every 15 s
let crawlerInterval: ReturnType<typeof setInterval>
onMounted(() => {
  store.refreshCrawlerStatus()
  crawlerInterval = setInterval(() => store.refreshCrawlerStatus(), 15_000)
})
onUnmounted(() => clearInterval(crawlerInterval))
</script>

<template>
  <div class="flex h-full overflow-hidden">

    <!-- ── Sidebar ─────────────────────────────────────────────────── -->
    <aside class="hidden md:flex flex-col w-64 bg-panel border-r border-border flex-shrink-0">
      <!-- Logo -->
      <div class="px-5 py-4 border-b border-border">
        <h1 class="text-lg font-bold text-white tracking-tight">
          🧠 BIDS-Eye
        </h1>
        <p class="text-xs text-muted mt-0.5">Neuroimaging dataset search</p>
      </div>

      <!-- New chat -->
      <div class="p-3">
        <button
          class="w-full text-sm text-left px-3 py-2 rounded-lg border border-border hover:bg-surface transition-colors text-muted hover:text-white"
          @click="store.clearMessages()"
        >
          + New search
        </button>
      </div>

      <div class="flex-1" />

      <!-- Crawler status -->
      <div class="border-t border-border">
        <CrawlerStatusDot :status="store.crawlerStatus" />
      </div>
    </aside>

    <!-- ── Main area ───────────────────────────────────────────────── -->
    <div class="flex flex-col flex-1 min-w-0">

      <!-- Mobile header -->
      <div class="md:hidden px-4 py-3 border-b border-border flex items-center justify-between">
        <h1 class="font-bold text-white">🧠 BIDS-Eye</h1>
        <CrawlerStatusDot :status="store.crawlerStatus" />
      </div>

      <!-- Messages -->
      <div class="flex-1 overflow-y-auto px-4 py-6 space-y-6">

        <!-- Empty state -->
        <div
          v-if="!store.messages.length"
          class="flex flex-col items-center justify-center h-full gap-6 text-center"
        >
          <div>
            <p class="text-2xl font-semibold text-white">What datasets are you looking for?</p>
            <p class="text-muted text-sm mt-1">Ask in plain English — BIDS-Eye will search the database.</p>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl">
            <button
              v-for="s in SUGGESTIONS"
              :key="s"
              class="text-left text-sm px-4 py-3 rounded-xl border border-border bg-panel hover:bg-surface hover:border-accent/50 text-muted hover:text-white transition-all"
              @click="useSuggestion(s)"
            >
              {{ s }}
            </button>
          </div>
        </div>

        <!-- Message thread -->
        <template v-else>
          <MessageBubble
            v-for="msg in store.messages"
            :key="msg.id"
            :message="msg"
          />
        </template>

        <div ref="messagesEnd" />
      </div>

      <!-- Input bar -->
      <div class="border-t border-border bg-panel px-4 py-3">
        <div class="max-w-3xl mx-auto flex items-end gap-3">
          <textarea
            ref="textarea"
            v-model="input"
            rows="1"
            placeholder="Describe the datasets you're looking for…"
            class="flex-1 bg-surface border border-border rounded-xl px-4 py-3 text-sm text-white placeholder-muted resize-none focus:outline-none focus:border-accent transition-colors"
            @keydown="onKeydown"
            @input="autoResize"
          />
          <button
            :disabled="!input.trim()"
            class="flex-shrink-0 bg-accent hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl px-4 py-3 text-sm font-medium transition-colors"
            @click="submit"
          >
            Search
          </button>
        </div>
        <p class="text-center text-xs text-muted mt-2">
          Press Enter to search · Shift+Enter for new line
        </p>
      </div>
    </div>
  </div>
</template>
