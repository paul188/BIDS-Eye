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

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  return new Date(iso).toLocaleDateString()
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

      <!-- New search button -->
      <div class="p-3">
        <button
          class="w-full text-sm text-left px-3 py-2 rounded-lg border border-border hover:bg-surface hover:border-accent/50 transition-colors text-muted hover:text-white"
          @click="store.newConversation()"
        >
          + New search
        </button>
      </div>

      <div class="h-px bg-border mx-3" />

      <!-- Conversation history list -->
      <div class="flex-1 overflow-y-auto py-2">
        <p v-if="!store.conversations.length" class="px-4 py-3 text-xs text-muted italic">No history yet</p>
        <div
          v-for="conv in [...store.conversations].reverse()"
          :key="conv.id"
          class="group relative mx-2 mb-0.5"
        >
          <button
            class="w-full text-left px-3 py-2 rounded-lg text-sm transition-colors flex flex-col gap-0.5 pr-7"
            :class="conv.id === store.currentId
              ? 'bg-accent/20 border-l-2 border-accent text-white'
              : 'text-muted hover:bg-surface hover:text-white border-l-2 border-transparent'"
            @click="store.switchConversation(conv.id)"
          >
            <span class="truncate font-medium leading-snug">{{ conv.title }}</span>
            <span class="text-xs opacity-60">{{ relativeTime(conv.createdAt) }}</span>
          </button>
          <button
            class="absolute right-2 top-2.5 opacity-0 group-hover:opacity-100 text-muted hover:text-red-400 transition-all text-xs leading-none"
            title="Delete"
            @click.stop="store.deleteConversation(conv.id)"
          >
            ✕
          </button>
        </div>
      </div>

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
          v-if="!store.currentMessages.length"
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
            v-for="msg in store.currentMessages"
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
