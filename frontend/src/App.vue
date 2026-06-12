<script setup lang="ts">
import { ref, nextTick, onMounted, onUnmounted } from 'vue'
import { useChatStore } from './stores/chat'
import MessageBubble from './components/MessageBubble.vue'
import CrawlerStatusDot from './components/CrawlerStatusDot.vue'
import LoginView from './components/LoginView.vue'

const CRAWLER_ENABLED = import.meta.env.VITE_CRAWLER_ENABLED === 'true'

const isLoggedIn = ref(!!localStorage.getItem('bids_eye_token'))

function onLoggedIn(token: string) {
  localStorage.setItem('bids_eye_token', token)
  isLoggedIn.value = true
  if (CRAWLER_ENABLED) {
    store.refreshCrawlerStatus()
    crawlerInterval = setInterval(() => store.refreshCrawlerStatus(), 15_000)
  }
}

function logout() {
  localStorage.removeItem('bids_eye_token')
  isLoggedIn.value = false
  clearInterval(crawlerInterval)
}

const store = useChatStore()
const input = ref('')
const messagesContainer = ref<HTMLElement | null>(null)
const textarea = ref<HTMLTextAreaElement | null>(null)
const drawerOpen = ref(false)

const SUGGESTIONS = [
  'Datasets with at least 40 Parkinson\'s patients',
  'Show datasets with Alzheimer\'s disease participants',
  'List all resting-state fMRI datasets',
  'Find datasets with both structural and functional MRI',
]

async function submit() {
  const q = input.value.trim()
  if (!q) return
  input.value = ''
  autoResize()
  // sendQuestion pushes the user message + loading bubble synchronously, then
  // awaits the API. Scroll the new question to the top *before* the answer lands
  // so the user isn't pushed down and never has to scroll back up.
  const pending = store.sendQuestion(q)
  await nextTick()
  scrollQuestionToTop()
  await pending
}

function scrollQuestionToTop() {
  const c = messagesContainer.value
  if (!c) return
  const users = c.querySelectorAll('[data-role="user"]')
  const last = users[users.length - 1] as HTMLElement | undefined
  last?.scrollIntoView({ behavior: 'smooth', block: 'start' })
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

function selectConversation(id: number) {
  store.switchConversation(id)
  drawerOpen.value = false
}

function startNewConversation() {
  store.newConversation()
  drawerOpen.value = false
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

// Poll crawler status every 15 s — only while logged in and crawler is enabled
let crawlerInterval: ReturnType<typeof setInterval>
onMounted(() => {
  if (CRAWLER_ENABLED && isLoggedIn.value) {
    store.refreshCrawlerStatus()
    crawlerInterval = setInterval(() => store.refreshCrawlerStatus(), 15_000)
  }
  window.addEventListener('bids-unauthorized', () => {
    isLoggedIn.value = false
    clearInterval(crawlerInterval)
  })
})
onUnmounted(() => clearInterval(crawlerInterval))
</script>

<template>
  <LoginView v-if="!isLoggedIn" @logged-in="onLoggedIn" />

  <div v-else class="flex flex-col h-full overflow-hidden bg-surface">

    <!-- ── Header bar ─────────────────────────────────────────────────── -->
    <header class="flex items-center gap-3 h-14 px-4 border-b border-border flex-shrink-0">
      <button
        class="p-2 -ml-2 rounded-full text-muted hover:bg-panel-soft transition-colors"
        title="Menu"
        @click="drawerOpen = true"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
          <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      <!-- Wordmark: "BIDS-Eye | Search" -->
      <div class="flex items-center gap-2 select-none">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" class="text-ink">
          <path d="M12 3 1 8l11 5 9-4.09V14h2V8L12 3zM5 13.18v3L12 20l7-3.82v-3L12 17l-7-3.82z" />
        </svg>
        <span class="text-xl text-ink" style="font-weight: 500">BIDS-Eye</span>
        <span class="text-muted-soft text-xl font-light">|</span>
        <span class="text-xl text-accent" style="font-weight: 500">Search</span>
      </div>

      <div class="flex-1" />

      <CrawlerStatusDot v-if="CRAWLER_ENABLED" :status="store.crawlerStatus" />
      <button
        class="p-2 rounded-full text-muted hover:bg-panel-soft transition-colors"
        title="Sign out"
        @click="logout"
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10zm0 2c-4 0-8 2-8 5v1h16v-1c0-3-4-5-8-5z" />
        </svg>
      </button>
    </header>

    <!-- ── History drawer ─────────────────────────────────────────────── -->
    <transition name="fade">
      <div v-if="drawerOpen" class="fixed inset-0 z-30 bg-black/20" @click="drawerOpen = false" />
    </transition>
    <aside
      class="fixed top-0 left-0 z-40 h-full w-72 bg-panel border-r border-border shadow-xl flex flex-col transition-transform duration-200"
      :class="drawerOpen ? 'translate-x-0' : '-translate-x-full'"
    >
      <div class="flex items-center justify-between px-4 h-14 border-b border-border">
        <span class="text-sm font-medium text-ink">Search history</span>
        <button class="p-2 -mr-2 rounded-full text-muted hover:bg-panel-soft" @click="drawerOpen = false">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <line x1="6" y1="6" x2="18" y2="18" /><line x1="6" y1="18" x2="18" y2="6" />
          </svg>
        </button>
      </div>

      <div class="p-3">
        <button
          class="w-full text-sm text-left px-3 py-2 rounded-full border border-border hover:bg-panel-soft transition-colors text-accent"
          @click="startNewConversation"
        >
          + New search
        </button>
      </div>

      <div class="flex-1 overflow-y-auto pb-2">
        <p v-if="!store.conversations.length" class="px-4 py-3 text-xs text-muted italic">No history yet</p>
        <div
          v-for="conv in [...store.conversations].reverse()"
          :key="conv.id"
          class="group relative mx-2 mb-0.5"
        >
          <button
            class="w-full text-left px-3 py-2 rounded-lg text-sm transition-colors flex flex-col gap-0.5 pr-7"
            :class="conv.id === store.currentId
              ? 'bg-[#e8f0fe] text-accent'
              : 'text-ink hover:bg-panel-soft'"
            @click="selectConversation(conv.id)"
          >
            <span class="truncate font-medium leading-snug">{{ conv.title }}</span>
            <span class="text-xs text-muted">{{ relativeTime(conv.createdAt) }}</span>
          </button>
          <button
            class="absolute right-2 top-2.5 opacity-0 group-hover:opacity-100 text-muted hover:text-red-500 transition-all text-xs leading-none"
            title="Delete"
            @click.stop="store.deleteConversation(conv.id)"
          >
            ✕
          </button>
        </div>
      </div>
    </aside>

    <!-- ── Messages ───────────────────────────────────────────────────── -->
    <div ref="messagesContainer" class="flex-1 overflow-y-auto">
      <div class="max-w-3xl mx-auto px-6 sm:px-8 py-8">

        <!-- Empty state -->
        <div
          v-if="!store.currentMessages.length"
          class="flex flex-col items-center text-center pt-12 gap-8"
        >
          <div class="flex items-center gap-3">
            <svg width="34" height="34" viewBox="0 0 24 24" fill="currentColor" class="text-muted-soft">
              <path d="M12 3 1 8l11 5 9-4.09V14h2V8L12 3zM5 13.18v3L12 20l7-3.82v-3L12 17l-7-3.82z" />
            </svg>
            <p class="text-2xl text-muted" style="font-weight: 500">
              Ask a detailed research question to find datasets
            </p>
          </div>

          <div class="w-full text-left">
            <p class="text-sm font-medium text-muted mb-3">Example questions</p>
            <div class="flex flex-col gap-2">
              <button
                v-for="s in SUGGESTIONS"
                :key="s"
                class="text-left text-sm px-4 py-3.5 rounded-lg bg-panel-soft hover:bg-[#ececed] text-accent transition-colors"
                @click="useSuggestion(s)"
              >
                {{ s }}
              </button>
            </div>
          </div>
        </div>

        <!-- Message thread -->
        <div v-else class="flex flex-col gap-6">
          <MessageBubble
            v-for="msg in store.currentMessages"
            :key="msg.id"
            :message="msg"
          />
        </div>
      </div>
    </div>

    <!-- ── Input bar ──────────────────────────────────────────────────── -->
    <div class="px-4 pb-4 pt-2 flex-shrink-0">
      <div class="max-w-3xl mx-auto">
        <div class="flex items-end gap-2 bg-panel border border-border rounded-3xl shadow-sm px-4 py-2 focus-within:border-accent focus-within:shadow-md transition-all">
          <textarea
            ref="textarea"
            v-model="input"
            rows="1"
            placeholder="Ask BIDS-Eye"
            class="flex-1 bg-transparent py-1.5 text-sm text-ink placeholder-muted-soft resize-none focus:outline-none"
            @keydown="onKeydown"
            @input="autoResize"
          />
          <button
            :disabled="!input.trim()"
            class="flex-shrink-0 mb-0.5 p-1.5 rounded-full text-accent hover:bg-panel-soft disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
            title="Search"
            @click="submit"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
              <path d="M3 20.5v-7l8-1.5-8-1.5v-7l18 8z" />
            </svg>
          </button>
        </div>
        <p class="text-center text-xs text-muted-soft mt-2">
          BIDS-Eye is experimental and can make mistakes.
        </p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
