<script setup lang="ts">
import { ref, nextTick, onMounted, onUnmounted } from 'vue'
import { useChatStore } from './stores/chat'
import { RateLimitError } from './api'
import MessageBubble from './components/MessageBubble.vue'
import CrawlerStatusDot from './components/CrawlerStatusDot.vue'
import LoginView from './components/LoginView.vue'
import LogoMark from './components/LogoMark.vue'
import InputBar from './components/InputBar.vue'

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

// ── Rate limiting ──────────────────────────────────────────────────────────────
const rateLimitedUntil = ref<string | null>(null)
const countdownDisplay = ref('00:00:00')
let countdownInterval: ReturnType<typeof setInterval> | undefined

function formatCountdown(until: string): string {
  const ms = new Date(until).getTime() - Date.now()
  if (ms <= 0) return '00:00:00'
  const h = Math.floor(ms / 3_600_000)
  const m = Math.floor((ms % 3_600_000) / 60_000)
  const s = Math.floor((ms % 60_000) / 1_000)
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function startCountdown() {
  clearInterval(countdownInterval)
  if (!rateLimitedUntil.value) return
  countdownDisplay.value = formatCountdown(rateLimitedUntil.value)
  countdownInterval = setInterval(() => {
    if (!rateLimitedUntil.value) { clearInterval(countdownInterval); return }
    const remaining = new Date(rateLimitedUntil.value).getTime() - Date.now()
    if (remaining <= 0) {
      rateLimitedUntil.value = null
      clearInterval(countdownInterval)
    } else {
      countdownDisplay.value = formatCountdown(rateLimitedUntil.value)
    }
  }, 1_000)
}

const store = useChatStore()
const input = ref('')
const messagesContainer = ref<HTMLElement | null>(null)
const inputBar = ref<InstanceType<typeof InputBar> | null>(null)
const drawerOpen = ref(false)

const SUGGESTIONS = [
  { tag: 'Clinical',   text: 'Datasets with at least 40 Parkinson\'s patients' },
  { tag: 'Clinical',   text: 'Show datasets with Alzheimer\'s disease participants' },
  { tag: 'Modality',   text: 'List all resting-state fMRI datasets' },
  { tag: 'Multimodal', text: 'Find datasets with both structural and functional MRI' },
]

async function submit() {
  const q = input.value.trim()
  if (!q || rateLimitedUntil.value) return
  input.value = ''
  // sendQuestion pushes the user message + loading bubble synchronously, then
  // awaits the API. Scroll the new question to the top *before* the answer lands
  // so the user isn't pushed down and never has to scroll back up.
  const pending = store.sendQuestion(q)
  await nextTick()
  scrollQuestionToTop()
  try {
    await pending
  } catch (err) {
    if (err instanceof RateLimitError) {
      input.value = q  // restore so the user doesn't lose their question
      rateLimitedUntil.value = err.retryAfter
      startCountdown()
    }
  }
}

function scrollQuestionToTop() {
  const c = messagesContainer.value
  if (!c) return
  const users = c.querySelectorAll('[data-role="user"]')
  const last = users[users.length - 1] as HTMLElement | undefined
  last?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function useSuggestion(s: string) {
  input.value = s
  nextTick(() => inputBar.value?.focus())
}

function selectConversation(id: number) {
  store.switchConversation(id)
  drawerOpen.value = false
}

function startNewConversation() {
  store.newConversation()
  drawerOpen.value = false
  nextTick(() => inputBar.value?.focus())
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
  // Pick up JWT delivered via ?token= after GitHub OAuth redirect
  const params = new URLSearchParams(window.location.search)
  const urlToken = params.get('token')
  if (urlToken) {
    localStorage.setItem('bids_eye_token', urlToken)
    isLoggedIn.value = true
    history.replaceState({}, '', '/')
  }

  if (CRAWLER_ENABLED && isLoggedIn.value) {
    store.refreshCrawlerStatus()
    crawlerInterval = setInterval(() => store.refreshCrawlerStatus(), 15_000)
  }
  window.addEventListener('bids-unauthorized', () => {
    isLoggedIn.value = false
    clearInterval(crawlerInterval)
  })
})
onUnmounted(() => {
  clearInterval(crawlerInterval)
  clearInterval(countdownInterval)
})
</script>

<template>
  <LoginView v-if="!isLoggedIn" @logged-in="onLoggedIn" />

  <div v-else class="flex flex-col h-full overflow-hidden bg-surface">

    <!-- ── Header bar ─────────────────────────────────────────────────── -->
    <header class="flex items-center gap-2 sm:gap-3 h-14 px-3 sm:px-5 border-b border-border bg-surface/90 backdrop-blur flex-shrink-0">
      <button
        class="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-sm text-muted hover:bg-panel-soft hover:text-ink transition-colors"
        title="Search history"
        @click="drawerOpen = true"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v4h4" /><path d="M12 8v4l3 2" />
        </svg>
        <span class="hidden md:inline">History</span>
      </button>

      <button class="flex items-center gap-2.5 select-none" title="New search" @click="startNewConversation">
        <LogoMark :size="26" />
        <span class="text-[17px] font-semibold tracking-tight text-ink">BIDS-Eye</span>
        <span class="hidden lg:inline text-sm text-muted-soft ml-1">Open neuroimaging dataset search</span>
      </button>

      <div class="flex-1" />

      <CrawlerStatusDot v-if="CRAWLER_ENABLED" :status="store.crawlerStatus" />
      <button
        v-if="store.currentMessages.length"
        class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-accent hover:bg-accent-soft transition-colors"
        @click="startNewConversation"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true">
          <path d="M12 5v14M5 12h14" />
        </svg>
        <span class="hidden sm:inline">New search</span>
      </button>
      <button
        class="p-2 rounded-lg text-muted hover:bg-panel-soft hover:text-ink transition-colors"
        title="Sign out"
        @click="logout"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3" /><path d="M10 17l5-5-5-5" /><path d="M15 12H4" />
        </svg>
      </button>
    </header>

    <!-- ── History drawer ─────────────────────────────────────────────── -->
    <transition name="fade">
      <div v-if="drawerOpen" class="fixed inset-0 z-30 bg-ink/20" @click="drawerOpen = false" />
    </transition>
    <aside
      class="fixed top-0 left-0 z-40 h-full w-72 bg-panel border-r border-border shadow-xl flex flex-col transition-transform duration-200"
      :class="drawerOpen ? 'translate-x-0' : '-translate-x-full'"
    >
      <div class="flex items-center justify-between px-4 h-14 border-b border-border">
        <span class="text-sm font-semibold text-ink">Search history</span>
        <button class="p-2 -mr-2 rounded-lg text-muted hover:bg-panel-soft" title="Close" @click="drawerOpen = false">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
            <line x1="6" y1="6" x2="18" y2="18" /><line x1="6" y1="18" x2="18" y2="6" />
          </svg>
        </button>
      </div>

      <div class="p-3">
        <button
          class="w-full inline-flex items-center gap-2 text-sm font-medium px-3 py-2 rounded-lg border border-border hover:bg-accent-soft hover:border-accent/30 transition-colors text-accent"
          @click="startNewConversation"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true">
            <path d="M12 5v14M5 12h14" />
          </svg>
          New search
        </button>
      </div>

      <div class="flex-1 overflow-y-auto pb-2">
        <p v-if="!store.conversations.length" class="px-4 py-3 text-xs text-muted">No history yet</p>
        <div
          v-for="conv in [...store.conversations].reverse()"
          :key="conv.id"
          class="group relative mx-2 mb-0.5"
        >
          <button
            class="w-full text-left px-3 py-2 rounded-lg text-sm transition-colors flex flex-col gap-0.5 pr-7"
            :class="conv.id === store.currentId
              ? 'bg-accent-soft text-accent'
              : 'text-ink hover:bg-panel-soft'"
            @click="selectConversation(conv.id)"
          >
            <span class="truncate font-medium leading-snug">{{ conv.title }}</span>
            <span class="text-xs text-muted">{{ relativeTime(conv.createdAt) }}</span>
          </button>
          <button
            class="absolute right-1 top-1.5 w-7 h-7 inline-flex items-center justify-center rounded-md opacity-0 group-hover:opacity-100 focus-visible:opacity-100 text-muted hover:text-danger-ink hover:bg-danger-bg transition-all text-xs leading-none"
            title="Delete"
            @click.stop="store.deleteConversation(conv.id)"
          >
            ✕
          </button>
        </div>
      </div>
    </aside>

    <!-- ── Main ───────────────────────────────────────────────────────── -->
    <main ref="messagesContainer" class="flex-1 overflow-y-auto">

      <!-- Start page: the search box sits in the page, not at the bottom -->
      <div
        v-if="!store.currentMessages.length"
        class="max-w-5xl mx-auto px-4 sm:px-8 pt-12 sm:pt-24 pb-16"
      >
        <h1 class="font-serif text-[2rem] sm:text-[2.5rem] leading-[1.15] text-ink">
          Find BIDS datasets on OpenNeuro, using semantic search.
        </h1>
        <p class="mt-4 text-base text-muted max-w-3xl leading-relaxed">
          Describe the data you need in plain language. The search matches by meaning,
          not by exact keywords or filter fields, so you can put several criteria in one sentence.
          Each result shows who was scanned, so you can judge the fit at a glance.
        </p>

        <div class="mt-8">
          <InputBar
            ref="inputBar"
            v-model="input"
            large
            :rate-limited-until="rateLimitedUntil"
            :countdown="countdownDisplay"
            placeholder="e.g. resting-state fMRI in adolescents with ADHD"
            @submit="submit"
          />
        </div>

        <div class="mt-12">
          <p class="text-sm font-medium text-ink mb-3">Try one of these</p>
          <div class="grid sm:grid-cols-2 gap-3">
            <button
              v-for="s in SUGGESTIONS"
              :key="s.text"
              class="group text-left px-4 py-3.5 rounded-xl border border-border bg-panel hover:border-accent/40 hover:bg-accent-soft/40 transition-colors flex flex-col gap-1.5"
              @click="useSuggestion(s.text)"
            >
              <span class="text-[11px] font-semibold uppercase tracking-wider text-muted-soft">{{ s.tag }}</span>
              <span class="text-sm text-ink leading-snug group-hover:text-accent">{{ s.text }}</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Results thread -->
      <div v-else class="thread max-w-6xl mx-auto px-4 sm:px-8 pt-8 pb-10 flex flex-col">
        <MessageBubble
          v-for="msg in store.currentMessages"
          :key="msg.id"
          :message="msg"
        />
      </div>
    </main>

    <!-- ── Follow-up input, once there are results ────────────────────── -->
    <div
      v-if="store.currentMessages.length"
      class="flex-shrink-0 border-t border-border bg-surface pt-3 pb-3"
    >
      <div class="max-w-6xl mx-auto px-4 sm:px-8">
        <div class="max-w-4xl">
          <InputBar
            ref="inputBar"
            v-model="input"
            :rate-limited-until="rateLimitedUntil"
            :countdown="countdownDisplay"
            placeholder="Ask a new question"
            @submit="submit"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
