import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { api, type Dataset, type CrawlerStatus } from '../api'

export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  datasets: Dataset[]
  sql: string | null
  loading?: boolean
  error?: boolean
}

export interface Conversation {
  id: number
  title: string
  messages: Message[]
  createdAt: string
}

let _nextId = 1

const STORAGE_KEY = 'bids-eye-history'

function persist(conversations: Conversation[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
  } catch { /* quota exceeded — non-fatal */ }
}

function loadFromStorage(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return JSON.parse(raw) as Conversation[]
  } catch { /* corrupted — ignore */ }
  return []
}

function makeConversation(): Conversation {
  return { id: _nextId++, title: 'New search', messages: [], createdAt: new Date().toISOString() }
}

export const useChatStore = defineStore('chat', () => {
  const stored = loadFromStorage()
  const conversations = ref<Conversation[]>(stored.length ? stored : [makeConversation()])
  const currentId = ref<number>(conversations.value[conversations.value.length - 1].id)
  const crawlerStatus = ref<CrawlerStatus | null>(null)

  // Advance _nextId past any IDs already in storage so we never collide
  for (const c of conversations.value) {
    if (c.id >= _nextId) _nextId = c.id + 1
    for (const m of c.messages) {
      if (m.id >= _nextId) _nextId = m.id + 1
    }
  }

  const currentConversation = computed(() =>
    conversations.value.find(c => c.id === currentId.value) ?? conversations.value[0]
  )

  const currentMessages = computed(() => currentConversation.value?.messages ?? [])

  function _save() {
    persist(conversations.value)
  }

  function newConversation() {
    const c = makeConversation()
    conversations.value.push(c)
    currentId.value = c.id
    _save()
  }

  function switchConversation(id: number) {
    currentId.value = id
  }

  function deleteConversation(id: number) {
    const idx = conversations.value.findIndex(c => c.id === id)
    if (idx === -1) return
    conversations.value.splice(idx, 1)
    if (!conversations.value.length) {
      newConversation()
    } else if (currentId.value === id) {
      currentId.value = conversations.value[conversations.value.length - 1].id
    }
    _save()
  }

  async function sendQuestion(question: string) {
    const conv = currentConversation.value
    if (!conv) return

    // Title = first user question, truncated
    if (conv.title === 'New search') {
      conv.title = question.length > 50 ? question.slice(0, 50) + '…' : question
    }

    conv.messages.push({
      id: _nextId++,
      role: 'user',
      content: question,
      datasets: [],
      sql: null,
    })

    const loadingId = _nextId++
    conv.messages.push({
      id: loadingId,
      role: 'assistant',
      content: '',
      datasets: [],
      sql: null,
      loading: true,
    })

    _save()

    try {
      const response = await api.query(question)
      const idx = conv.messages.findIndex(m => m.id === loadingId)
      if (idx !== -1) {
        conv.messages[idx] = {
          id: loadingId,
          role: 'assistant',
          content: response.message,
          datasets: response.datasets,
          sql: response.translation?.sql ?? null,
          loading: false,
        }
      }
    } catch (err) {
      const idx = conv.messages.findIndex(m => m.id === loadingId)
      if (idx !== -1) {
        conv.messages[idx] = {
          id: loadingId,
          role: 'assistant',
          content: `Error: ${err instanceof Error ? err.message : String(err)}`,
          datasets: [],
          sql: null,
          loading: false,
          error: true,
        }
      }
    }

    _save()
  }

  async function refreshCrawlerStatus() {
    try {
      crawlerStatus.value = await api.crawlerStatus()
    } catch { /* non-fatal */ }
  }

  function clearMessages() {
    newConversation()
  }

  return {
    conversations,
    currentId,
    currentMessages,
    crawlerStatus,
    sendQuestion,
    refreshCrawlerStatus,
    newConversation,
    switchConversation,
    deleteConversation,
    clearMessages,
  }
})
