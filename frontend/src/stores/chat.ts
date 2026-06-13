import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { api, RateLimitError, type Dataset, type CrawlerStatus } from '../api'

export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  datasets: Dataset[]
  sql: string | null
  loading?: boolean
  error?: boolean
  // Pagination state for this result block
  queryId?: string | null
  total?: number
  page?: number
  pageSize?: number
  pageLoading?: boolean
}

export interface Conversation {
  id: number
  title: string
  messages: Message[]
  createdAt: string
}

let _nextId = 1

// Runtime-only page cache — not persisted to localStorage.
// Keyed by messageId → page number → Dataset[].
const _pageCache = new Map<number, Map<number, Dataset[]>>()

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

function _cacheSet(messageId: number, page: number, datasets: Dataset[]) {
  if (!_pageCache.has(messageId)) _pageCache.set(messageId, new Map())
  _pageCache.get(messageId)!.set(page, datasets)
}

async function _preloadPage(messageId: number, queryId: string, page: number, pageSize: number, total: number) {
  const pageCount = Math.ceil(total / pageSize)
  if (page < 1 || page > pageCount) return
  if (_pageCache.get(messageId)?.has(page)) return
  try {
    const response = await api.queryPage(queryId, page, pageSize)
    _cacheSet(messageId, page, response.datasets)
  } catch { /* silent — preload is best-effort */ }
}

function _preloadNeighbors(messageId: number, msg: { queryId?: string | null; pageSize?: number; total?: number }, currentPage: number) {
  if (!msg.queryId || !msg.total || !msg.pageSize) return
  _preloadPage(messageId, msg.queryId, currentPage - 1, msg.pageSize, msg.total)
  _preloadPage(messageId, msg.queryId, currentPage + 1, msg.pageSize, msg.total)
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

    const userMsgId = _nextId++
    conv.messages.push({
      id: userMsgId,
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
          queryId: response.query_id,
          total: response.total,
          page: response.page,
          pageSize: response.page_size,
        }
        // Seed page 1 in the cache and preload page 2 in the background.
        _cacheSet(loadingId, response.page ?? 1, response.datasets)
        _preloadNeighbors(loadingId, conv.messages[idx], response.page ?? 1)
      }
    } catch (err) {
      if (err instanceof RateLimitError) {
        // Remove the optimistic user + loading messages — the query never happened.
        conv.messages = conv.messages.filter(
          m => m.id !== userMsgId && m.id !== loadingId
        )
        _save()
        throw err  // let App.vue show the rate-limit banner
      }
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

  async function loadPage(messageId: number, page: number) {
    // Find the message across all conversations (the result block being paged)
    let msg: Message | undefined
    for (const c of conversations.value) {
      msg = c.messages.find(m => m.id === messageId)
      if (msg) break
    }
    if (!msg || !msg.queryId || msg.pageLoading) return
    if (page === msg.page) return

    // Cache hit — instant render, then preload the next set of neighbors.
    const cached = _pageCache.get(messageId)?.get(page)
    if (cached) {
      msg.datasets = cached
      msg.page = page
      _save()
      _preloadNeighbors(messageId, msg, page)
      return
    }

    msg.pageLoading = true
    try {
      const response = await api.queryPage(msg.queryId, page, msg.pageSize ?? 20)
      _cacheSet(messageId, page, response.datasets)
      msg.datasets = response.datasets
      msg.page = response.page
      msg.total = response.total
      msg.pageSize = response.page_size
      _preloadNeighbors(messageId, msg, page)
    } catch (err) {
      // Expired query (410) or transient failure — keep the current page and
      // tell the user to re-run. Non-fatal: don't clobber existing results.
      msg.content = `Could not load page ${page}: ${err instanceof Error ? err.message : String(err)}. Try running the search again.`
    } finally {
      msg.pageLoading = false
      _save()
    }
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
    loadPage,
    refreshCrawlerStatus,
    newConversation,
    switchConversation,
    deleteConversation,
    clearMessages,
  }
})
