import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { api, RateLimitError, SearchExpiredError, type Dataset, type QueryFilter, type QueryResponse } from '../api'
import { useLatestRequest } from '../composables/useLatestRequest'

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
  // Filter chips: how the question was understood (display only).
  filters?: QueryFilter[]
  // Set only on searches saved when chips could still be removed.
  removedFilters?: string[]
}

export interface Conversation {
  id: number
  title: string
  messages: Message[]
  createdAt: string
  // The saved search on the server this page is (backend search_history),
  // for a user who turned on "Remember my searches and search results".
  historyId?: string
  // Came from the server's list, not from a search in this browser: holds
  // only the question until it is opened, and is never written to localStorage.
  fromServer?: boolean
}

let _nextId = 1

// Runtime-only page cache — not persisted to localStorage.
// Keyed by messageId → page number → Dataset[].
const _pageCache = new Map<number, Map<number, Dataset[]>>()

// One page request per result block: asking for another page drops the one
// still on its way.
const _pageRequests = new Map<number, ReturnType<typeof useLatestRequest>>()

function _pageRequest(messageId: number) {
  if (!_pageRequests.has(messageId)) _pageRequests.set(messageId, useLatestRequest())
  return _pageRequests.get(messageId)!
}

const STORAGE_KEY = 'bids-eye-history'

// Searches the server sent stay out of the browser's storage: the server is
// where they are kept, and the next person signing in here must not find them.
function persist(conversations: Conversation[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations.filter(c => !c.fromServer)))
  } catch { /* quota exceeded — non-fatal */ }
}

function loadFromStorage(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return (JSON.parse(raw) as Conversation[]).map(withoutInterrupted)
  } catch { /* corrupted — ignore */ }
  return []
}

// A search still loading when the page was closed never gets its answer;
// a result page still loading then is not loading any more.
function withoutInterrupted(conv: Conversation): Conversation {
  const messages = conv.messages.map(m => m.loading
    ? { ...m, loading: false, error: true, content: 'This search was interrupted. Run it again to see its results.' }
    : m.pageLoading ? { ...m, pageLoading: false } : m)
  return { ...conv, messages }
}

function makeConversation(): Conversation {
  return { id: _nextId++, title: 'New search', messages: [], createdAt: new Date().toISOString() }
}

function answerMessage(id: number, response: QueryResponse): Message {
  return {
    id,
    role: 'assistant',
    content: response.message,
    datasets: response.datasets,
    sql: response.translation?.sql ?? null,
    loading: false,
    queryId: response.query_id,
    total: response.total,
    page: response.page,
    pageSize: response.page_size,
    filters: response.filters ?? [],
  }
}

function errorMessage(id: number, err: unknown): Message {
  return {
    id,
    role: 'assistant',
    content: `Error: ${err instanceof Error ? err.message : String(err)}`,
    datasets: [],
    sql: null,
    loading: false,
    error: true,
  }
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
  const stored = loadFromStorage().filter(c => c.messages.length)

  // Advance _nextId past any IDs already in storage so we never collide
  for (const c of stored) {
    if (c.id >= _nextId) _nextId = c.id + 1
    for (const m of c.messages) {
      if (m.id >= _nextId) _nextId = m.id + 1
    }
  }

  // Opening the site starts on an empty page; earlier searches are in History.
  const conversations = ref<Conversation[]>([...stored, makeConversation()])
  const currentId = ref<number>(conversations.value[conversations.value.length - 1].id)

  const currentConversation = computed(() =>
    conversations.value.find(c => c.id === currentId.value) ?? conversations.value[0]
  )

  const currentMessages = computed(() => currentConversation.value?.messages ?? [])

  function _save() {
    persist(conversations.value)
  }

  // An empty page is reused, so History never fills with empty entries.
  function newConversation() {
    const current = currentConversation.value
    if (current && !current.messages.length) return
    const c = makeConversation()
    conversations.value.push(c)
    currentId.value = c.id
    _save()
  }

  function switchConversation(id: number) {
    currentId.value = id
    const conv = conversations.value.find(c => c.id === id)
    if (conv) void _openSaved(conv)
  }

  // ── Saved searches on the server ──────────────────────────────────────────
  // Off until the user turns it on (the switch in the history panel). The
  // backend then saves each search as it answers it; this browser only reads
  // and deletes. The rules are in backend/services/stores/search_history.py.
  const rememberOnServer = ref(false)
  const retentionDays = ref(365)
  const rememberError = ref('')

  // A page from the server's list has only its question: fetch the answer
  // the first time it is opened.
  async function _openSaved(conv: Conversation) {
    if (!conv.fromServer || !conv.historyId || conv.messages.length !== 1) return
    const loadingId = _nextId++
    conv.messages.push({ id: loadingId, role: 'assistant', content: '', datasets: [], sql: null, loading: true })
    let answer: Message
    try {
      const entry = await api.historyEntry(conv.historyId)
      answer = answerMessage(loadingId, entry.answer)
      _cacheSet(loadingId, entry.answer.page ?? 1, entry.answer.datasets)
    } catch (err) {
      answer = errorMessage(loadingId, err)
    }
    const idx = conv.messages.findIndex(m => m.id === loadingId)
    if (idx !== -1) conv.messages[idx] = answer
  }

  function _removeServerPages() {
    conversations.value = conversations.value.filter(c => !c.fromServer)
    if (!conversations.value.length) conversations.value.push(makeConversation())
    if (!conversations.value.some(c => c.id === currentId.value)) {
      currentId.value = conversations.value[conversations.value.length - 1].id
    }
  }

  // Called once the user is signed in: whether saving is on, and the saved
  // searches this browser does not have yet, each as a page of its own.
  async function loadServerHistory() {
    let listing
    try {
      listing = await api.history()
    } catch {
      return  // the browser's own history still works
    }
    rememberOnServer.value = listing.enabled
    retentionDays.value = listing.retention_days
    const onServer = new Set(listing.entries.map(e => e.id))
    const here = new Set(conversations.value.map(c => c.historyId).filter(Boolean))
    // Deleted elsewhere, or expired: no longer a saved search.
    conversations.value = conversations.value.filter(
      c => !(c.fromServer && c.historyId && !onServer.has(c.historyId)) || c.id === currentId.value)
    for (const entry of listing.entries) {
      if (here.has(entry.id)) continue
      const q = entry.question
      conversations.value.push({
        id: _nextId++,
        title: q.length > 50 ? q.slice(0, 50) + '…' : q,
        messages: [{ id: _nextId++, role: 'user', content: q, datasets: [], sql: null }],
        createdAt: entry.asked_at,
        historyId: entry.id,
        fromServer: true,
      })
    }
    // History lists the array newest last; the empty start page stays current.
    conversations.value.sort((a, b) => Date.parse(a.createdAt) - Date.parse(b.createdAt))
  }

  // Turning it off deletes the saved searches on the server; what was
  // searched in this browser stays in this browser's history.
  async function setRememberOnServer(enabled: boolean) {
    rememberError.value = ''
    try {
      await api.setHistoryEnabled(enabled)
    } catch (err) {
      rememberError.value = err instanceof Error ? err.message : String(err)
      return
    }
    rememberOnServer.value = enabled
    if (enabled) {
      await loadServerHistory()  // turned on before, in another browser
    } else {
      _removeServerPages()
      for (const c of conversations.value) delete c.historyId
      _save()
    }
  }

  // Signing out: the next person at this browser must not see them.
  function forgetServerHistory() {
    rememberOnServer.value = false
    rememberError.value = ''
    _removeServerPages()
  }

  function deleteConversation(id: number) {
    const idx = conversations.value.findIndex(c => c.id === id)
    if (idx === -1) return
    const historyId = conversations.value[idx].historyId
    // Best effort: if the server copy survives, it is listed again next time.
    if (historyId && rememberOnServer.value) api.deleteHistoryEntry(historyId).catch(() => {})
    conversations.value.splice(idx, 1)
    if (!conversations.value.length) {
      newConversation()
    } else if (currentId.value === id) {
      currentId.value = conversations.value[conversations.value.length - 1].id
    }
    _save()
  }

  // Each search is a page of its own, listed in History, as chat apps keep
  // separate conversations: a new question opens a new page.
  async function sendQuestion(question: string) {
    if (currentConversation.value?.messages.length) newConversation()
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
      if (response.history_id) conv.historyId = response.history_id
      if (idx !== -1) {
        conv.messages[idx] = answerMessage(loadingId, response)
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
      if (idx !== -1) conv.messages[idx] = errorMessage(loadingId, err)
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
    if (!msg || !msg.queryId) return
    const request = _pageRequest(messageId)
    // Back to the page already shown: only a load on its way is dropped.
    if (page === msg.page) {
      request.cancel()
      msg.pageLoading = false
      return
    }

    // Cache hit — instant render, then preload the next set of neighbors.
    const cached = _pageCache.get(messageId)?.get(page)
    if (cached) {
      request.cancel()
      msg.pageLoading = false
      msg.datasets = cached
      msg.page = page
      _save()
      _preloadNeighbors(messageId, msg, page)
      return
    }

    msg.pageLoading = true
    const { queryId, pageSize = 20 } = msg
    try {
      const result = await request.run(signal => api.queryPage(queryId, page, pageSize, signal))
      // A newer call for this result block took over; pageLoading is its to clear.
      if (result.superseded) return
      const response = result.value
      _cacheSet(messageId, page, response.datasets)
      msg.datasets = response.datasets
      msg.page = response.page
      msg.total = response.total
      msg.pageSize = response.page_size
      _preloadNeighbors(messageId, msg, page)
    } catch (err) {
      // Expired query (410) or transient failure — keep the current page and
      // tell the user to re-run. Non-fatal: don't clobber existing results.
      msg.content = err instanceof SearchExpiredError
        ? err.message
        : `Could not load page ${page}: ${err instanceof Error ? err.message : String(err)}. Try running the search again.`
    }
    msg.pageLoading = false
    _save()
  }

  function clearMessages() {
    newConversation()
  }

  return {
    conversations,
    currentId,
    currentMessages,
    sendQuestion,
    loadPage,
    newConversation,
    switchConversation,
    deleteConversation,
    clearMessages,
    rememberOnServer,
    retentionDays,
    rememberError,
    loadServerHistory,
    setRememberOnServer,
    forgetServerHistory,
  }
})
