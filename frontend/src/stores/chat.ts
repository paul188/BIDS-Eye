import { defineStore } from 'pinia'
import { ref } from 'vue'
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

let _nextId = 1

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const crawlerStatus = ref<CrawlerStatus | null>(null)

  async function sendQuestion(question: string) {
    // Add user message
    messages.value.push({
      id: _nextId++,
      role: 'user',
      content: question,
      datasets: [],
      sql: null,
    })

    // Add loading placeholder for assistant
    const loadingId = _nextId++
    messages.value.push({
      id: loadingId,
      role: 'assistant',
      content: '',
      datasets: [],
      sql: null,
      loading: true,
    })

    try {
      const response = await api.query(question)
      const idx = messages.value.findIndex(m => m.id === loadingId)
      if (idx !== -1) {
        messages.value[idx] = {
          id: loadingId,
          role: 'assistant',
          content: response.message,
          datasets: response.datasets,
          sql: response.translation?.sql ?? null,
          loading: false,
        }
      }
    } catch (err) {
      const idx = messages.value.findIndex(m => m.id === loadingId)
      if (idx !== -1) {
        messages.value[idx] = {
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
  }

  async function refreshCrawlerStatus() {
    try {
      crawlerStatus.value = await api.crawlerStatus()
    } catch {
      // non-fatal
    }
  }

  function clearMessages() {
    messages.value = []
  }

  return { messages, crawlerStatus, sendQuestion, refreshCrawlerStatus, clearMessages }
})
