export interface Participant {
  participant_id: string
  age: number | null
  sex: string | null
  handedness: string | null
  diagnosis: string | null
  extra: Record<string, unknown> | null
}

export interface Dataset {
  id: string
  name: string
  accession_id: string | null
  bids_version: string | null
  dataset_type: string
  source_type: string
  remote_url: string | null
  validation_status: string | null
  subject_count: number | null
  participants: Participant[]
}

export interface TextToSQLResult {
  sql: string
  params: Record<string, unknown>
  explanation: string | null
}

export interface QueryResponse {
  message: string
  translation: TextToSQLResult | null
  datasets: Dataset[]
}

export interface CrawlerStatus {
  running: boolean
  current_accession: string | null
  queue: string[]
  last_run_started: string | null
  last_run_finished: string | null
  last_error: string | null
  indexed_count: number
  error_count: number
}

const BASE = '/api'

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('bids_eye_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

function handleUnauthorized() {
  localStorage.removeItem('bids_eye_token')
  window.location.reload()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  })
  if (res.status === 401) { handleUnauthorized(); throw new Error('Unauthorized') }
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: authHeaders() })
  if (res.status === 401) { handleUnauthorized(); throw new Error('Unauthorized') }
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export const api = {
  query: (question: string) =>
    post<QueryResponse>('/query', { question }),

  crawlerStatus: () =>
    get<CrawlerStatus>('/crawler/status'),
}
