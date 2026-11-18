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
  authors: string[] | null
  institutions: string[] | null
  description_text: string | null
  /** Set when the dataset was removed from, or flagged deprecated on, its source. */
  deprecated_reason?: string | null
  /** 1 = probably relevant, 2 = possibly relevant: a filter could not be checked. */
  uncertain_level?: number | null
  /** Why, e.g. "ages not recorded". */
  uncertain_reasons?: string | null
  subject_count: number | null
  participants: Participant[]
  /** Counts over all participants; `participants` itself is capped at 50. */
  participant_summary?: {
    total: number
    diagnoses: Record<string, number>
    female: number
    male: number
    age_min: number | null
    age_max: number | null
  } | null
  /** README passage: the part matching the search, else the start. Matches sit between ⟦ and ⟧. */
  snippet?: string | null
  /** 2–3 generated sentences, only where the README is thin; show them labelled as generated. */
  generated_summary?: string | null
  /** The primary paper, when it was checked to report this dataset (or a larger sample including it). */
  linked_paper?: { title: string; year: number | null; url: string | null; relation: string; relation_label: string } | null
}

export interface TextToSQLResult {
  sql: string
  params: Record<string, unknown>
  explanation: string | null
}

/** One condition of the search, as the backend read it from the generated SQL. */
export interface QueryFilter {
  /** Position in the original query ("w0", "h0"). */
  id: string
  /** 'topic': a qualified subject searched as one unit; 'sense': the meanings it
   *  excludes (id = topic id + "n"; removing it removes only the exclusion);
   *  'sort': the query's ORDER BY / LIMIT (id "o0"; removing it shows every match). */
  kind: 'diagnosis' | 'modality' | 'task' | 'age' | 'sex' | 'subjects' | 'count' | 'text' | 'field' | 'topic' | 'sense' | 'sort' | 'other'
  field: string
  label: string
  negated: boolean
  /** Words also searched in the dataset text; a "text" chip is only these. */
  text_words: string[]
  detail: string
}

export interface QueryResponse {
  message: string
  translation: TextToSQLResult | null
  /** How the question was understood; [] when it has no readable conditions. */
  filters?: QueryFilter[]
  datasets: Dataset[]
  query_id: string | null
  total: number
  page: number
  page_size: number
  has_more: boolean
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

// A 429 from the backend's rate limiter (backend/rate_limits.py): which kind
// of request ran out ('search', 'page', ...), in which window ('minute' or
// 'day'), and when it may be tried again.
export class RateLimitError extends Error {
  retryAfter: string
  limit: number
  category: string
  window: string
  constructor(detail: { retry_after: string; limit: number; category?: string; window?: string }) {
    super('rate_limit_exceeded')
    this.retryAfter = detail.retry_after
    this.limit = detail.limit
    this.category = detail.category ?? ''
    this.window = detail.window ?? ''
  }
}

async function throwIfRateLimited(res: Response): Promise<void> {
  if (res.status !== 429) return
  const data = await res.json()
  throw new RateLimitError(data.detail)
}

const BASE = '/api'

// The login is an HttpOnly session cookie set by the backend: page scripts
// cannot read it, the browser sends it with every same-origin /api request.
//
// CSRF: the backend also sets a readable bids_eye_csrf cookie and rejects any
// POST from a signed-in browser that does not repeat it in this header. Other
// sites cannot read our cookies, so they cannot forge the header.
function csrfHeader(): Record<string, string> {
  const cookie = document.cookie
    .split('; ')
    .find(row => row.startsWith('bids_eye_csrf='))
  return cookie ? { 'x-csrftoken': decodeURIComponent(cookie.split('=').slice(1).join('=')) } : {}
}

function handleUnauthorized() {
  window.dispatchEvent(new CustomEvent('bids-unauthorized'))
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', ...csrfHeader() },
    body: JSON.stringify(body),
  })
  if (res.status === 401) { handleUnauthorized(); throw new Error('Unauthorized') }
  await throwIfRateLimited(res)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { credentials: 'same-origin' })
  if (res.status === 401) { handleUnauthorized(); throw new Error('Unauthorized') }
  await throwIfRateLimited(res)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export const api = {
  query: (question: string) =>
    post<QueryResponse>('/query', { question }),

  queryPage: (queryId: string, page: number, pageSize: number) =>
    post<QueryResponse>('/query/page', { query_id: queryId, page, page_size: pageSize }),

  crawlerStatus: () =>
    get<CrawlerStatus>('/crawler/status'),

  // 401 when not signed in: the frontend cannot read the cookie itself.
  me: () =>
    get<{ login: string }>('/auth/me'),

  logout: () =>
    post<{ ok: boolean }>('/auth/logout', {}),
}
