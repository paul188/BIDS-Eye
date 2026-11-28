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
  /** Counts over all participants; null for a dataset without a participants.tsv. */
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
  /** Why it matched, per filter that can match in several ways
   *  ("Age: 12 of 40 participants aged 4–23 months"); older backends omit it. */
  match_reasons?: string[]
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
   *  'sort': the query's ORDER BY / LIMIT (id "o0"; removing it shows every match).
   *  Searches saved before 2026-11-26 call 'column' 'field'. */
  kind: 'diagnosis' | 'modality' | 'task' | 'age' | 'sex' | 'subjects' | 'count' | 'text' | 'column' | 'topic' | 'sense' | 'sort' | 'other'
  /** What the chip is about ("Diagnosis"). */
  field: string
  label: string
  negated: boolean
  /** Words also searched in the dataset text; a "text" chip is only these. */
  text_words: string[]
  detail: string
  /** Of a label that lists values: all of them, and how many the label lists
   *  (the page shows "+N" for the rest). [] and 0 for any other label. Saved
   *  searches from before 2026-11-26 have no `shown` and "+N" in the label. */
  values?: string[]
  shown?: number
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
  /** The saved-search entry this answer was stored as; null unless the user
   *  turned on "Remember my searches and search results". */
  history_id?: string | null
}

/** One saved search in the list (backend/routers/history.py). */
export interface HistoryEntrySummary {
  id: string
  asked_at: string
  question: string
  total: number
}

export interface HistoryList {
  enabled: boolean
  retention_days: number
  entries: HistoryEntrySummary[]
}

export interface HistoryEntry {
  id: string
  asked_at: string
  question: string
  /** The answer as it was given then; its query_id has usually expired. */
  answer: QueryResponse
}

// A 429 from the backend's rate limiter (backend/security/rate_limits.py): which kind
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

// The backend no longer knows this search (410): it has to be run again.
export class SearchExpiredError extends Error {}

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

// Every call to the backend goes through here. An error carries the backend's
// own sentence (``detail``), which is written for the user. A 204 has no body.
async function request<T>(
  method: 'GET' | 'POST' | 'PUT' | 'DELETE',
  path: string,
  options: { body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    credentials: 'same-origin',
    headers: method === 'GET' ? undefined : { 'Content-Type': 'application/json', ...csrfHeader() },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
  })
  if (res.status === 401) { handleUnauthorized(); throw new Error('Unauthorized') }
  if (!res.ok) {
    const detail = await res.json().then(data => data?.detail, () => null)
    if (res.status === 429 && detail?.retry_after) throw new RateLimitError(detail)
    const message = typeof detail === 'string' ? detail : `${res.status} ${res.statusText}`
    throw res.status === 410 ? new SearchExpiredError(message) : new Error(message)
  }
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const api = {
  query: (question: string) =>
    request<QueryResponse>('POST', '/query', { body: { question } }),

  // `signal` lets the caller abort a page it no longer wants.
  queryPage: (queryId: string, page: number, pageSize: number, signal?: AbortSignal) =>
    request<QueryResponse>('GET', `/query/${encodeURIComponent(queryId)}?page=${page}&page_size=${pageSize}`, { signal }),

  // 401 when not signed in: the frontend cannot read the cookie itself.
  me: () =>
    request<{ login: string }>('GET', '/auth/me'),

  logout: () =>
    request<{ ok: boolean }>('POST', '/auth/logout', { body: {} }),

  // Saved searches on the server, for a user who turned them on.
  history: () =>
    request<HistoryList>('GET', '/history'),

  // Off deletes everything saved for the user.
  setHistoryEnabled: (enabled: boolean) =>
    request<{ enabled: boolean }>('PUT', '/history/enabled', { body: { enabled } }),

  historyEntry: (id: string) =>
    request<HistoryEntry>('GET', `/history/${encodeURIComponent(id)}`),

  deleteHistoryEntry: (id: string) =>
    request<void>('DELETE', `/history/${encodeURIComponent(id)}`),
}
