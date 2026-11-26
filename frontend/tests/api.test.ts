// How every backend call is sent and how its errors are read
// (request() in src/api/index.ts). Run: npm test.
import assert from 'node:assert/strict'
import { beforeEach, test } from 'node:test'
import { api, RateLimitError, SearchExpiredError } from '../src/api/index.ts'

let calls: { url: string; init: RequestInit }[] = []
let unauthorized = 0

function answerWith(status: number, body?: unknown, statusText = '') {
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    calls.push({ url, init })
    return new Response(body === undefined ? null : JSON.stringify(body), { status, statusText })
  }) as typeof fetch
}

beforeEach(() => {
  calls = []
  unauthorized = 0
  Object.assign(globalThis, {
    document: { cookie: 'other=1; bids_eye_csrf=abc%3D' },
    window: {
      dispatchEvent: () => {
        unauthorized++
        return true
      },
    },
  })
})

test('a POST sends the body as JSON with the CSRF header', async () => {
  answerWith(200, { message: 'ok' })
  await api.query('adhd fmri')
  const { url, init } = calls[0]
  assert.equal(url, '/api/query')
  assert.equal(init.method, 'POST')
  assert.equal(init.body, '{"question":"adhd fmri"}')
  assert.deepEqual(init.headers, { 'Content-Type': 'application/json', 'x-csrftoken': 'abc=' })
})

test('a GET sends no body and no CSRF header', async () => {
  answerWith(200, { enabled: false, entries: [] })
  await api.history()
  const { url, init } = calls[0]
  assert.equal(url, '/api/history')
  assert.equal(init.method, 'GET')
  assert.equal(init.body, undefined)
  assert.equal(init.headers, undefined)
})

test('the abort signal is passed on', async () => {
  answerWith(200, { datasets: [] })
  const controller = new AbortController()
  await api.queryPage('q1', 2, 20, controller.signal)
  assert.equal(calls[0].init.signal, controller.signal)
})

test('401 reports the sign-out, whatever the method', async () => {
  answerWith(401, { detail: 'Not signed in' })
  await assert.rejects(api.me(), /Unauthorized/)
  await assert.rejects(api.logout(), /Unauthorized/)
  await assert.rejects(api.deleteHistoryEntry('h1'), /Unauthorized/)
  assert.equal(unauthorized, 3)
})

test('429 becomes a RateLimitError, whatever the method', async () => {
  const detail = { retry_after: '2026-11-27T00:00:00Z', limit: 20, category: 'search', window: 'day' }
  answerWith(429, { detail })
  for (const call of [() => api.query('q'), () => api.history(), () => api.setHistoryEnabled(true)]) {
    await assert.rejects(call(), (err: unknown) => {
      assert.ok(err instanceof RateLimitError)
      assert.equal(err.retryAfter, detail.retry_after)
      assert.equal(err.window, 'day')
      return true
    })
  }
})

test('410 becomes a SearchExpiredError with the backend sentence', async () => {
  answerWith(410, { detail: 'This search has expired. Please run the query again.' })
  await assert.rejects(api.queryPage('q1', 2, 20), (err: unknown) => {
    assert.ok(err instanceof SearchExpiredError)
    assert.equal(err.message, 'This search has expired. Please run the query again.')
    return true
  })
})

test('another error carries the backend sentence', async () => {
  answerWith(500, { detail: 'The request could not be completed.' })
  await assert.rejects(api.query('q'), /^Error: The request could not be completed\.$/)
})

test('an error without a readable body carries the status', async () => {
  globalThis.fetch = (async () => new Response('<html>', { status: 502, statusText: 'Bad Gateway' })) as typeof fetch
  await assert.rejects(api.history(), /^Error: 502 Bad Gateway$/)
})

test('204 resolves without a body', async () => {
  answerWith(204)
  assert.equal(await api.deleteHistoryEntry('h1'), undefined)
})
