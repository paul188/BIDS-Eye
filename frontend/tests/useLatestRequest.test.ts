// One request at a time (src/composables/useLatestRequest.ts). Run: npm test.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { useLatestRequest } from '../src/composables/useLatestRequest.ts'

// A request the test answers by hand; rejects when aborted, as fetch does.
function pending<T>() {
  let resolve!: (value: T) => void
  let reject!: (err: Error) => void
  let signal!: AbortSignal
  const fetcher = (s: AbortSignal) => {
    signal = s
    return new Promise<T>((res, rej) => {
      resolve = res
      reject = rej
      s.addEventListener('abort', () => rej(new DOMException('aborted', 'AbortError')))
    })
  }
  return { fetcher, resolve: (v: T) => resolve(v), reject: (e: Error) => reject(e), aborted: () => signal.aborted }
}

test('a single request returns its value', async () => {
  const latest = useLatestRequest()
  assert.deepEqual(await latest.run(async () => 'page 2'), { superseded: false, value: 'page 2' })
})

test('starting the next request aborts the one before', async () => {
  const latest = useLatestRequest()
  const first = pending<string>()
  const second = pending<string>()
  const firstResult = latest.run(first.fetcher)
  const secondResult = latest.run(second.fetcher)
  assert.equal(first.aborted(), true)
  assert.equal(second.aborted(), false)
  second.resolve('page 3')
  assert.deepEqual(await firstResult, { superseded: true })
  assert.deepEqual(await secondResult, { superseded: false, value: 'page 3' })
})

test('an answer that ignores the abort and arrives late is dropped', async () => {
  const latest = useLatestRequest()
  let answerFirst!: (value: string) => void
  const firstResult = latest.run(
    () =>
      new Promise<string>(res => {
        answerFirst = res
      }),
  )
  const secondResult = latest.run(async () => 'page 3')
  assert.deepEqual(await secondResult, { superseded: false, value: 'page 3' })
  answerFirst('page 2')
  assert.deepEqual(await firstResult, { superseded: true })
})

test('the failure of the current request is thrown', async () => {
  const latest = useLatestRequest()
  await assert.rejects(
    latest.run(async () => {
      throw new Error('500 Internal Server Error')
    }),
    /500/,
  )
})

test('the failure of a superseded request is not', async () => {
  const latest = useLatestRequest()
  const first = pending<string>()
  const firstResult = latest.run(first.fetcher)
  const secondResult = latest.run(async () => 'page 3')
  first.reject(new Error('network down'))
  assert.deepEqual(await firstResult, { superseded: true })
  assert.deepEqual(await secondResult, { superseded: false, value: 'page 3' })
})

test('cancel drops the request in flight', async () => {
  const latest = useLatestRequest()
  const first = pending<string>()
  const firstResult = latest.run(first.fetcher)
  latest.cancel()
  assert.equal(first.aborted(), true)
  assert.deepEqual(await firstResult, { superseded: true })
})
