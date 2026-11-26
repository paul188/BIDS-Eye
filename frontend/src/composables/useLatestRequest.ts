// One request at a time. Starting the next aborts the one before, and an
// answer or error that arrives after a newer start is dropped, so a slow
// response cannot overwrite a newer one.

export type LatestResult<T> = { superseded: false; value: T } | { superseded: true }

export function useLatestRequest() {
  let seq = 0
  let controller: AbortController | null = null

  // Drops the request in flight without starting another.
  function cancel() {
    seq++
    controller?.abort()
    controller = null
  }

  async function run<T>(fetcher: (signal: AbortSignal) => Promise<T>): Promise<LatestResult<T>> {
    cancel()
    const mine = seq
    controller = new AbortController()
    try {
      const value = await fetcher(controller.signal)
      return mine === seq ? { superseded: false, value } : { superseded: true }
    } catch (err) {
      // The abort itself rejects the fetch: not a failure.
      if (mine !== seq) return { superseded: true }
      throw err
    }
  }

  return { run, cancel }
}
