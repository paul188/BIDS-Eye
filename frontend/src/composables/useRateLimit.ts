import { onScopeDispose, ref } from 'vue'
import type { RateLimitError } from '../api/index.ts'

// Time left until `until`, as HH:MM:SS; 00:00:00 once it has passed.
export function formatCountdown(until: string, now = Date.now()): string {
  const ms = new Date(until).getTime() - now
  if (ms <= 0) return '00:00:00'
  const h = Math.floor(ms / 3_600_000)
  const m = Math.floor((ms % 3_600_000) / 60_000)
  const s = Math.floor((ms % 60_000) / 1_000)
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

// The state behind the rate-limit banner: set from a 429, counted down each
// second, cleared when the wait is over.
export function useRateLimit() {
  const rateLimitedUntil = ref<string | null>(null)
  // Which limit was hit: 20 searches a day, or too many searches in one minute.
  const rateLimitWindow = ref('')
  const rateLimitCount = ref(0)
  const countdownDisplay = ref('00:00:00')
  let interval: ReturnType<typeof setInterval> | undefined

  function tick() {
    if (!rateLimitedUntil.value) return clearInterval(interval)
    if (new Date(rateLimitedUntil.value).getTime() <= Date.now()) {
      rateLimitedUntil.value = null
      clearInterval(interval)
    } else {
      countdownDisplay.value = formatCountdown(rateLimitedUntil.value)
    }
  }

  function startRateLimit(err: RateLimitError) {
    rateLimitedUntil.value = err.retryAfter
    rateLimitWindow.value = err.window
    rateLimitCount.value = err.limit
    countdownDisplay.value = formatCountdown(err.retryAfter)
    clearInterval(interval)
    interval = setInterval(tick, 1_000)
  }

  onScopeDispose(() => clearInterval(interval), true)

  return { rateLimitedUntil, rateLimitWindow, rateLimitCount, countdownDisplay, startRateLimit }
}
