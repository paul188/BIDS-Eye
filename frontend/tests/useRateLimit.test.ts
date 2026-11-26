// The rate-limit banner's state (src/composables/useRateLimit.ts). Run: npm test.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { effectScope } from 'vue'
import { RateLimitError } from '../src/api/index.ts'
import { formatCountdown, useRateLimit } from '../src/composables/useRateLimit.ts'

const NOW = Date.parse('2026-11-26T12:00:00Z')

function limitedUntil(iso: string) {
  return new RateLimitError({ retry_after: iso, limit: 20, category: 'search', window: 'day' })
}

test('formatCountdown gives hours, minutes and seconds', () => {
  assert.equal(formatCountdown('2026-11-26T13:02:03Z', NOW), '01:02:03')
  assert.equal(formatCountdown('2026-11-26T12:00:59Z', NOW), '00:00:59')
})

test('formatCountdown stops at zero', () => {
  assert.equal(formatCountdown('2026-11-26T11:00:00Z', NOW), '00:00:00')
})

test('a 429 sets the banner and the countdown runs until the wait is over', t => {
  t.mock.timers.enable({ apis: ['setInterval', 'Date'], now: NOW })
  const scope = effectScope()
  const limit = scope.run(() => useRateLimit())!

  limit.startRateLimit(limitedUntil('2026-11-26T12:00:03Z'))
  assert.equal(limit.rateLimitedUntil.value, '2026-11-26T12:00:03Z')
  assert.equal(limit.rateLimitWindow.value, 'day')
  assert.equal(limit.rateLimitCount.value, 20)
  assert.equal(limit.countdownDisplay.value, '00:00:03')

  t.mock.timers.tick(1_000)
  assert.equal(limit.countdownDisplay.value, '00:00:02')

  t.mock.timers.tick(2_000)
  assert.equal(limit.rateLimitedUntil.value, null)
  scope.stop()
})

test('a second 429 replaces the first', t => {
  t.mock.timers.enable({ apis: ['setInterval', 'Date'], now: NOW })
  const scope = effectScope()
  const limit = scope.run(() => useRateLimit())!

  limit.startRateLimit(limitedUntil('2026-11-26T12:00:02Z'))
  limit.startRateLimit(limitedUntil('2026-11-26T12:01:00Z'))
  t.mock.timers.tick(3_000)
  assert.equal(limit.rateLimitedUntil.value, '2026-11-26T12:01:00Z')
  assert.equal(limit.countdownDisplay.value, '00:00:57')
  scope.stop()
})
