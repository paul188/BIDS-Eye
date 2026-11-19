<script setup lang="ts">
import { ref } from 'vue'

// The site sets only cookies it needs to work (sign-in and CSRF protection),
// so the banner informs and has one button: there is no cookie to opt out of.
// It also says what happens to a search, since that is the data people care
// about: questions are stored on the server without who asked (query_log,
// backend/models.py; tests/test_query_logging.py checks no row records the
// asker) and sent to Google's Gemini API on a paid key, whose prompts Google
// does not train on. Keep this text in line with the backend. If a tracking or third-party
// cookie is ever added, this must become a real consent choice that is asked
// before that cookie is set.
const STORAGE_KEY = 'bids-eye-cookie-notice'

function wasDismissed(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    return false  // storage blocked: show the notice each visit
  }
}

const visible = ref(!wasDismissed())
const showDetails = ref(false)

function dismiss() {
  visible.value = false
  try {
    localStorage.setItem(STORAGE_KEY, '1')
  } catch { /* storage blocked: hidden for this visit only */ }
}
</script>

<template>
  <transition name="cookie">
    <section
      v-if="visible"
      role="region"
      aria-label="Cookies and your searches"
      class="fixed z-50 bottom-4 inset-x-4 sm:inset-x-auto sm:right-6 sm:bottom-6 sm:max-w-md max-h-[calc(100%-2rem)] overflow-y-auto bg-panel border border-border rounded-xl shadow-lg p-4 text-sm text-ink"
    >
      <p class="leading-relaxed">
        BIDS-Eye uses only the cookies it needs to work: no tracking or advertising.
        Your searches are sent to Google's Gemini AI to interpret them, and stored
        anonymously, without your name, so we can improve search quality.
      </p>

      <div v-if="showDetails" class="mt-3 space-y-3 text-xs text-muted leading-relaxed">
        <div>
          <p class="font-medium text-ink">Cookies and browser storage</p>
          <ul class="mt-1 space-y-1">
            <li>
              <span class="font-mono text-ink">bids_eye_session</span> — your sign-in,
              set when you sign in with GitHub, expires after 7 days or when you sign out.
            </li>
            <li>
              <span class="font-mono text-ink">bids_eye_csrf</span> — a random token that
              lets the server reject requests made from other websites.
            </li>
            <li>
              <span class="font-mono text-ink">bids-eye-history</span> (local storage) — the
              search history shown in this browser, until you delete it there.
            </li>
          </ul>
        </div>
        <div>
          <p class="font-medium text-ink">Your searches</p>
          <ul class="mt-1 space-y-1">
            <li>
              Each question is sent to Google's Gemini API, which turns it into a
              database query and helps rank the results. Google does not use it to
              train its models.
            </li>
            <li>
              Each question is stored on our server without your GitHub username or
              anything else that says who asked, together with how it was understood
              and how many datasets it found. We use this to find and fix bad results.
              Please don't type personal information into the search.
            </li>
            <li>
              At sign-in we store your GitHub username and email address for your
              account, and count your searches per day for the daily limit.
            </li>
          </ul>
        </div>
      </div>

      <div class="mt-3 flex items-center justify-end gap-2">
        <button
          class="px-3 py-1.5 rounded-lg text-muted hover:bg-panel-soft hover:text-ink transition-colors"
          :aria-expanded="showDetails"
          @click="showDetails = !showDetails"
        >
          {{ showDetails ? 'Hide details' : 'Details' }}
        </button>
        <button
          class="px-4 py-1.5 rounded-lg bg-ink hover:bg-ink/90 text-white font-medium transition-colors"
          @click="dismiss"
        >
          OK
        </button>
      </div>
    </section>
  </transition>
</template>

<style scoped>
.cookie-enter-active, .cookie-leave-active { transition: opacity 0.2s, transform 0.2s; }
.cookie-enter-from, .cookie-leave-to { opacity: 0; transform: translateY(8px); }
</style>
