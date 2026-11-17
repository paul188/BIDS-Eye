<script setup lang="ts">
import { ref, computed, nextTick, watch, onMounted } from 'vue'

const CHAR_LIMIT = 2000
const CHAR_WARN_AT = 1500
const MAX_HEIGHT = 240

// The search box: on the start page, and above the latest search's results,
// holding its question. The question in large type, a clear button, Enter
// searches.
const props = withDefaults(defineProps<{
  modelValue: string
  /** ISO time the daily limit resets; shows the countdown under the box. */
  rateLimitedUntil: string | null
  countdown: string
  placeholder?: string
}>(), { placeholder: 'Describe the datasets you need' })

const emit = defineEmits<{
  (e: 'update:modelValue', v: string): void
  (e: 'submit'): void
}>()

const textarea = ref<HTMLTextAreaElement | null>(null)
const charCount = computed(() => props.modelValue.length)
const showCharCount = computed(() => charCount.value >= CHAR_WARN_AT)

function autoResize() {
  const el = textarea.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, MAX_HEIGHT) + 'px'
}

// Clearing or restoring the text from outside must resize the box too.
watch(() => props.modelValue, () => nextTick(autoResize))
// The box above the results mounts with the question already in it.
onMounted(autoResize)

function clear() {
  emit('update:modelValue', '')
  textarea.value?.focus()
}

function onInput(e: Event) {
  emit('update:modelValue', (e.target as HTMLTextAreaElement).value)
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    emit('submit')
  }
}

defineExpose({ focus: (options?: FocusOptions) => textarea.value?.focus(options) })
</script>

<template>
  <div
    class="flex items-start gap-3 bg-panel border border-border-input rounded-xl px-4 sm:px-5 py-3 shadow-sm transition-shadow hover:shadow-md focus-within:border-accent focus-within:ring-4 focus-within:ring-accent/10"
  >
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" class="mt-[5px] flex-shrink-0 text-muted" aria-hidden="true">
      <circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" />
    </svg>
    <textarea
      ref="textarea"
      :value="modelValue"
      rows="1"
      :maxlength="CHAR_LIMIT"
      :placeholder="placeholder"
      aria-label="Search question"
      class="flex-1 bg-transparent text-ink placeholder-muted-soft placeholder:font-normal resize-none focus:outline-none focus-visible:outline-none text-[1.125rem] sm:text-[1.3125rem] font-medium leading-snug"
      @keydown="onKeydown"
      @input="onInput"
    />
    <button
      v-if="modelValue"
      type="button"
      class="flex-shrink-0 w-8 h-8 -my-0.5 inline-flex items-center justify-center rounded-lg text-muted hover:bg-panel-soft hover:text-ink transition-colors"
      title="Clear"
      aria-label="Clear the search box"
      @click="clear"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" aria-hidden="true">
        <path d="M6 6l12 12M18 6 6 18" />
      </svg>
    </button>
    <button
      type="button"
      :disabled="!modelValue.trim()"
      class="flex-shrink-0 -my-0.5 p-2 inline-flex items-center rounded-lg bg-accent text-white hover:bg-accent-hover disabled:bg-panel-soft disabled:text-muted-soft transition-colors"
      title="Search (Enter)"
      aria-label="Search"
      @click="emit('submit')"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" aria-hidden="true">
        <circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" />
      </svg>
    </button>
  </div>
  <p
    v-if="showCharCount"
    class="mt-2 px-1 text-right text-xs tabular-nums"
    :class="charCount >= CHAR_LIMIT ? 'text-danger-ink font-medium' : 'text-muted'"
  >{{ charCount }}&thinsp;/&thinsp;{{ CHAR_LIMIT }}</p>

  <!-- Daily limit reached: the box stays, the notice goes under it. -->
  <div
    v-if="rateLimitedUntil"
    class="mt-3 flex items-center gap-3 bg-panel border border-border rounded-xl px-5 py-4"
  >
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" class="text-muted flex-shrink-0" aria-hidden="true">
      <circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />
    </svg>
    <p class="text-sm text-ink">
      You've used all 20 free searches for today. You can search again in
      <span class="font-mono font-medium text-accent tabular-nums">{{ countdown }}</span>.
    </p>
  </div>
</template>
