<script setup lang="ts">
import { ref, computed, nextTick, watch } from 'vue'

const CHAR_LIMIT = 2000
const CHAR_WARN_AT = 1500

const props = withDefaults(defineProps<{
  modelValue: string
  /** ISO time the daily limit resets; shows the countdown instead of the input. */
  rateLimitedUntil: string | null
  countdown: string
  /** Larger box for the start page. */
  large?: boolean
  placeholder?: string
}>(), { large: false, placeholder: 'Describe the datasets you need' })

const emit = defineEmits<{
  (e: 'update:modelValue', v: string): void
  (e: 'submit'): void
}>()

const textarea = ref<HTMLTextAreaElement | null>(null)
const charCount = computed(() => props.modelValue.length)
const showCharCount = computed(() => charCount.value >= CHAR_WARN_AT)
const maxHeight = computed(() => (props.large ? 240 : 160))

function autoResize() {
  const el = textarea.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, maxHeight.value) + 'px'
}

// Clearing or restoring the text from outside must resize the box too.
watch(() => props.modelValue, () => nextTick(autoResize))

function onInput(e: Event) {
  emit('update:modelValue', (e.target as HTMLTextAreaElement).value)
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    emit('submit')
  }
}

defineExpose({ focus: () => textarea.value?.focus() })
</script>

<template>
  <!-- Rate-limit notice -->
  <div
    v-if="rateLimitedUntil"
    class="flex items-center gap-3 bg-panel border border-border rounded-xl px-5 py-4"
  >
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" class="text-muted flex-shrink-0" aria-hidden="true">
      <circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />
    </svg>
    <p class="text-sm text-ink">
      You've used all 20 free searches for today. You can search again in
      <span class="font-mono font-medium text-accent tabular-nums">{{ countdown }}</span>.
    </p>
  </div>

  <template v-else>
    <div
      class="flex items-end gap-2 bg-panel border border-border-input rounded-xl transition-shadow focus-within:border-accent focus-within:ring-4 focus-within:ring-accent/10"
      :class="large ? 'px-5 py-4 shadow-sm' : 'px-4 py-2.5 shadow-[0_1px_0_rgba(0,0,0,0.02),0_8px_24px_-12px_rgba(29,33,40,0.18)]'"
    >
      <textarea
        ref="textarea"
        :value="modelValue"
        rows="1"
        :maxlength="CHAR_LIMIT"
        :placeholder="placeholder"
        aria-label="Search question"
        class="flex-1 bg-transparent text-ink placeholder-muted-soft resize-none focus:outline-none focus-visible:outline-none leading-relaxed"
        :class="large ? 'text-base py-1' : 'text-[15px] py-1'"
        @keydown="onKeydown"
        @input="onInput"
      />
      <button
        :disabled="!modelValue.trim()"
        class="flex-shrink-0 inline-flex items-center gap-1.5 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-hover disabled:bg-panel-soft disabled:text-muted-soft transition-colors"
        :class="large ? 'p-2.5 sm:px-4 sm:py-2' : 'p-2'"
        title="Search (Enter)"
        @click="emit('submit')"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true">
          <circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" />
        </svg>
        <span v-if="large" class="hidden sm:inline">Search</span>
      </button>
    </div>
    <p
      v-if="showCharCount"
      class="mt-2 px-1 text-right text-xs tabular-nums"
      :class="charCount >= CHAR_LIMIT ? 'text-danger-ink font-medium' : 'text-muted'"
    >{{ charCount }}&thinsp;/&thinsp;{{ CHAR_LIMIT }}</p>
  </template>
</template>
