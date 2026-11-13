<script setup lang="ts">
import { computed } from 'vue'
import type { QueryFilter } from '../api'

// How the question was understood: one chip per condition of the query that
// ran (backend services/query_filters.py). Removing a chip re-runs the search
// without it, with no LLM call and without using a daily search.
const props = defineProps<{
  filters: QueryFilter[]
  removed: string[]
  busy?: boolean
  error?: string | null
}>()

const emit = defineEmits<{
  (e: 'toggle', id: string): void
  (e: 'reset'): void
}>()

const removedSet = computed(() => new Set(props.removed))

function tooltip(f: QueryFilter, isRemoved: boolean): string {
  const action = isRemoved ? 'Removed. Click ↺ to put it back.' : ''
  return [f.detail, action].filter(Boolean).join(' ')
}
</script>

<template>
  <div class="flex flex-col gap-2">
    <div class="flex flex-wrap items-center gap-2" :class="{ 'opacity-60 pointer-events-none': busy }" :aria-busy="busy">
      <span class="text-[13px] text-muted mr-0.5">Understood as</span>

      <span
        v-for="f in filters"
        :key="f.id"
        class="group inline-flex items-center rounded-lg border text-[13px] leading-none transition-colors"
        :class="[
          removedSet.has(f.id)
            ? 'border-border bg-transparent text-muted-soft'
            : 'bg-panel text-ink border-border-input',
          f.kind === 'text' ? 'border-dashed' : '',
        ]"
        :title="tooltip(f, removedSet.has(f.id))"
      >
        <span class="pl-2.5 py-1.5" :class="removedSet.has(f.id) ? 'line-through decoration-1' : ''">
          <span class="text-muted">{{ f.negated ? `No ${f.field.toLowerCase()}` : f.field }}</span>
          <span class="mx-1 text-border-strong" aria-hidden="true">·</span>
          <span class="font-medium">{{ f.label }}</span>
          <span
            v-if="f.kind !== 'text' && f.text_words.length && !removedSet.has(f.id)"
            class="ml-1.5 text-[11px] font-medium text-muted-soft"
          >or in text</span>
        </span>
        <button
          class="ml-0.5 mr-0.5 w-6 h-6 inline-flex items-center justify-center rounded-md text-muted hover:bg-panel-soft hover:text-ink"
          :aria-label="removedSet.has(f.id) ? `Put back ${f.field}: ${f.label}` : `Remove ${f.field}: ${f.label}`"
          :title="removedSet.has(f.id) ? 'Put this filter back' : 'Remove this filter and search again'"
          @click="emit('toggle', f.id)"
        >
          <svg v-if="removedSet.has(f.id)" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v4h4" />
          </svg>
          <svg v-else width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" aria-hidden="true">
            <path d="M6 6l12 12M18 6 6 18" />
          </svg>
        </button>
      </span>

      <button
        v-if="removed.length"
        class="text-[13px] font-medium text-accent hover:underline px-1 py-1"
        @click="emit('reset')"
      >
        Reset filters
      </button>
      <span v-if="busy" class="text-[13px] text-muted" role="status">Searching…</span>
    </div>

    <p v-if="error" class="text-[13px] text-danger-ink">{{ error }}</p>
    <p v-else class="text-xs text-muted-soft">
      Remove a filter to widen the search. This doesn't use one of your daily searches.
      Dashed chips match words in the dataset text.
    </p>
  </div>
</template>
