<script setup lang="ts">
import { computed, ref } from 'vue'
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

// A label that stands for several values ends in "+N". The count is a button
// that opens the full list under the chips, so nothing searched for is hidden.
const expanded = ref<Set<string>>(new Set())

function hiddenCount(f: QueryFilter): string | null {
  if (!f.values?.length) return null
  return f.label.match(/\s(\+\d+)$/)?.[1] ?? null
}

function shortLabel(f: QueryFilter): string {
  return hiddenCount(f) ? f.label.replace(/\s\+\d+$/, '') : f.label
}

function toggleExpanded(id: string) {
  const next = new Set(expanded.value)
  if (!next.delete(id)) next.add(id)
  expanded.value = next
}

const openLists = computed(() =>
  props.filters.filter(f => expanded.value.has(f.id) && hiddenCount(f)),
)

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
          f.kind === 'text' || f.kind === 'topic' ? 'border-dashed' : '',
        ]"
        :title="tooltip(f, removedSet.has(f.id))"
      >
        <span class="pl-2.5 py-1.5" :class="removedSet.has(f.id) ? 'line-through decoration-1' : ''">
          <span class="text-muted">{{ f.negated ? `No ${f.field.toLowerCase()}` : f.field }}</span>
          <span class="mx-1 text-border-strong" aria-hidden="true">·</span>
          <span class="font-medium">{{ shortLabel(f) }}</span>
          <button
            v-if="hiddenCount(f)"
            type="button"
            class="ml-1 -my-0.5 px-1 py-0.5 rounded font-medium text-accent hover:bg-panel-soft hover:underline"
            :aria-expanded="expanded.has(f.id)"
            :aria-controls="`chip-values-${f.id}`"
            :title="expanded.has(f.id) ? 'Hide the full list' : `Show all ${f.values!.length}`"
            @click="toggleExpanded(f.id)"
          >{{ hiddenCount(f) }}</button>
          <span
            v-if="!['text', 'topic'].includes(f.kind) && f.text_words.length && !removedSet.has(f.id)"
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

    <div
      v-for="f in openLists"
      :id="`chip-values-${f.id}`"
      :key="f.id"
      class="rounded-lg border border-border bg-panel px-3 py-2 text-[13px]"
    >
      <div class="flex items-baseline justify-between gap-3">
        <span class="text-muted">
          {{ f.negated ? `No ${f.field.toLowerCase()}` : f.field }} · any of these {{ f.values!.length }}
        </span>
        <button type="button" class="text-accent hover:underline shrink-0" @click="toggleExpanded(f.id)">Hide</button>
      </div>
      <ul class="mt-1.5 flex flex-wrap gap-x-1.5 gap-y-1 max-h-48 overflow-y-auto">
        <li
          v-for="v in f.values"
          :key="v"
          class="rounded-md bg-panel-soft px-1.5 py-0.5 text-ink"
        >{{ v }}</li>
      </ul>
    </div>

    <p v-if="error" class="text-[13px] text-danger-ink">{{ error }}</p>
    <p v-else class="text-xs text-muted-soft">
      Remove a filter to widen the search. This doesn't use one of your daily searches.
      Dashed chips match words in the dataset text.
    </p>
  </div>
</template>
