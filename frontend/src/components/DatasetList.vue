<script setup lang="ts">
import { computed } from 'vue'
import type { Dataset, QueryFilter } from '../api'
import DatasetCard from './DatasetCard.vue'
import { highlightTerms as highlightTermsOf } from '../highlight'

const props = defineProps<{
  datasets: Dataset[]
  total?: number
  page?: number
  pageSize?: number
  loading?: boolean
  /** The search's filter chips and which were removed: drive the highlighting. */
  filters?: QueryFilter[]
  removedFilters?: string[]
}>()

const emit = defineEmits<{ (e: 'page-change', page: number): void }>()

const activeFilters = computed(() => {
  const removed = new Set(props.removedFilters ?? [])
  return (props.filters ?? []).filter(f => !removed.has(f.id) && !f.negated)
})

const highlightTerms = computed(() => highlightTermsOf(activeFilters.value))

const emphasisKinds = computed(() => activeFilters.value.map(f => f.kind))

const pageCount = computed(() => {
  const size = props.pageSize ?? 0
  if (!size || !props.total) return 1
  return Math.max(1, Math.ceil(props.total / size))
})
const currentPage = computed(() => props.page ?? 1)
const showPagination = computed(() => (props.total ?? 0) > 0 && pageCount.value > 1)

function go(page: number) {
  if (props.loading) return
  if (page < 1 || page > pageCount.value || page === currentPage.value) return
  emit('page-change', page)
}
</script>

<template>
  <div class="flex flex-col">
    <!-- Results: one row after the next, separated by rules -->
    <div
      v-if="datasets.length"
      class="flex flex-col transition-opacity"
      :class="{ 'opacity-50 pointer-events-none': loading }"
    >
      <template v-for="(ds, i) in datasets" :key="ds.id">
        <!-- Banner where the uncertain tier starts (services/uncertain_tier.py) -->
        <div
          v-if="ds.uncertain_level && (i === 0 || !datasets[i - 1].uncertain_level)"
          class="my-3 px-4 py-2.5 rounded-lg bg-warn-bg text-warn-ink text-[13px] leading-relaxed"
        >
          <span class="font-semibold">Uncertain results.</span>
          These datasets match only part of your search, or don't record something
          it filters on (for example participant ages). Each card says what could
          not be checked.
        </div>
        <DatasetCard :dataset="ds" :terms="highlightTerms" :emphasis="emphasisKinds" />
      </template>
    </div>
    <p v-else class="text-muted text-sm py-4">No datasets found.</p>

    <!-- Pagination: centred under the results column (as wide as the search box) -->
    <div
      v-if="showPagination"
      class="max-w-4xl w-full flex items-center justify-center gap-3 sm:gap-4 text-sm text-muted mt-8"
    >
      <button
        class="px-3 py-1.5 rounded-lg border border-border bg-panel hover:text-accent hover:border-accent/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        :disabled="loading || currentPage <= 1"
        @click="go(currentPage - 1)"
      >
        ← Previous
      </button>
      <span class="tabular-nums text-center">
        Page {{ currentPage }} of {{ pageCount }}<span class="hidden sm:inline"> · {{ total }} datasets</span>
      </span>
      <button
        class="px-3 py-1.5 rounded-lg border border-border bg-panel hover:text-accent hover:border-accent/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        :disabled="loading || currentPage >= pageCount"
        @click="go(currentPage + 1)"
      >
        Next →
      </button>
    </div>

  </div>
</template>
