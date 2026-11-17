<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Dataset, QueryFilter } from '../api'
import DatasetCard from './DatasetCard.vue'
import { highlightTerms as highlightTermsOf } from '../highlight'

const props = defineProps<{
  datasets: Dataset[]
  sql: string | null
  total?: number
  page?: number
  pageSize?: number
  loading?: boolean
  /** The search's filter chips and which were removed: drive the highlighting. */
  filters?: QueryFilter[]
  removedFilters?: string[]
}>()

const emit = defineEmits<{ (e: 'page-change', page: number): void }>()

const showSql = ref(false)

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

    <!-- Pagination controls -->
    <div
      v-if="showPagination"
      class="flex items-center justify-between gap-3 text-sm text-muted mt-5"
    >
      <span class="tabular-nums">
        Page {{ currentPage }} of {{ pageCount }} · {{ total }} datasets
      </span>
      <div class="flex items-center gap-2">
        <button
          class="px-3 py-1.5 rounded-lg border border-border bg-panel hover:text-accent hover:border-accent/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          :disabled="loading || currentPage <= 1"
          @click="go(currentPage - 1)"
        >
          ← Previous
        </button>
        <button
          class="px-3 py-1.5 rounded-lg border border-border bg-panel hover:text-accent hover:border-accent/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          :disabled="loading || currentPage >= pageCount"
          @click="go(currentPage + 1)"
        >
          Next →
        </button>
      </div>
    </div>

    <!-- SQL toggle (for development / transparency) -->
    <div v-if="sql" class="mt-4">
      <button
        class="text-xs text-muted-soft hover:text-accent transition-colors"
        @click="showSql = !showSql"
      >
        {{ showSql ? '▾ Hide SQL' : '▸ Show generated SQL' }}
      </button>
      <pre
        v-if="showSql"
        class="mt-2 text-xs bg-panel-soft border border-border rounded-lg p-3 overflow-x-auto text-ink font-mono whitespace-pre-wrap"
      >{{ sql }}</pre>
    </div>
  </div>
</template>
