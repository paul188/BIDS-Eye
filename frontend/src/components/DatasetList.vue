<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Dataset } from '../api'
import DatasetCard from './DatasetCard.vue'

const props = defineProps<{
  datasets: Dataset[]
  sql: string | null
  total?: number
  page?: number
  pageSize?: number
  loading?: boolean
}>()

const emit = defineEmits<{ (e: 'page-change', page: number): void }>()

const showSql = ref(false)

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
  <div class="flex flex-col gap-3">
    <!-- Dataset list (single column — one result after the next) -->
    <div
      v-if="datasets.length"
      class="flex flex-col gap-3"
      :class="{ 'opacity-50 pointer-events-none': loading }"
    >
      <DatasetCard v-for="ds in datasets" :key="ds.id" :dataset="ds" />
    </div>
    <p v-else class="text-muted text-sm italic">No datasets found.</p>

    <!-- Pagination controls -->
    <div
      v-if="showPagination"
      class="flex items-center justify-between gap-3 text-xs text-muted"
    >
      <span>
        Page {{ currentPage }} of {{ pageCount }} · {{ total }} datasets
      </span>
      <div class="flex items-center gap-2">
        <button
          class="px-3 py-1 rounded-full border border-border hover:text-accent hover:border-accent disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          :disabled="loading || currentPage <= 1"
          @click="go(currentPage - 1)"
        >
          ‹ Prev
        </button>
        <button
          class="px-3 py-1 rounded-full border border-border hover:text-accent hover:border-accent disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          :disabled="loading || currentPage >= pageCount"
          @click="go(currentPage + 1)"
        >
          Next ›
        </button>
      </div>
    </div>

    <!-- SQL toggle (for development / transparency) -->
    <div v-if="sql" class="mt-1">
      <button
        class="text-xs text-muted hover:text-accent transition-colors"
        @click="showSql = !showSql"
      >
        {{ showSql ? '▾ Hide SQL' : '▸ Show generated SQL' }}
      </button>
      <pre
        v-if="showSql"
        class="mt-2 text-xs bg-panel-soft border border-border rounded-lg p-3 overflow-x-auto text-[#0b5394] font-mono whitespace-pre-wrap"
      >{{ sql }}</pre>
    </div>
  </div>
</template>
