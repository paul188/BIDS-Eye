<script setup lang="ts">
import { ref } from 'vue'
import type { Dataset } from '../api'
import DatasetCard from './DatasetCard.vue'

const props = defineProps<{ datasets: Dataset[]; sql: string | null }>()
const showSql = ref(false)
</script>

<template>
  <div class="flex flex-col gap-3">
    <!-- Dataset grid -->
    <div
      v-if="datasets.length"
      class="grid grid-cols-1 sm:grid-cols-2 gap-3"
    >
      <DatasetCard v-for="ds in datasets" :key="ds.id" :dataset="ds" />
    </div>
    <p v-else class="text-muted text-sm italic">No datasets found.</p>

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
        class="mt-2 text-xs bg-surface border border-border rounded-lg p-3 overflow-x-auto text-green-300 font-mono whitespace-pre-wrap"
      >{{ sql }}</pre>
    </div>
  </div>
</template>
