<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Message } from '../stores/chat'
import { useChatStore } from '../stores/chat'
import DatasetList from './DatasetList.vue'
import QueryChips from './QueryChips.vue'

const props = defineProps<{ message: Message }>()
const emit = defineEmits<{ (e: 'edit', question: string): void }>()
const store = useChatStore()
const showSql = ref(false)

// Searches saved before the chips became display-only may have had filters
// removed; their chips are hidden, as their results ran without them.
const shownFilters = computed(() => {
  const removed = new Set(props.message.removedFilters ?? [])
  return (props.message.filters ?? []).filter(f => !removed.has(f.id))
})

function onPageChange(page: number) {
  store.loadPage(props.message.id, page)
}
</script>

<template>
  <!-- The question, in a search box above its results, as search engines
       keep the query in the box. The latest search passes the live search
       field in the "query" slot; an earlier one shows its question, and
       clicking it copies the text into that field. -->
  <section v-if="message.role === 'user'" data-role="user" class="scroll-mt-6">
    <div v-if="$slots.query" class="relative max-w-4xl">
      <h2 class="sr-only">Search: {{ message.content }}</h2>
      <slot name="query" />
    </div>
    <h2 v-else class="max-w-4xl">
      <button
        type="button"
        class="group w-full flex items-start gap-3 text-left rounded-xl border border-border-input bg-panel px-4 sm:px-5 py-3 shadow-sm hover:border-ink hover:shadow-md transition"
        :aria-label="`Edit search: ${message.content}`"
        title="Edit this search"
        @click="emit('edit', message.content)"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" class="mt-[5px] flex-shrink-0 text-muted" aria-hidden="true">
          <circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" />
        </svg>
        <span class="flex-1 text-[1.125rem] sm:text-[1.3125rem] font-medium leading-snug text-ink break-words">{{ message.content }}</span>
        <span class="mt-[3px] flex-shrink-0 inline-flex items-center gap-1 text-[13px] text-muted opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100 transition-opacity" aria-hidden="true">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
          </svg>
          Edit
        </span>
      </button>
    </h2>
  </section>

  <!-- Answer -->
  <section v-else data-role="assistant" class="mt-4">
    <!-- Loading -->
    <div v-if="message.loading" class="flex items-center gap-3 text-sm text-muted py-6" role="status">
      <span class="inline-flex gap-1" aria-hidden="true">
        <span class="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
        <span class="w-1.5 h-1.5 rounded-full bg-accent animate-pulse [animation-delay:150ms]" />
        <span class="w-1.5 h-1.5 rounded-full bg-accent animate-pulse [animation-delay:300ms]" />
      </span>
      Searching datasets…
    </div>

    <!-- Error -->
    <div
      v-else-if="message.error"
      class="bg-danger-bg border border-danger-ink/20 rounded-xl px-4 py-3 text-sm text-danger-ink"
    >
      {{ message.content }}
    </div>

    <!-- Results -->
    <template v-else>
      <QueryChips v-if="shownFilters.length" class="mb-4" :filters="shownFilters" />
      <!-- Result count, and the SQL that ran (for transparency), under the search box -->
      <div v-if="message.content || message.sql" class="flex flex-wrap items-baseline gap-x-4 gap-y-1 max-w-4xl">
        <p v-if="message.content" class="text-sm text-muted leading-relaxed">{{ message.content }}</p>
        <button
          v-if="message.sql"
          class="text-xs text-muted-soft hover:text-ink hover:underline transition-colors"
          :aria-expanded="showSql"
          @click="showSql = !showSql"
        >
          {{ showSql ? '▾ Hide SQL' : '▸ Show generated SQL' }}
        </button>
      </div>
      <pre
        v-if="showSql && message.sql"
        class="mt-2 text-xs bg-panel-soft border border-border rounded-lg p-3 overflow-x-auto text-ink font-mono whitespace-pre-wrap"
      >{{ message.sql }}</pre>
      <DatasetList
        v-if="message.datasets.length || message.sql || message.removedFilters?.length"
        class="mt-4"
        :datasets="message.datasets"
        :total="message.total"
        :page="message.page"
        :page-size="message.pageSize"
        :loading="message.pageLoading"
        :filters="message.filters"
        :removed-filters="message.removedFilters"
        @page-change="onPageChange"
      />
    </template>
  </section>
</template>
