<script setup lang="ts">
import type { Message } from '../stores/chat'
import { useChatStore } from '../stores/chat'
import DatasetList from './DatasetList.vue'
import QueryChips from './QueryChips.vue'

const props = defineProps<{ message: Message }>()
const store = useChatStore()

function onPageChange(page: number) {
  store.loadPage(props.message.id, page)
}

function onToggleFilter(chipId: string) {
  store.toggleFilter(props.message.id, chipId)
}

function onResetFilters() {
  store.resetFilters(props.message.id)
}
</script>

<template>
  <!-- The question, shown as the heading of its results -->
  <section v-if="message.role === 'user'" data-role="user" class="scroll-mt-6">
    <p class="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-soft mb-2">Search</p>
    <h2 class="font-serif text-[1.375rem] sm:text-[1.625rem] leading-snug text-ink max-w-3xl break-words">
      {{ message.content }}
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
      <QueryChips
        v-if="message.filters?.length"
        class="mb-4"
        :filters="message.filters"
        :removed="message.removedFilters ?? []"
        :busy="message.refining"
        :error="message.refineError"
        @toggle="onToggleFilter"
        @reset="onResetFilters"
      />
      <p v-if="message.content" class="text-sm text-muted max-w-3xl leading-relaxed">
        {{ message.content }}
      </p>
      <DatasetList
        v-if="message.datasets.length || message.sql || message.removedFilters?.length"
        class="mt-4"
        :datasets="message.datasets"
        :sql="message.sql"
        :total="message.total"
        :page="message.page"
        :page-size="message.pageSize"
        :loading="message.pageLoading || message.refining"
        @page-change="onPageChange"
      />
    </template>
  </section>
</template>
