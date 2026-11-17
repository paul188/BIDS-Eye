<script setup lang="ts">
import type { QueryFilter } from '../api'

// How the question was understood: one chip per condition of the query that
// ran (backend services/query_filters.py). Display only; to change the
// search, the user edits the question.
defineProps<{ filters: QueryFilter[] }>()

function isTextChip(f: QueryFilter): boolean {
  return f.kind === 'text' || f.kind === 'topic'
}
</script>

<template>
  <div class="flex flex-col gap-2">
    <div class="flex flex-wrap items-center gap-2">
      <span class="text-[13px] text-muted mr-0.5">Understood as</span>
      <ul class="contents">
        <!-- Teal: structured filter; violet: words matched in the dataset text -->
        <li
          v-for="f in filters"
          :key="f.id"
          class="inline-flex items-center rounded-lg px-2.5 py-1.5 text-[13px] leading-none text-ink"
          :class="isTextChip(f) ? 'bg-chip-text-bg' : 'bg-chip-bg'"
          :title="f.detail || undefined"
        >
          <span :class="isTextChip(f) ? 'text-chip-text-ink' : 'text-chip-ink'">{{ f.negated ? `No ${f.field.toLowerCase()}` : f.field }}</span>
          <span class="mx-1 opacity-50" :class="isTextChip(f) ? 'text-chip-text-ink' : 'text-chip-ink'" aria-hidden="true">·</span>
          <span class="font-semibold">{{ f.label }}</span>
          <span
            v-if="!isTextChip(f) && f.text_words.length"
            class="ml-1.5 text-[11px] font-medium text-chip-ink"
          >or in text</span>
        </li>
      </ul>
    </div>
    <p v-if="filters.some(isTextChip)" class="text-xs text-muted-soft">
      Violet chips (Text, Topic) match words in the dataset text.
    </p>
  </div>
</template>
