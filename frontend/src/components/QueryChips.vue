<script setup lang="ts">
import { computed, ref } from 'vue'
import type { QueryFilter } from '../api'

// How the question was understood: one chip per condition of the query that
// ran (backend services/results/display/chips.py). Display only; to change the
// search, the user edits the question.
const props = defineProps<{ filters: QueryFilter[] }>()

function isTextChip(f: QueryFilter): boolean {
  return f.kind === 'text' || f.kind === 'topic'
}

// What a chip searched for beyond its label: the values behind "+N" and,
// for a structured chip, the words also matched in the dataset text. Either
// opens the full list under the chips, so nothing searched for is hidden.
function hiddenCount(f: QueryFilter): string | null {
  const hidden = (f.values?.length ?? 0) - (f.shown ?? f.values?.length ?? 0)
  return hidden > 0 ? `+${hidden}` : null
}
function alsoText(f: QueryFilter): boolean {
  return !isTextChip(f) && f.text_words.length > 0
}

// The chip's `detail` explains it; the lists the chip carries are added here,
// so the backend writes them only once.
function listed(items: string[], max: number): string {
  return items.slice(0, max).join(', ') + (items.length > max ? '…' : '')
}
function tooltip(f: QueryFilter): string | undefined {
  const parts = [f.detail]
  if (hiddenCount(f)) parts.push(`Any of these ${f.values!.length}: ${listed(f.values!, 8)}.`)
  if (f.text_words.length) {
    parts.push(`${isTextChip(f) ? 'Searched as' : 'Or the dataset text mentions'}: ${listed(f.text_words, 8)}.`)
  }
  return parts.filter(Boolean).join(' ') || undefined
}

const open = ref<Set<string>>(new Set())
function toggle(id: string) {
  const next = new Set(open.value)
  if (!next.delete(id)) next.add(id)
  open.value = next
}
const openLists = computed(() =>
  props.filters.filter(f => open.value.has(f.id) && (hiddenCount(f) || alsoText(f))))
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
          :title="tooltip(f)"
        >
          <span :class="isTextChip(f) ? 'text-chip-text-ink' : 'text-chip-ink'">{{ f.negated ? `No ${f.field.toLowerCase()}` : f.field }}</span>
          <span class="mx-1 opacity-50" :class="isTextChip(f) ? 'text-chip-text-ink' : 'text-chip-ink'" aria-hidden="true">·</span>
          <span class="font-semibold">{{ f.label }}</span>
          <button
            v-if="hiddenCount(f) || alsoText(f)"
            type="button"
            class="ml-1.5 -my-1 px-1 py-1 rounded font-medium underline decoration-dotted underline-offset-2 hover:decoration-solid"
            :class="isTextChip(f) ? 'text-chip-text-ink' : 'text-chip-ink'"
            :aria-expanded="open.has(f.id)"
            :aria-controls="`chip-values-${f.id}`"
            @click="toggle(f.id)"
          >{{ [hiddenCount(f), alsoText(f) ? 'or in text' : null].filter(Boolean).join(' · ') }}</button>
        </li>
      </ul>
    </div>

    <div
      v-for="f in openLists"
      :id="`chip-values-${f.id}`"
      :key="f.id"
      class="rounded-lg border border-border bg-panel px-3 py-2 text-[13px] max-w-4xl"
    >
      <div class="flex items-baseline justify-between gap-3">
        <span class="font-medium text-ink">{{ f.negated ? `No ${f.field.toLowerCase()}` : f.field }}: {{ f.label }}</span>
        <button type="button" class="text-muted hover:text-ink shrink-0" @click="toggle(f.id)">Hide</button>
      </div>
      <template v-if="hiddenCount(f)">
        <p class="mt-1.5 text-muted">Any of these {{ f.values!.length }}:</p>
        <ul class="mt-1 flex flex-wrap gap-1 max-h-40 overflow-y-auto">
          <li v-for="v in f.values" :key="v" class="rounded-md bg-panel-soft px-1.5 py-0.5 text-ink">{{ v }}</li>
        </ul>
      </template>
      <template v-if="alsoText(f)">
        <p class="mt-1.5 text-muted">{{ hiddenCount(f) ? 'Or' : 'Also' }} datasets whose text mentions:</p>
        <ul class="mt-1 flex flex-wrap gap-1 max-h-40 overflow-y-auto">
          <li v-for="w in f.text_words" :key="w" class="rounded-md bg-chip-text-bg px-1.5 py-0.5 text-ink">{{ w }}</li>
        </ul>
      </template>
    </div>
  </div>
</template>
