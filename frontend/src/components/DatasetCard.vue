<script setup lang="ts">
import type { Dataset } from '../api'

const props = defineProps<{ dataset: Dataset }>()

function openneuroUrl(id: string) {
  return `https://openneuro.org/datasets/${id}`
}

function subjectLabel(n: number | null) {
  if (n === null) return '—'
  return `${n} subject${n !== 1 ? 's' : ''}`
}

function diagnosisList(d: Dataset): string[] {
  const seen = new Set<string>()
  for (const p of d.participants) {
    if (p.diagnosis) seen.add(p.diagnosis)
  }
  return [...seen]
}
</script>

<template>
  <div class="bg-panel border border-border rounded-xl p-4 flex flex-col gap-2 hover:border-accent/60 transition-colors">
    <!-- Header -->
    <div class="flex items-start justify-between gap-2">
      <div class="flex flex-col min-w-0">
        <span class="font-semibold text-white truncate">{{ dataset.name }}</span>
        <span v-if="dataset.accession_id" class="text-xs text-muted font-mono">
          {{ dataset.accession_id }}
        </span>
      </div>
      <div class="flex gap-1 flex-shrink-0">
        <span class="text-xs px-2 py-0.5 rounded-full bg-accent/20 text-accent">
          {{ dataset.source_type }}
        </span>
        <span class="text-xs px-2 py-0.5 rounded-full bg-border text-muted">
          {{ dataset.dataset_type }}
        </span>
      </div>
    </div>

    <!-- Stats row -->
    <div class="flex gap-4 text-sm text-muted">
      <span>👥 {{ subjectLabel(dataset.subject_count) }}</span>
      <span v-if="dataset.bids_version">BIDS {{ dataset.bids_version }}</span>
      <span v-if="dataset.validation_status" :class="dataset.validation_status === 'valid' ? 'text-green-400' : 'text-yellow-400'">
        {{ dataset.validation_status }}
      </span>
    </div>

    <!-- Diagnoses -->
    <div v-if="diagnosisList(dataset).length" class="flex flex-wrap gap-1">
      <span
        v-for="dx in diagnosisList(dataset)"
        :key="dx"
        class="text-xs px-2 py-0.5 rounded-full bg-surface border border-border text-muted"
      >
        {{ dx }}
      </span>
    </div>

    <!-- Link -->
    <div v-if="dataset.accession_id" class="mt-1">
      <a
        :href="openneuroUrl(dataset.accession_id)"
        target="_blank"
        rel="noopener"
        class="text-xs text-accent hover:underline"
      >
        View on OpenNeuro ↗
      </a>
    </div>
  </div>
</template>
