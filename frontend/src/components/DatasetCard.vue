<script setup lang="ts">
import { computed } from 'vue'
import type { Dataset } from '../api'

const props = defineProps<{ dataset: Dataset }>()

// The backend sends at most this many participants per dataset
// (_MAX_PARTICIPANTS_PER_DATASET in backend/routers/query.py). Below it the
// list is complete and counts are exact; at it, they would undercount.
const PARTICIPANT_CAP = 50
const MAX_GROUPS_SHOWN = 4

const CONTROL_LABELS = new Set([
  'control', 'controls', 'hc', 'healthy', 'healthy control', 'healthy controls',
  'healthy_control', 'ctrl', 'ctl', 'cn', 'nc', 'con', 'normal', 'typical',
])

const href = computed(() => {
  const d = props.dataset
  if (d.accession_id) return `https://openneuro.org/datasets/${d.accession_id}`
  return d.remote_url || null
})

const participantsComplete = computed(() => {
  const n = props.dataset.participants.length
  return n > 0 && n < PARTICIPANT_CAP
})

// Diagnosis groups, patient groups first by size, controls last. Counts only
// when the participant list is complete.
const groups = computed(() => {
  const counts = new Map<string, number>()
  for (const p of props.dataset.participants) {
    const raw = p.diagnosis?.trim()
    if (!raw) continue
    const label = CONTROL_LABELS.has(raw.toLowerCase()) ? 'control' : raw
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  const all = [...counts.entries()]
    .map(([label, n]) => ({ label, n: participantsComplete.value ? n : null }))
    .sort((a, b) => {
      if ((a.label === 'control') !== (b.label === 'control')) return a.label === 'control' ? 1 : -1
      return (b.n ?? 0) - (a.n ?? 0)
    })
  return { shown: all.slice(0, MAX_GROUPS_SHOWN), more: Math.max(0, all.length - MAX_GROUPS_SHOWN) }
})

const ageRange = computed(() => {
  if (!participantsComplete.value) return ''
  const ages = props.dataset.participants
    .map(p => p.age)
    .filter((a): a is number => typeof a === 'number' && a > 0)
  if (!ages.length) return ''
  const lo = Math.round(Math.min(...ages))
  const hi = Math.round(Math.max(...ages))
  return lo === hi ? `${lo}` : `${lo}–${hi}`
})

const sexSplit = computed(() => {
  if (!participantsComplete.value) return ''
  let f = 0, m = 0
  for (const p of props.dataset.participants) {
    const s = p.sex?.trim().toLowerCase()
    if (s === 'f' || s === 'female') f++
    else if (s === 'm' || s === 'male') m++
  }
  return f + m ? `${f} F · ${m} M` : ''
})

const bidsLine = computed(() => {
  const d = props.dataset
  return [d.bids_version && `v${d.bids_version}`, d.validation_status].filter(Boolean).join(' · ')
})

// Authors: first three, then "+N"; institutions follow on the same line.
const authorLine = computed(() => {
  const a = props.dataset.authors
  if (!a || !a.length) return ''
  const shown = a.slice(0, 3).join(', ')
  return a.length > 3 ? `${shown} +${a.length - 3}` : shown
})
const institutionLine = computed(() => (props.dataset.institutions ?? []).join(' · '))
</script>

<template>
  <article class="py-6 grid gap-x-10 gap-y-4 md:grid-cols-[minmax(0,1fr)_16rem]">
    <!-- Left: what the dataset is -->
    <div class="min-w-0 flex flex-col gap-1.5">
      <h3 class="text-[17px] leading-snug font-semibold">
        <a
          v-if="href"
          :href="href"
          target="_blank"
          rel="noopener"
          class="text-accent visited:text-link-visited hover:underline decoration-1 underline-offset-2 break-words"
        >{{ dataset.name }}</a>
        <span v-else class="text-ink break-words">{{ dataset.name }}</span>
      </h3>

      <p v-if="authorLine || institutionLine" class="text-[13px] text-muted break-words">
        <span v-if="authorLine" class="text-ink/80">{{ authorLine }}</span>
        <span v-if="authorLine && institutionLine" class="mx-1.5 text-border-strong">—</span>
        <span v-if="institutionLine">{{ institutionLine }}</span>
      </p>

      <!-- Uncertain tier: which filter could not be checked for this dataset -->
      <p
        v-if="dataset.uncertain_level && dataset.uncertain_reasons"
        class="text-[13px] text-warn-ink"
      >
        <span class="font-semibold">{{ dataset.uncertain_level === 1 ? 'Probably relevant' : 'Possibly relevant' }}:</span>
        {{ dataset.uncertain_reasons }}
      </p>

      <!-- Why the dataset is marked deprecated (e.g. no longer on OpenNeuro) -->
      <p v-if="dataset.deprecated_reason" class="text-[13px] text-danger-ink">
        <span class="inline-block text-[11px] font-semibold uppercase tracking-wider px-1.5 py-px mr-1.5 rounded bg-danger-bg align-[1px]">Deprecated</span>
        {{ dataset.deprecated_reason }}
      </p>

      <!-- Description snippet (always visible, clamped) -->
      <p v-if="dataset.description_text" class="mt-1 text-sm leading-relaxed text-ink/75 line-clamp-3">
        {{ dataset.description_text }}
      </p>
    </div>

    <!-- Right: who was scanned, at a glance -->
    <dl class="max-w-sm md:max-w-none md:border-l md:border-border md:pl-6 grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[13px] content-start">
      <template v-if="dataset.accession_id">
        <dt class="text-muted-soft">Accession</dt>
        <dd>
          <a
            :href="href ?? undefined"
            target="_blank"
            rel="noopener"
            class="font-mono text-accent hover:underline"
            title="Open on OpenNeuro"
          >{{ dataset.accession_id }}&thinsp;↗</a>
        </dd>
      </template>

      <dt class="text-muted-soft">Subjects</dt>
      <dd class="text-ink font-semibold tabular-nums">{{ dataset.subject_count ?? '—' }}</dd>

      <template v-if="groups.shown.length">
        <dt class="text-muted-soft">Groups</dt>
        <dd class="text-ink flex flex-col gap-0.5">
          <span v-for="g in groups.shown" :key="g.label" class="flex justify-between gap-2">
            <span class="break-words min-w-0" :class="g.label === 'control' ? 'text-muted' : ''">{{ g.label }}</span>
            <span v-if="g.n !== null" class="tabular-nums text-muted">{{ g.n }}</span>
          </span>
          <span v-if="groups.more" class="text-muted-soft">+{{ groups.more }} more</span>
          <span v-if="!participantsComplete" class="text-[12px] text-muted-soft leading-snug">
            from the first {{ dataset.participants.length }} participants; others may differ
          </span>
        </dd>
      </template>

      <template v-if="ageRange">
        <dt class="text-muted-soft">Ages</dt>
        <dd class="text-ink tabular-nums">{{ ageRange }}</dd>
      </template>

      <template v-if="sexSplit">
        <dt class="text-muted-soft">Sex</dt>
        <dd class="text-ink tabular-nums">{{ sexSplit }}</dd>
      </template>

      <template v-if="bidsLine">
        <dt class="text-muted-soft">BIDS</dt>
        <dd class="text-muted">{{ bidsLine }}</dd>
      </template>
    </dl>
  </article>
</template>
