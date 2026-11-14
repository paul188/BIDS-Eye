<script setup lang="ts">
import { computed } from 'vue'
import type { Dataset } from '../api'

const props = withDefaults(defineProps<{
  dataset: Dataset
  /** Words of the active filter chips, to bold in the description. */
  terms?: string[]
  /** Chip kinds in effect (diagnosis, age, subjects, sex): those facts are emphasised. */
  emphasis?: string[]
}>(), { terms: () => [], emphasis: () => [] })

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

// The backend's counts over all participants; older backends send only the
// capped list, counted here when it is complete.
const summary = computed(() => props.dataset.participant_summary ?? null)

const participantsComplete = computed(() => {
  if (summary.value) return true
  const n = props.dataset.participants.length
  return n > 0 && n < PARTICIPANT_CAP
})

// Diagnosis groups, patient groups first by size, controls last. Counts only
// when they cover every participant.
const groups = computed(() => {
  const raw = new Map<string, number>()
  if (summary.value) {
    for (const [label, n] of Object.entries(summary.value.diagnoses)) raw.set(label, n)
  } else {
    for (const p of props.dataset.participants) {
      const label = p.diagnosis?.trim()
      if (label) raw.set(label, (raw.get(label) ?? 0) + 1)
    }
  }
  const counts = new Map<string, number>()
  for (const [rawLabel, n] of raw) {
    const label = CONTROL_LABELS.has(rawLabel.toLowerCase()) ? 'control' : rawLabel
    counts.set(label, (counts.get(label) ?? 0) + n)
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
  let lo: number, hi: number
  if (summary.value) {
    if (summary.value.age_min === null || summary.value.age_max === null) return ''
    lo = Math.round(summary.value.age_min)
    hi = Math.round(summary.value.age_max)
  } else {
    const ages = props.dataset.participants
      .map(p => p.age)
      .filter((a): a is number => typeof a === 'number' && a > 0)
    if (!ages.length) return ''
    lo = Math.round(Math.min(...ages))
    hi = Math.round(Math.max(...ages))
  }
  return lo === hi ? `${lo}` : `${lo}–${hi}`
})

const sexSplit = computed(() => {
  if (!participantsComplete.value) return ''
  let f = 0, m = 0
  if (summary.value) {
    f = summary.value.female
    m = summary.value.male
  } else {
    for (const p of props.dataset.participants) {
      const s = p.sex?.trim().toLowerCase()
      if (s === 'f' || s === 'female') f++
      else if (s === 'm' || s === 'male') m++
    }
  }
  return f + m ? `${f} F, ${m} M` : ''
})

// Authors: first three, then "+N"; institutions follow on the same line.
const authorLine = computed(() => {
  const a = props.dataset.authors
  if (!a || !a.length) return ''
  const shown = a.slice(0, 3).join(', ')
  return a.length > 3 ? `${shown} +${a.length - 3}` : shown
})
const institutionLine = computed(() => (props.dataset.institutions ?? []).join(', '))

// "42 PD, 4 control" (counts only when the participant list is complete).
const groupsText = computed(() => {
  const parts = groups.value.shown.map(g => (g.n !== null ? `${g.n} ${g.label}` : g.label))
  if (groups.value.more) parts.push(`+${groups.value.more} more`)
  return parts.join(', ')
})

// A fact the search filtered on is shown in ink and semibold, the rest muted.
const emphasised = computed(() => new Set(props.emphasis))
function factClass(kind: string): string {
  if (!emphasised.value.size) return 'text-ink'
  return emphasised.value.has(kind) ? 'text-ink font-semibold' : 'text-muted'
}

// Description with the search words bolded. When the first match lies past
// what three lines show, start at the sentence that contains it.
const SNIPPET_START_LIMIT = 220

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

const termPattern = computed(() => {
  const terms = [...new Set(props.terms.map(t => t.trim()).filter(Boolean))]
    .sort((a, b) => b.length - a.length)
  if (!terms.length) return null
  // Prefix match, so "parkinson" also marks "Parkinson's" and "parkinsonism".
  return new RegExp(`\\b(?:${terms.map(escapeRegExp).join('|')})[\\w'’-]*`, 'gi')
})

// The backend marks README matches with ⟦ ⟧ (Postgres ts_headline, so
// stemming is right: "resting" also marks "rest").
function markedParts(text: string): { text: string; hit: boolean }[] {
  return text.split(/(⟦[^⟧]*⟧)/).filter(Boolean).map(part =>
    part.startsWith('⟦') ? { text: part.slice(1, -1), hit: true } : { text: part, hit: false })
}

// What describes the dataset: the README passage; for a thin README, the
// generated summary instead; the description field when there is neither.
const generated = computed(() => !!props.dataset.generated_summary)
const hasDescription = computed(() =>
  generated.value || !!(props.dataset.snippet || props.dataset.description_text))

const snippet = computed(() => {
  if (generated.value) return highlighted(props.dataset.generated_summary ?? '')
  if (props.dataset.snippet) return markedParts(props.dataset.snippet)
  return highlighted(props.dataset.description_text ?? '')
})

// *text* with the search words bolded, starting at the sentence of the first
// match when that lies past what three lines show.
function highlighted(text: string): { text: string; hit: boolean }[] {
  const pattern = termPattern.value
  if (!text || !pattern) return [{ text, hit: false }]

  pattern.lastIndex = 0
  const first = pattern.exec(text)
  let body = text
  if (first && first.index > SNIPPET_START_LIMIT) {
    const before = text.slice(0, first.index)
    const sentenceStart = Math.max(before.lastIndexOf('. '), before.lastIndexOf('\n'))
    if (sentenceStart > 0) body = '… ' + text.slice(sentenceStart + 1).trimStart()
  }

  const parts: { text: string; hit: boolean }[] = []
  let last = 0
  pattern.lastIndex = 0
  for (const match of body.matchAll(pattern)) {
    const at = match.index ?? 0
    if (at > last) parts.push({ text: body.slice(last, at), hit: false })
    parts.push({ text: match[0], hit: true })
    last = at + match[0].length
  }
  if (last < body.length) parts.push({ text: body.slice(last), hit: false })
  return parts
}
</script>

<template>
  <article class="py-5 flex flex-col gap-1.5">
    <h3 class="text-[17px] leading-snug font-semibold">
      <a
        v-if="href"
        :href="href"
        target="_blank"
        rel="noopener"
        class="text-link visited:text-link-visited hover:underline decoration-1 underline-offset-2 break-words"
      >{{ dataset.name }}</a>
      <span v-else class="text-ink break-words">{{ dataset.name }}</span>
    </h3>
    <p v-if="authorLine || institutionLine" class="-mt-0.5 text-[13px] text-muted break-words">
      <span v-if="authorLine">{{ authorLine }}</span>
      <span v-if="authorLine && institutionLine"> — </span>
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
      <span class="font-semibold">Deprecated:</span>
      {{ dataset.deprecated_reason }}
    </p>

    <!-- What the dataset is: the description, at a readable line length -->
    <div v-if="hasDescription" class="max-w-3xl">
      <p class="text-sm leading-relaxed text-ink/80 line-clamp-3">
        <template v-for="(part, i) in snippet" :key="i"><strong v-if="part.hit" class="font-semibold text-ink">{{ part.text }}</strong><template v-else>{{ part.text }}</template></template>
      </p>
      <p v-if="generated" class="mt-0.5 text-[12px] text-muted-soft">
        Generated summary: the dataset's README is short.
      </p>
    </div>

    <!-- The paper, only when checked to report this dataset -->
    <p v-if="dataset.linked_paper" class="text-[13px] text-muted max-w-3xl">
      Linked paper:
      <a
        v-if="dataset.linked_paper.url"
        :href="dataset.linked_paper.url"
        target="_blank"
        rel="noopener"
        class="text-link visited:text-link-visited hover:underline"
      >{{ dataset.linked_paper.title }}</a>
      <span v-else class="text-ink">{{ dataset.linked_paper.title }}</span>
      <template v-if="dataset.linked_paper.year"> ({{ dataset.linked_paper.year }})</template>,
      {{ dataset.linked_paper.relation_label }}
    </p>

    <!-- Who was scanned: same order on every row -->
    <p class="flex flex-wrap items-baseline gap-x-5 gap-y-1 text-[13px]">
      <a
        v-if="dataset.accession_id && href"
        :href="href"
        target="_blank"
        rel="noopener"
        class="font-mono text-link visited:text-link-visited hover:underline"
        title="Open on OpenNeuro"
      >{{ dataset.accession_id }}</a>
      <span v-if="dataset.subject_count !== null" :class="factClass('subjects')">
        <span class="font-semibold tabular-nums">{{ dataset.subject_count }}</span>
        subject{{ dataset.subject_count === 1 ? '' : 's' }}
      </span>
      <span
        v-if="groups.shown.length"
        class="tabular-nums"
        :class="factClass('diagnosis')"
        :title="participantsComplete ? undefined : `Groups seen in the first ${dataset.participants.length} participants; others may differ`"
      >
        {{ groupsText }}
        <span v-if="!participantsComplete" class="font-normal text-muted-soft">(first {{ dataset.participants.length }} participants)</span>
      </span>
      <span v-if="ageRange" class="tabular-nums" :class="factClass('age')">ages {{ ageRange }}</span>
      <span v-if="sexSplit" class="tabular-nums" :class="factClass('sex')">{{ sexSplit }}</span>
    </p>

  </article>
</template>
