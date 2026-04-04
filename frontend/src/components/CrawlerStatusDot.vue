<script setup lang="ts">
import type { CrawlerStatus } from '../api'

defineProps<{ status: CrawlerStatus | null }>()
</script>

<template>
  <div class="flex items-center gap-2 text-xs text-muted px-3 py-2">
    <!-- Status dot -->
    <span
      class="w-2 h-2 rounded-full flex-shrink-0"
      :class="status?.running ? 'bg-green-400 animate-pulse' : 'bg-border'"
    />
    <span v-if="status?.running">
      Crawling
      <span v-if="status.current_accession" class="font-mono text-accent">
        {{ status.current_accession }}
      </span>
    </span>
    <span v-else-if="status?.last_run_finished">
      Last crawl: {{ new Date(status.last_run_finished).toLocaleTimeString() }}
    </span>
    <span v-else>Crawler idle</span>
    <span v-if="status?.indexed_count" class="text-accent/70">
      ({{ status.indexed_count }} indexed)
    </span>
  </div>
</template>
