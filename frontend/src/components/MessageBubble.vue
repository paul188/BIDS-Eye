<script setup lang="ts">
import type { Message } from '../stores/chat'
import { useChatStore } from '../stores/chat'
import DatasetList from './DatasetList.vue'

const props = defineProps<{ message: Message }>()
const store = useChatStore()

function onPageChange(page: number) {
  store.loadPage(props.message.id, page)
}
</script>

<template>
  <!-- User message -->
  <div v-if="message.role === 'user'" class="flex justify-end">
    <div class="max-w-[80%] bg-user-bubble rounded-2xl rounded-tr-sm px-4 py-3 text-sm text-white">
      {{ message.content }}
    </div>
  </div>

  <!-- Assistant message -->
  <div v-else class="flex justify-start">
    <div class="max-w-[95%] flex flex-col gap-3">
      <!-- Loading animation -->
      <div v-if="message.loading" class="flex items-center gap-2 text-muted text-sm px-1">
        <span class="animate-pulse">●</span>
        <span class="animate-pulse delay-150">●</span>
        <span class="animate-pulse delay-300">●</span>
      </div>

      <!-- Error -->
      <div
        v-else-if="message.error"
        class="bg-red-900/30 border border-red-700/50 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-red-200"
      >
        {{ message.content }}
      </div>

      <!-- Normal response -->
      <template v-else>
        <div
          v-if="message.content"
          class="bg-ai-bubble rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-white"
        >
          {{ message.content }}
        </div>
        <DatasetList
          v-if="message.datasets.length || message.sql"
          :datasets="message.datasets"
          :sql="message.sql"
          :total="message.total"
          :page="message.page"
          :page-size="message.pageSize"
          :loading="message.pageLoading"
          @page-change="onPageChange"
        />
      </template>
    </div>
  </div>
</template>
