<script setup lang="ts">
// The BIDS-Eye mark: an eye whose iris is a scan slice. Used in the header,
// the start page and the sign-in card (and mirrored by public/favicon.svg).
withDefaults(defineProps<{ size?: number }>(), { size: 28 })
</script>

<template>
  <svg :width="size" :height="size" viewBox="0 0 32 32" aria-hidden="true" class="flex-shrink-0">
    <rect width="32" height="32" rx="8" class="fill-accent" />
    <path
      d="M5 16s4.2-7.5 11-7.5S27 16 27 16s-4.2 7.5-11 7.5S5 16 5 16z"
      fill="none"
      stroke="#fff"
      stroke-width="2"
      stroke-linejoin="round"
    />
    <circle cx="16" cy="16" r="4.2" fill="#fff" />
    <circle cx="16" cy="16" r="1.8" class="fill-accent" />
  </svg>
</template>
