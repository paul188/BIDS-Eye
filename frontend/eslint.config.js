import pluginVue from 'eslint-plugin-vue'
import { defineConfigWithVueTs, vueTsConfigs } from '@vue/eslint-config-typescript'
import skipFormatting from '@vue/eslint-config-prettier/skip-formatting'

// Lint rules only; layout is Prettier's (skipFormatting turns those rules off).
export default defineConfigWithVueTs(
  { name: 'app/files-to-lint', files: ['**/*.{ts,vue}'] },
  { name: 'app/files-to-ignore', ignores: ['dist/**'] },
  pluginVue.configs['flat/essential'],
  vueTsConfigs.recommended,
  skipFormatting,
)
