import { defineConfig, presetUno, presetIcons } from 'unocss'
import extractorSvelte from '@unocss/extractor-svelte'

export default defineConfig({
  presets: [
    presetUno(),
    presetIcons(),
  ],
  extractors: [
    extractorSvelte(),
  ],
})