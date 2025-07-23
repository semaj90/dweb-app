import { defineConfig } from 'unocss'
import { presetUno } from '@unocss/preset-uno'
import { presetIcons } from '@unocss/preset-icons'

export default defineConfig({
  presets: [
    presetUno(),
    presetIcons({
      collections: {
        lucide: () => import('@iconify-json/lucide/icons.json').then(i => i.default)
      }
    })
  ],
  theme: {
    colors: {
      'nier-bg': 'hsl(var(--background))',
      'nier-fg': 'hsl(var(--foreground))',
      'nier-surface': 'hsl(var(--card))',
      'nier-accent': 'hsl(var(--accent))',
      'nier-border': 'hsl(var(--border))',
      background: 'hsl(var(--background))',
      foreground: 'hsl(var(--foreground))',
      primary: {
        DEFAULT: 'hsl(var(--primary))',
        foreground: 'hsl(var(--primary-foreground))'
      },
      accent: {
        DEFAULT: 'hsl(var(--accent))',
        foreground: 'hsl(var(--accent-foreground))'
      }
    },
    fontFamily: {
      'nier': ['Gothic A1', 'JetBrains Mono', 'system-ui', 'sans-serif']
    },
    borderRadius: { 'nier': '0.25rem' }
  },
  shortcuts: {
    'nier-btn': 'inline-flex items-center justify-center font-nier text-sm font-medium transition-all duration-200 border border-nier-border bg-nier-surface hover:bg-accent rounded-nier',
    'nier-btn-primary': 'nier-btn bg-primary text-primary-foreground hover:bg-primary/90',
    'nier-input': 'flex w-full border border-nier-border bg-nier-surface px-3 py-2 text-sm font-nier rounded-nier',
    'nier-surface': 'bg-nier-surface border border-nier-border relative',
    'nier-glow': 'shadow-[0_0_10px_hsl(var(--accent)/0.3)]'
  }
})
