import { defineConfig, presetUno, presetAttributify, presetIcons, presetTypography, presetWebFonts, transformerDirectives, transformerVariantGroup } from 'unocss'
import extractorSvelte from '@unocss/extractor-svelte'

export default defineConfig({
  presets: [
    presetUno(),
    presetAttributify(),
    presetIcons({
      scale: 1.2,
      cdn: 'https://esm.sh/'
    }),
    presetTypography(),
    presetWebFonts({
      fonts: {
        mono: ['JetBrains Mono', 'Roboto Mono', 'SF Mono', 'Monaco', 'Consolas'],
        gothic: ['MS Gothic', 'MS UI Gothic', 'monospace']
      }
    })
  ],
  transformers: [
    transformerDirectives(),
    transformerVariantGroup()
  ],
  extractors: [
    extractorSvelte(),
  ],
  theme: {
    colors: {
      // NieR: Automata Color Palette
      'nier-bg-primary': '#f4f1ea',
      'nier-bg-secondary': '#faf8f3',
      'nier-bg-tertiary': '#e8e3d8',
      'nier-text-primary': '#3a372f',
      'nier-text-secondary': '#75726a',
      'nier-text-muted': '#ada895',
      'nier-border-primary': '#3a372f',
      'nier-border-secondary': '#75726a',
      'nier-border-muted': '#ada895',
      'nier-accent-warm': '#d5cfb7',
      'nier-accent-cool': '#b8b3a1',
      'nier-shadow': 'rgba(58, 55, 47, 0.15)',
      'nier-overlay': 'rgba(58, 55, 47, 0.08)',
      
      // Gothic YoRHa Evidence Colors
      'gothic-bg-primary': '#ffffff',
      'gothic-bg-secondary': '#f8f8f8',
      'gothic-text-primary': '#000000',
      'gothic-text-secondary': '#333333',
      'gothic-text-muted': '#666666',
      'gothic-border-primary': '#000000',
      'gothic-border-secondary': '#333333',
      'gothic-accent': '#cccccc',
      'gothic-shadow': 'rgba(0, 0, 0, 0.25)',
      'gothic-overlay': 'rgba(0, 0, 0, 0.05)',
      
      // AI System Status Colors
      'ai-status-online': '#10b981',
      'ai-status-processing': '#f59e0b',
      'ai-status-offline': '#ef4444',
      'ai-status-warning': '#f97316',
      
      // Multi-Agent Colors
      'agent-autogen': '#8b5cf6',
      'agent-crewai': '#06b6d4',
      'agent-autonomous': '#10b981',
      'agent-semantic': '#3b82f6',
      'agent-memory': '#f59e0b'
    },
    fontFamily: {
      'mono': ['JetBrains Mono', 'Roboto Mono', 'SF Mono', 'Monaco', 'Consolas', 'monospace'],
      'gothic': ['MS Gothic', 'MS UI Gothic', 'monospace']
    },
    spacing: {
      'nier-xs': '4px',
      'nier-sm': '8px',
      'nier-md': '16px',
      'nier-lg': '24px',
      'nier-xl': '32px'
    },
    animation: {
      'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      'bounce-subtle': 'bounce 2s ease-in-out infinite',
      'fade-in': 'fadeIn 0.5s ease-in-out',
      'slide-up': 'slideUp 0.3s ease-out',
      'processing': 'processing 2s linear infinite'
    },
    keyframes: {
      fadeIn: {
        '0%': { opacity: '0', transform: 'translateY(10px)' },
        '100%': { opacity: '1', transform: 'translateY(0)' }
      },
      slideUp: {
        '0%': { transform: 'translateY(100%)', opacity: '0' },
        '100%': { transform: 'translateY(0)', opacity: '1' }
      },
      processing: {
        '0%': { transform: 'translateX(-100%)' },
        '100%': { transform: 'translateX(200%)' }
      }
    }
  },
  shortcuts: {
    // NieR Flexbox Layout
    'yorha-flex-container': 'flex h-full w-full',
    'yorha-flex-small': 'flex-1 min-w-0',
    'yorha-flex-medium': 'flex-[1.618] min-w-0',
    'yorha-flex-large': 'flex-[2.618] min-w-0',
    
    // NieR Components
    'yorha-button': 'bg-nier-bg-secondary border-2 border-nier-border-primary text-foreground px-4 py-2 uppercase tracking-wider transition-all duration-150 relative overflow-hidden cursor-pointer min-h-10 flex items-center justify-center hover:bg-nier-bg-tertiary hover:-translate-y-px hover:shadow-lg active:translate-y-0 active:shadow-sm',
    'yorha-button-primary': 'yorha-button bg-nier-text-primary text-nier-bg-primary border-nier-border-primary hover:bg-nier-text-secondary',
    'yorha-card': 'bg-card border border-nier-border-secondary relative transition-all duration-300 hover:shadow-lg',
    'yorha-card-elevated': 'shadow-lg',
    'yorha-panel': 'bg-card border border-nier-border-secondary relative',
    'yorha-panel-header': 'border-b border-nier-border-secondary p-4 bg-nier-bg-tertiary',
    'yorha-panel-content': 'p-4',
    'yorha-separator': 'h-px bg-nier-border-secondary relative my-4',
    'yorha-input': 'bg-input border border-nier-border-secondary px-4 py-2 text-foreground transition-all duration-150 min-h-10 focus:outline-none focus:border-nier-border-primary focus:shadow-[0_0_0_1px_var(--color-nier-border-primary)]',
    
    // NieR Evidence Board
    'yorha-evidence-board': 'bg-background relative overflow-hidden',
    'yorha-evidence-grid': 'bg-[linear-gradient(var(--color-nier-border-muted)_1px,transparent_1px),linear-gradient(90deg,var(--color-nier-border-muted)_1px,transparent_1px)] bg-[length:20px_20px] opacity-30 absolute inset-0 pointer-events-none',
    
    // Gothic YoRHa Evidence Item - Now with black/white styling
    'yorha-evidence-item': 'bg-gothic-bg-primary border-3 border-gothic-border-primary text-gothic-text-primary absolute transition-all duration-300 cursor-move min-w-50 max-w-75 overflow-hidden hover:scale-102 hover:shadow-xl hover:border-gothic-border-primary',
    'yorha-evidence-item-selected': 'border-gothic-border-primary shadow-[0_0_0_1px_var(--color-gothic-border-primary),0_0_0_3px_var(--color-gothic-bg-primary),0_0_0_4px_var(--color-gothic-border-primary)]',
    
    // NieR Priority Indicators
    'yorha-priority-critical': 'border-l-4 border-l-red-500',
    'yorha-priority-high': 'border-l-4 border-l-orange-500',
    'yorha-priority-medium': 'border-l-4 border-l-yellow-600',
    'yorha-priority-low': 'border-l-4 border-l-gothic-border-secondary',
    
    // NieR Drop Zones
    'yorha-drop-zone': 'border-2 border-dashed border-nier-border-secondary bg-nier-overlay transition-all duration-300',
    'yorha-drop-zone-active': 'border-nier-border-primary bg-nier-bg-tertiary shadow-[0_0_20px_var(--color-nier-shadow)]',
    'yorha-drop-zone-reject': 'border-red-500 bg-red-500 bg-opacity-10',
    
    // NieR Media
    'yorha-media-preview': 'relative overflow-hidden border border-nier-border-secondary bg-nier-bg-tertiary',
    'yorha-media-overlay': 'absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-nier-overlay opacity-0 transition-opacity duration-300',
    'yorha-media-controls': 'absolute bottom-2 left-2 right-2 flex justify-between items-center opacity-0 translate-y-2 transition-all duration-300',
    
    // NieR Loading and Animation
    'yorha-loading': 'relative overflow-hidden',
    'yorha-pulse': 'animate-pulse',
    
    // NieR Scrollbar
    'yorha-scrollbar': '[&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar]:h-2 [&::-webkit-scrollbar-track]:bg-nier-bg-tertiary [&::-webkit-scrollbar-thumb]:bg-nier-border-secondary [&::-webkit-scrollbar-thumb]:rounded-none [&::-webkit-scrollbar-thumb:hover]:bg-nier-border-primary [&::-webkit-scrollbar-corner]:bg-nier-bg-tertiary',
    
    // AI System Components
    'ai-status-indicator': 'w-3 h-3 rounded-full transition-all duration-300',
    'ai-status-online': 'bg-ai-status-online animate-pulse-slow',
    'ai-status-processing': 'bg-ai-status-processing animate-bounce-subtle',
    'ai-status-offline': 'bg-ai-status-offline',
    'ai-status-warning': 'bg-ai-status-warning',
    
    // Multi-Agent System Components
    'agent-card': 'yorha-card p-4 space-y-3 hover:scale-[1.02] transition-transform duration-200',
    'agent-autogen': 'border-l-4 border-l-agent-autogen bg-gradient-to-r from-purple-50 to-transparent',
    'agent-crewai': 'border-l-4 border-l-agent-crewai bg-gradient-to-r from-cyan-50 to-transparent',
    'agent-autonomous': 'border-l-4 border-l-agent-autonomous bg-gradient-to-r from-green-50 to-transparent',
    'agent-semantic': 'border-l-4 border-l-agent-semantic bg-gradient-to-r from-blue-50 to-transparent',
    'agent-memory': 'border-l-4 border-l-agent-memory bg-gradient-to-r from-amber-50 to-transparent',
    
    // Copilot Integration Components
    'copilot-prompt-area': 'yorha-panel min-h-32 relative',
    'copilot-response-area': 'yorha-panel-content max-h-96 overflow-y-auto yorha-scrollbar',
    'copilot-status-bar': 'flex items-center justify-between p-2 bg-nier-bg-tertiary border-t border-nier-border-secondary text-sm',
    
    // Processing States
    'processing-overlay': 'absolute inset-0 bg-nier-overlay z-10 flex items-center justify-center',
    'processing-bar': 'relative h-1 bg-nier-bg-tertiary overflow-hidden',
    'processing-indicator': 'absolute top-0 left-0 h-full bg-gradient-to-r from-blue-500 to-purple-500 animate-processing',
    
    // Results Display
    'result-card': 'yorha-card animate-fade-in',
    'result-header': 'yorha-panel-header flex items-center justify-between',
    'result-content': 'yorha-panel-content prose prose-sm max-w-none',
    'result-metadata': 'text-xs text-nier-text-muted space-y-1',
    
    // Action Items
    'action-item': 'flex items-start gap-3 p-3 border border-nier-border-secondary rounded-lg hover:bg-nier-bg-secondary transition-colors',
    'action-priority-critical': 'action-item yorha-priority-critical bg-red-50',
    'action-priority-high': 'action-item yorha-priority-high bg-orange-50',
    'action-priority-medium': 'action-item yorha-priority-medium bg-yellow-50',
    'action-priority-low': 'action-item yorha-priority-low bg-gray-50',
    
    // Recommendation Cards
    'recommendation-card': 'yorha-card p-4 space-y-2',
    'recommendation-architecture': 'recommendation-card border-l-4 border-l-blue-500',
    'recommendation-performance': 'recommendation-card border-l-4 border-l-green-500',
    'recommendation-security': 'recommendation-card border-l-4 border-l-red-500',
    'recommendation-testing': 'recommendation-card border-l-4 border-l-purple-500',
    'recommendation-deployment': 'recommendation-card border-l-4 border-l-orange-500',
    
    // Execution Plan
    'execution-phase': 'yorha-panel mb-4',
    'execution-phase-header': 'yorha-panel-header',
    'execution-phase-content': 'yorha-panel-content',
    'execution-phase-parallel': 'border-l-4 border-l-green-500',
    'execution-phase-sequential': 'border-l-4 border-l-orange-500',
    
    // Demo Components
    'demo-example-card': 'yorha-card p-4 cursor-pointer hover:bg-nier-bg-secondary transition-colors',
    'demo-config-section': 'space-y-4',
    'demo-results-section': 'space-y-6 animate-slide-up',
    
    // Bits UI and Shadcn-Svelte Component Shortcuts
    // Enhanced Button Components
    'bits-btn': 'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
    'bits-btn-default': 'bits-btn bg-primary text-primary-foreground hover:bg-primary/90',
    'bits-btn-destructive': 'bits-btn bg-destructive text-destructive-foreground hover:bg-destructive/90',
    'bits-btn-outline': 'bits-btn border border-input bg-background hover:bg-accent hover:text-accent-foreground',
    'bits-btn-secondary': 'bits-btn bg-secondary text-secondary-foreground hover:bg-secondary/80',
    'bits-btn-ghost': 'bits-btn hover:bg-accent hover:text-accent-foreground',
    'bits-btn-link': 'bits-btn text-primary underline-offset-4 hover:underline',
    'bits-btn-sm': 'h-9 rounded-md px-3',
    'bits-btn-lg': 'h-11 rounded-md px-8',
    'bits-btn-icon': 'h-10 w-10',
    
    // Enhanced Card Components
    'shadcn-card': 'rounded-lg border bg-card text-card-foreground shadow-sm',
    'shadcn-card-header': 'flex flex-col space-y-1.5 p-6',
    'shadcn-card-title': 'text-2xl font-semibold leading-none tracking-tight',
    'shadcn-card-description': 'text-sm text-muted-foreground',
    'shadcn-card-content': 'p-6 pt-0',
    'shadcn-card-footer': 'flex items-center p-6 pt-0',
    
    // Dialog and Modal Components
    'bits-dialog-overlay': 'fixed inset-0 z-50 bg-background/80 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
    'bits-dialog-content': 'fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg',
    'bits-dialog-header': 'flex flex-col space-y-1.5 text-center sm:text-left',
    'bits-dialog-footer': 'flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2',
    
    // Select Components
    'bits-select-trigger': 'flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1',
    'bits-select-content': 'relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
    'bits-select-item': 'relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
    
    // Input Components
    'bits-input': 'flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
    'bits-textarea': 'flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
    'bits-label': 'text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70',
    
    // Dropdown Menu Components
    'bits-dropdown-content': 'z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
    'bits-dropdown-item': 'relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
    'bits-dropdown-separator': 'h-px bg-muted my-1',
    
    // Context Menu Components
    'bits-context-content': 'z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-lg data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
    'bits-context-item': 'relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
    
    // Tooltip Components
    'bits-tooltip-content': 'z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
    
    // Badge Components
    'bits-badge': 'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
    'bits-badge-default': 'bits-badge border-transparent bg-primary text-primary-foreground hover:bg-primary/80',
    'bits-badge-secondary': 'bits-badge border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80',
    'bits-badge-destructive': 'bits-badge border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80',
    'bits-badge-outline': 'bits-badge text-foreground',
    
    // Separator Components
    'bits-separator': 'shrink-0 bg-border h-[1px] w-full',
    'bits-separator-vertical': 'shrink-0 bg-border w-[1px] h-full',
    
    // Scroll Area Components
    'bits-scroll-area': 'relative overflow-hidden',
    'bits-scroll-viewport': 'h-full w-full rounded-[inherit]',
    'bits-scroll-bar': 'flex touch-none select-none transition-colors',
    'bits-scroll-thumb': 'relative flex-1 rounded-full bg-border',
    
    // NieR + Bits UI Hybrid Components
    'nier-bits-card': 'yorha-card shadcn-card',
    'nier-bits-button': 'yorha-button bits-btn',
    'nier-bits-input': 'yorha-input bits-input',
    'nier-bits-dialog': 'bits-dialog-content yorha-panel',
    'nier-bits-select': 'bits-select-trigger yorha-input',
    
    // Vector Intelligence UI Components
    'vector-search-input': 'bits-input w-full relative placeholder:text-muted-foreground/70 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500',
    'vector-result-item': 'yorha-card p-4 hover:bg-accent/50 transition-all duration-200 cursor-pointer border-l-4 border-l-transparent hover:border-l-blue-500',
    'vector-confidence-badge': 'bits-badge inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full',
    'vector-confidence-high': 'vector-confidence-badge bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    'vector-confidence-medium': 'vector-confidence-badge bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    'vector-confidence-low': 'vector-confidence-badge bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    'vector-metadata-grid': 'grid grid-cols-2 md:grid-cols-4 gap-3 text-sm text-muted-foreground',
    'vector-highlight': 'bg-yellow-200 dark:bg-yellow-800/30 px-1 py-0.5 rounded text-foreground',
    
    // AI Recommendation Components
    'recommendation-container': 'space-y-4',
    'recommendation-item': 'vector-result-item border-l-4',
    'recommendation-action': 'recommendation-item border-l-blue-500',
    'recommendation-insight': 'recommendation-item border-l-green-500',
    'recommendation-warning': 'recommendation-item border-l-red-500',
    'recommendation-opportunity': 'recommendation-item border-l-purple-500',
    'recommendation-header': 'flex items-center justify-between mb-2',
    'recommendation-title': 'font-semibold text-foreground',
    'recommendation-category': 'bits-badge-outline text-xs',
    'recommendation-description': 'text-sm text-muted-foreground mb-3',
    'recommendation-actions': 'flex items-center gap-2 text-xs text-muted-foreground',
    
    // Semantic Analysis Components
    'semantic-entity-container': 'flex flex-wrap gap-2 mb-4',
    'semantic-entity-tag': 'bits-badge-secondary text-xs px-2 py-1 rounded-md',
    'semantic-entity-person': 'semantic-entity-tag bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
    'semantic-entity-organization': 'semantic-entity-tag bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400',
    'semantic-entity-location': 'semantic-entity-tag bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    'semantic-entity-date': 'semantic-entity-tag bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400',
    'semantic-entity-legal': 'semantic-entity-tag bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    
    // Theme Analysis Components
    'theme-weight-bar': 'h-2 bg-muted rounded-full overflow-hidden',
    'theme-weight-fill': 'h-full bg-primary transition-all duration-300',
    'theme-item': 'flex items-center justify-between p-3 border rounded-lg',
    'theme-title': 'font-medium text-sm',
    'theme-weight': 'text-xs text-muted-foreground',
    
    // Relationship Mapping Components
    'relationship-container': 'space-y-2',
    'relationship-item': 'flex items-center gap-2 p-2 bg-muted/50 rounded-lg text-sm',
    'relationship-from': 'font-medium text-foreground',
    'relationship-type': 'text-xs text-muted-foreground px-2 py-1 bg-background rounded',
    'relationship-to': 'font-medium text-foreground',
    'relationship-strength': 'ml-auto text-xs font-mono'
  },
  rules: [
    // Flex golden ratio rules
    [/^flex-golden-(\w+)$/, ([, ratio]) => {
      const ratios = {
        'small': '1',
        'medium': '1.618',
        'large': '2.618'
      };
      return ratios[ratio] ? { flex: ratios[ratio] } : {};
    }],
    
    // NieR spacing rules
    [/^nier-p-(\w+)$/, ([, size]) => {
      const sizes = {
        'xs': '4px',
        'sm': '8px',
        'md': '16px',
        'lg': '24px',
        'xl': '32px'
      };
      return sizes[size] ? { padding: sizes[size] } : {};
    }],
    
    [/^nier-m-(\w+)$/, ([, size]) => {
      const sizes = {
        'xs': '4px',
        'sm': '8px',
        'md': '16px',
        'lg': '24px',
        'xl': '32px'
      };
      return sizes[size] ? { margin: sizes[size] } : {};
    }],
    
    // AI status dynamic rules
    [/^ai-confidence-(\d+)$/, ([, confidence]) => {
      const conf = parseInt(confidence);
      if (conf >= 80) return { 'border-color': '#10b981', 'background-color': 'rgba(16, 185, 129, 0.1)' };
      if (conf >= 60) return { 'border-color': '#f59e0b', 'background-color': 'rgba(245, 158, 11, 0.1)' };
      return { 'border-color': '#ef4444', 'background-color': 'rgba(239, 68, 68, 0.1)' };
    }],
    
    // Priority weight rules
    [/^priority-weight-(\d+)$/, ([, weight]) => {
      const w = parseInt(weight);
      if (w >= 90) return { 'font-weight': '700', 'color': '#dc2626' };
      if (w >= 70) return { 'font-weight': '600', 'color': '#ea580c' };
      if (w >= 50) return { 'font-weight': '500', 'color': '#ca8a04' };
      return { 'font-weight': '400', 'color': '#6b7280' };
    }]
  ],
  safelist: [
    // NieR Core Components
    'yorha-flex-container',
    'yorha-flex-small',
    'yorha-flex-medium',
    'yorha-flex-large',
    'yorha-button',
    'yorha-button-primary',
    'yorha-card',
    'yorha-card-elevated',
    'yorha-panel',
    'yorha-panel-header',
    'yorha-panel-content',
    'yorha-separator',
    'yorha-input',
    
    // NieR Evidence Board
    'yorha-evidence-board',
    'yorha-evidence-grid',
    'yorha-evidence-item',
    'yorha-evidence-item-selected',
    
    // NieR Priority Classes
    'yorha-priority-critical',
    'yorha-priority-high',
    'yorha-priority-medium',
    'yorha-priority-low',
    
    // NieR Drop Zones
    'yorha-drop-zone',
    'yorha-drop-zone-active',
    'yorha-drop-zone-reject',
    
    // NieR Media
    'yorha-media-preview',
    'yorha-media-overlay',
    'yorha-media-controls',
    
    // NieR States
    'yorha-loading',
    'yorha-pulse',
    'yorha-scrollbar',
    
    // AI System Classes
    'ai-status-indicator',
    'ai-status-online',
    'ai-status-processing',
    'ai-status-offline',
    'ai-status-warning',
    
    // Multi-Agent Classes
    'agent-card',
    'agent-autogen',
    'agent-crewai',
    'agent-autonomous',
    'agent-semantic',
    'agent-memory',
    
    // Copilot Classes
    'copilot-prompt-area',
    'copilot-response-area',
    'copilot-status-bar',
    
    // Processing Classes
    'processing-overlay',
    'processing-bar',
    'processing-indicator',
    
    // Results Classes
    'result-card',
    'result-header',
    'result-content',
    'result-metadata',
    
    // Action Classes
    'action-item',
    'action-priority-critical',
    'action-priority-high',
    'action-priority-medium',
    'action-priority-low',
    
    // Recommendation Classes
    'recommendation-card',
    'recommendation-architecture',
    'recommendation-performance',
    'recommendation-security',
    'recommendation-testing',
    'recommendation-deployment',
    
    // Execution Plan Classes
    'execution-phase',
    'execution-phase-header',
    'execution-phase-content',
    'execution-phase-parallel',
    'execution-phase-sequential',
    
    // Demo Classes
    'demo-example-card',
    'demo-config-section',
    'demo-results-section',
    
    // NieR Color classes
    'bg-nier-bg-primary',
    'bg-nier-bg-secondary',
    'bg-nier-bg-tertiary',
    'text-nier-text-primary',
    'text-nier-text-secondary',
    'text-nier-text-muted',
    'border-nier-border-primary',
    'border-nier-border-secondary',
    'border-nier-border-muted',
    
    // Gothic Color classes
    'bg-gothic-bg-primary',
    'bg-gothic-bg-secondary',
    'text-gothic-text-primary',
    'text-gothic-text-secondary',
    'text-gothic-text-muted',
    'border-gothic-border-primary',
    'border-gothic-border-secondary',
    
    // AI Status Colors
    'bg-ai-status-online',
    'bg-ai-status-processing',
    'bg-ai-status-offline',
    'bg-ai-status-warning',
    'text-ai-status-online',
    'text-ai-status-processing',
    'text-ai-status-offline',
    'text-ai-status-warning',
    
    // Agent Colors
    'bg-agent-autogen',
    'bg-agent-crewai',
    'bg-agent-autonomous',
    'bg-agent-semantic',
    'bg-agent-memory',
    'text-agent-autogen',
    'text-agent-crewai',
    'text-agent-autonomous',
    'text-agent-semantic',
    'text-agent-memory',
    'border-agent-autogen',
    'border-agent-crewai',
    'border-agent-autonomous',
    'border-agent-semantic',
    'border-agent-memory',
    
    // Animation classes
    'animate-pulse-slow',
    'animate-bounce-subtle',
    'animate-fade-in',
    'animate-slide-up',
    'animate-processing',
    
    // Bits UI and Shadcn-Svelte Integration Classes
    // Button variants and states
    'btn-default',
    'btn-primary',
    'btn-secondary',
    'btn-outline',
    'btn-ghost',
    'btn-danger',
    'btn-destructive',
    'btn-success',
    'btn-warning',
    'btn-info',
    'btn-nier',
    'btn-crimson',
    'btn-gold',
    'btn-xs',
    'btn-sm',
    'btn-md',
    'btn-lg',
    'btn-xl',
    'btn-loading',
    'nier-btn',
    
    // Card and Panel Components
    'bg-card',
    'text-card-foreground',
    'border-card',
    'shadow-card',
    'bg-muted',
    'text-muted-foreground',
    'bg-popover',
    'text-popover-foreground',
    'border-popover',
    'bg-background',
    'text-foreground',
    'border-border',
    'border-input',
    'bg-input',
    'bg-primary',
    'text-primary-foreground',
    'bg-secondary',
    'text-secondary-foreground',
    'bg-accent',
    'text-accent-foreground',
    'ring-ring',
    'text-destructive',
    'border-destructive',
    'bg-destructive',
    'text-destructive-foreground',
    
    // Dialog and Modal States
    'data-[state=open]:animate-in',
    'data-[state=closed]:animate-out',
    'data-[state=closed]:fade-out-0',
    'data-[state=open]:fade-in-0',
    'data-[state=closed]:zoom-out-95',
    'data-[state=open]:zoom-in-95',
    'data-[side=bottom]:slide-in-from-top-2',
    'data-[side=left]:slide-in-from-right-2',
    'data-[side=right]:slide-in-from-left-2',
    'data-[side=top]:slide-in-from-bottom-2',
    
    // Dropdown and Context Menu States
    'data-[highlighted]:bg-accent',
    'data-[highlighted]:text-accent-foreground',
    'data-[disabled]:pointer-events-none',
    'data-[disabled]:opacity-50',
    
    // Select Component States
    'data-[placeholder]:text-muted-foreground',
    'data-[state=open]:border-ring',
    'focus:border-ring',
    'focus:ring-2',
    'focus:ring-ring',
    'focus:ring-offset-2',
    
    // Tooltip and Hover States
    'hover:bg-accent',
    'hover:text-accent-foreground',
    'hover:border-accent',
    'focus-visible:outline-none',
    'focus-visible:ring-2',
    'focus-visible:ring-ring',
    'focus-visible:ring-offset-2',
    
    // Accessibility and Screen Reader
    'sr-only',
    'not-sr-only',
    'focus:not-sr-only',
    
    // Grid and Layout
    'grid-cols-1',
    'grid-cols-2',
    'grid-cols-3',
    'grid-cols-4',
    'grid-cols-5',
    'grid-cols-6',
    'grid-cols-12',
    'md:grid-cols-2',
    'md:grid-cols-3',
    'md:grid-cols-4',
    'lg:grid-cols-3',
    'lg:grid-cols-4',
    'lg:grid-cols-5',
    'xl:grid-cols-4',
    'xl:grid-cols-5',
    'xl:grid-cols-6',
    
    // Responsive Design
    'sm:text-sm',
    'md:text-base',
    'lg:text-lg',
    'xl:text-xl',
    'sm:p-4',
    'md:p-6',
    'lg:p-8',
    'xl:p-10',
    'sm:m-2',
    'md:m-4',
    'lg:m-6',
    'xl:m-8',
    
    // Dark Mode Support
    'dark:bg-card',
    'dark:text-card-foreground',
    'dark:border-card',
    'dark:bg-background',
    'dark:text-foreground',
    'dark:border-border',
    'dark:bg-muted',
    'dark:text-muted-foreground',
    'dark:bg-accent',
    'dark:text-accent-foreground',
    'dark:bg-destructive',
    'dark:text-destructive-foreground',
    'dark:ring-ring',
    
    // Vector Intelligence Components
    'vector-search-container',
    'vector-result-card',
    'vector-confidence-indicator',
    'vector-metadata-panel',
    'recommendation-confidence-high',
    'recommendation-confidence-medium',
    'recommendation-confidence-low',
    'semantic-highlight',
    'entity-tag',
    'relationship-connector',
    'theme-weight-indicator'
  ]
})
