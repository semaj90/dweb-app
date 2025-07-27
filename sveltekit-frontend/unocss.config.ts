import {
  defineConfig,
  presetUno,
  presetAttributify,
  presetIcons,
  transformerDirectives,
  transformerVariantGroup,
} from "unocss";

export default defineConfig({
  // Legal-focused design system with PicoCSS compatibility
  presets: [
    presetUno(),
    presetAttributify(),
    presetIcons({
      scale: 1.2,
      cdn: 'https://esm.sh/',
      collections: {
        lucide: () => import('@iconify-json/lucide/icons.json').then(i => i.default).catch(() => ({})),
        mdi: () => import('@iconify-json/mdi/icons.json').then(i => i.default).catch(() => ({})),
        tabler: () => import('@iconify-json/tabler/icons.json').then(i => i.default).catch(() => ({})),
        heroicons: () => import('@iconify-json/heroicons/icons.json').then(i => i.default).catch(() => ({}))
      },
      autoInstall: true
    }),
    // REMOVED presetWebFonts to fix timeout error
    // presetWebFonts({
    //   fonts: {
    //     'inter': 'Inter:400,500,600,700',
    //     'fira-code': 'Fira Code:400,500',
    //   },
    // }),
  ],
  transformers: [transformerDirectives(), transformerVariantGroup()],
  theme: {
    colors: {
      // Legal-focused color palette
      primary: {
        50: "#f0f9ff",
        100: "#e0f2fe",
        200: "#bae6fd",
        300: "#7dd3fc",
        400: "#38bdf8",
        500: "#0ea5e9",
        600: "#0284c7",
        700: "#0369a1",
        800: "#075985",
        900: "#0c4a6e",
      },
      legal: {
        navy: "#1e3a8a",
        gold: "#d97706",
        neutral: "#374151",
        success: "#059669",
        warning: "#d97706",
        error: "#dc2626",
      },
      semantic: {
        contract: "#3b82f6",
        evidence: "#059669",
        statute: "#7c3aed",
        case: "#dc2626",
        regulation: "#ea580c",
      },
    },
    fontFamily: {
      // Use system fonts as fallbacks to prevent timeout issues
      sans: [
        "system-ui",
        "-apple-system",
        "BlinkMacSystemFont",
        "Segoe UI",
        "Roboto",
        "sans-serif",
      ],
      mono: ["Consolas", "Monaco", "Courier New", "monospace"],
    },
    borderRadius: {
      none: "0",
      sm: "0.125rem",
      DEFAULT: "0.25rem",
      md: "0.375rem",
      lg: "0.5rem",
      xl: "0.75rem",
      "2xl": "1rem",
      "3xl": "1.5rem",
      full: "9999px",
    },
    boxShadow: {
      "legal-card":
        "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
      "legal-elevated":
        "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
      "legal-focus": "0 0 0 3px rgba(59, 130, 246, 0.1)",
    },
  },
  shortcuts: {
    // Legal document styling shortcuts
    "legal-heading": "text-legal-navy font-semibold tracking-tight",
    "legal-body": "text-legal-neutral leading-relaxed",
    "legal-card":
      "bg-white border border-gray-200 rounded-lg shadow-legal-card p-6",
    "legal-button":
      "inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md",
    "legal-button-primary":
      "legal-button bg-primary-600 text-white hover:bg-primary-700 focus:ring-2 focus:ring-primary-500",
    "legal-button-secondary":
      "legal-button bg-white text-gray-700 border-gray-300 hover:bg-gray-50 focus:ring-2 focus:ring-primary-500",
    "legal-input":
      "block w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500",
    "legal-textarea": "legal-input resize-none",

    // Citation and evidence styling
    "citation-block":
      "border-l-4 border-primary-500 bg-primary-50 p-4 my-4 rounded-r",
    "evidence-highlight":
      "bg-yellow-100 border-l-4 border-yellow-500 p-2 rounded-r",
    "statute-reference":
      "bg-purple-50 border border-purple-200 px-2 py-1 rounded text-purple-800 text-sm",

    // AI and search interface
    "ai-response":
      "bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4",
    "search-result":
      "hover:bg-gray-50 transition-colors duration-150 p-3 rounded cursor-pointer",
    "rag-context":
      "bg-green-50 border-l-4 border-green-500 p-3 text-sm text-green-800",

    // Editor and WYSIWYG
    "editor-toolbar":
      "bg-white border-b border-gray-200 p-2 flex items-center space-x-2",
    "editor-content": "prose prose-legal max-w-none p-6 focus:outline-none",

    // Layout utilities
    "sidebar-nav":
      "w-64 bg-white border-r border-gray-200 h-full overflow-y-auto",
    "main-content": "flex-1 overflow-y-auto bg-gray-50",
    "content-header": "bg-white border-b border-gray-200 px-6 py-4",
  },
  rules: [
    // Custom rules for legal document styling
    [
      /^text-case-(.+)$/,
      ([, type]) => {
        const colors = {
          contract: "#3b82f6",
          evidence: "#059669",
          statute: "#7c3aed",
          case: "#dc2626",
          regulation: "#ea580c",
        };
        return {
          color: colors[type as keyof typeof colors] || colors.contract,
        };
      },
    ],
    [
      /^bg-case-(.+)$/,
      ([, type]) => {
        const backgrounds = {
          contract: "#eff6ff",
          evidence: "#ecfdf5",
          statute: "#f3e8ff",
          case: "#fef2f2",
          regulation: "#fff7ed",
        };
        return {
          "background-color":
            backgrounds[type as keyof typeof backgrounds] ||
            backgrounds.contract,
        };
      },
    ],
  ],
  safelist: [
    // Ensure these classes are always available
    "prose",
    "prose-legal",
    "legal-card",
    "legal-button-primary",
    "legal-button-secondary",
    "citation-block",
    "evidence-highlight",
    "ai-response",
    "i-phosphor-scales",
    "i-lucide-gavel",
    "i-mdi-book-open-variant",
  ],
});
