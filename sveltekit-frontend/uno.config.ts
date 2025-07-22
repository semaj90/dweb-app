import {
  defineConfig,
  presetUno,
  presetIcons,
  presetAttributify,
  presetTypography,
  presetWebFonts,
} from "unocss";
import { presetWind } from "@unocss/preset-wind";

export default defineConfig({
  presets: [
    presetUno(),
    presetWind(), // Tailwind CSS compatibility for shadcn-svelte
    presetIcons({
      scale: 1.2,
      warn: true,
      extraProperties: {
        display: "inline-block",
        "vertical-align": "middle",
      },
    }),
    presetAttributify(),
    presetTypography(),
    presetWebFonts({
      fonts: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "SF Mono",
          "Consolas",
          "monospace",
        ],
        display: ["Space Grotesk", "Inter", "sans-serif"],
      },
    }),
  ],
  theme: {
    colors: {
      // NieR: Automata Core Palette
      nier: {
        black: "#0a0a0a",
        "dark-gray": "#1a1a1a",
        gray: "#2a2a2a",
        "light-gray": "#3a3a3a",
        silver: "#c0c0c0",
        white: "#f5f5f5",
        gold: "#d4af37",
        amber: "#ffb000",
        blue: "#87ceeb",
        cyan: "#00ffff",
      },
      // Harvard Crimson Palette
      harvard: {
        crimson: "#a51c30",
        "crimson-dark": "#8b1538",
        "crimson-light": "#c5203b",
        "crimson-pale": "#f5e6e8",
      },
      // Tech/Digital Accents
      digital: {
        green: "#00ff41",
        orange: "#ff6b35",
        purple: "#9d4edd",
        blue: "#0077be",
      },
      // Status Colors
      status: {
        success: "#00ff41",
        warning: "#ffb000",
        error: "#a51c30",
        info: "#0077be",
      },
    },
    spacing: {
      xs: "0.25rem",
      sm: "0.5rem",
      md: "0.75rem",
      lg: "1rem",
      xl: "1.25rem",
      "2xl": "1.5rem",
      "3xl": "2rem",
      "4xl": "2.5rem",
      "5xl": "3rem",
      "6xl": "4rem",
    },
    borderRadius: {
      none: "0",
      sm: "0.125rem",
      DEFAULT: "0.25rem",
      md: "0.375rem",
      lg: "0.5rem",
      xl: "0.75rem",
      "2xl": "1rem",
      full: "9999px",
    },
  },
  shortcuts: {
    // NieR: Automata styled components
    "nier-card":
      'bg-nier-white dark:bg-nier-black border border-nier-light-gray dark:border-digital-green/20 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 overflow-hidden relative before:content-[""] before:absolute before:inset-0 before:bg-gradient-to-br before:from-transparent before:to-digital-green/5 before:opacity-0 hover:before:opacity-100 before:transition-opacity before:duration-300',

    "nier-button":
      "relative px-6 py-3 font-medium rounded-md transition-all duration-300 transform hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-offset-2",

    "nier-button-primary":
      "nier-button bg-nier-black text-nier-white hover:bg-nier-dark-gray focus:ring-nier-gray dark:bg-nier-white dark:text-nier-black dark:hover:bg-nier-silver",

    "nier-button-crimson":
      "nier-button bg-harvard-crimson text-nier-white hover:bg-harvard-crimson-dark focus:ring-harvard-crimson shadow-lg hover:shadow-harvard-crimson/30",

    "nier-button-gold":
      "nier-button bg-nier-gold text-nier-black hover:bg-nier-amber focus:ring-nier-gold shadow-lg hover:shadow-nier-gold/30",

    "nier-button-digital":
      "nier-button bg-digital-green text-nier-black hover:bg-digital-green/80 focus:ring-digital-green shadow-[0_0_15px_rgba(0,255,65,0.5)] hover:shadow-[0_0_25px_rgba(0,255,65,0.7)] animate-pulse",

    "nier-button-outline":
      "nier-button bg-transparent border-2 border-current hover:bg-current hover:text-nier-white dark:hover:text-nier-black",

    "nier-input":
      "w-full px-4 py-3 bg-nier-white dark:bg-nier-dark-gray border border-nier-light-gray dark:border-nier-gray rounded-lg focus:outline-none focus:ring-2 focus:ring-digital-green focus:border-transparent transition-all duration-300 placeholder-nier-light-gray dark:placeholder-nier-gray text-nier-black dark:text-nier-white",

    "nier-panel":
      "bg-nier-white/95 dark:bg-nier-black/95 backdrop-blur-lg border border-nier-light-gray dark:border-digital-green/20 rounded-xl shadow-2xl",

    "nier-glass":
      "bg-nier-white/80 dark:bg-nier-black/80 backdrop-blur-md border border-nier-light-gray/50 dark:border-digital-green/10",

    "nier-nav":
      "nier-glass sticky top-0 z-50 border-b border-nier-light-gray dark:border-digital-green/20",

    "nier-badge":
      "inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold tracking-wide",

    "nier-badge-success":
      "nier-badge bg-digital-green/10 text-digital-green border border-digital-green/20",

    "nier-badge-warning":
      "nier-badge bg-nier-amber/10 text-nier-amber border border-nier-amber/20",

    "nier-badge-error":
      "nier-badge bg-harvard-crimson/10 text-harvard-crimson border border-harvard-crimson/20",

    "nier-badge-info":
      "nier-badge bg-digital-blue/10 text-digital-blue border border-digital-blue/20",

    "nier-heading":
      "font-display font-bold tracking-tight text-nier-black dark:text-nier-white",

    "nier-link":
      "text-harvard-crimson hover:text-harvard-crimson-dark dark:text-digital-green dark:hover:text-digital-green/80 transition-colors duration-200 underline-offset-2 hover:underline",

    "nier-divider": "border-nier-light-gray dark:border-nier-gray/50",

    "nier-shadow":
      "shadow-[0_4px_20px_rgba(0,0,0,0.1)] dark:shadow-[0_4px_20px_rgba(0,255,65,0.1)]",

    "nier-glow": "shadow-[0_0_30px_rgba(0,255,65,0.3)]",

    // Legal-specific components with NieR styling
    "case-card":
      "nier-card p-6 hover:border-harvard-crimson dark:hover:border-digital-green transition-colors duration-300",

    "evidence-item":
      "bg-nier-white dark:bg-nier-dark-gray border border-nier-light-gray dark:border-nier-gray rounded-lg p-4 mb-3 hover:border-harvard-crimson dark:hover:border-digital-green transition-all duration-300 group",

    "status-badge": "nier-badge uppercase",

    "nav-item":
      'text-nier-gray dark:text-nier-silver hover:text-harvard-crimson dark:hover:text-digital-green font-medium transition-all duration-200 relative after:content-[""] after:absolute after:bottom-0 after:left-0 after:w-0 after:h-0.5 after:bg-current after:transition-all after:duration-300 hover:after:w-full',

    "nier-table": "w-full border-collapse",

    "nier-th":
      "text-left p-4 border-b-2 border-nier-light-gray dark:border-nier-gray font-semibold text-nier-black dark:text-nier-white bg-nier-white/50 dark:bg-nier-black/50",

    "nier-td": "p-4 border-b border-nier-light-gray dark:border-nier-gray/30",

    "nier-modal-overlay":
      "fixed inset-0 bg-nier-black/60 backdrop-blur-sm z-50 animate-fade-in",

    "nier-modal":
      "fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 nier-panel p-8 max-w-2xl w-full max-h-[90vh] overflow-auto z-50 animate-slide-up",

    "nier-toast": "nier-panel px-6 py-4 min-w-[300px] animate-slide-in-right",

    // AI Assistant specific
    "ai-chat-bubble": "max-w-[80%] p-4 rounded-2xl animate-fade-in",

    "ai-chat-user": "ai-chat-bubble bg-harvard-crimson text-nier-white ml-auto",

    "ai-chat-assistant":
      "ai-chat-bubble bg-nier-white dark:bg-nier-dark-gray border border-nier-light-gray dark:border-digital-green/20",

    "ai-thinking":
      "flex items-center space-x-2 text-digital-green animate-pulse",

    // Animation utilities
    "nier-transition": "transition-all duration-300 ease-in-out",

    "nier-hover-lift": "hover:transform hover:-translate-y-1 hover:shadow-xl",

    "nier-active-press": "active:transform active:scale-95",

    "nier-focus-glow":
      "focus:shadow-[0_0_20px_rgba(0,255,65,0.5)] focus:outline-none",
  },
  rules: [
    // Custom animation rules
    ["animate-fade-in", { animation: "fadeIn 0.3s ease-out" }],
    ["animate-slide-up", { animation: "slideUp 0.3s ease-out" }],
    ["animate-slide-in-right", { animation: "slideInRight 0.3s ease-out" }],
    ["animate-digital-glow", { animation: "digitalGlow 2s infinite" }],
    ["animate-crimson-pulse", { animation: "crimsonPulse 2s infinite" }],

    // Custom gradient rules
    [
      /^nier-gradient-(.+)$/,
      ([, type]) => {
        const gradients = {
          dark: "linear-gradient(135deg, #0a0a0a 0%, #2a2a2a 100%)",
          crimson: "linear-gradient(135deg, #a51c30 0%, #c5203b 100%)",
          gold: "linear-gradient(135deg, #d4af37 0%, #ffb000 100%)",
          digital: "linear-gradient(135deg, #00ff41 0%, #00ffff 100%)",
          glass:
            "linear-gradient(135deg, rgba(245,245,245,0.1) 0%, rgba(245,245,245,0.05) 100%)",
        };
        return { "background-image": gradients[type] || gradients.dark };
      },
    ],

    // Priority indicators for legal cases
    [
      /^priority-(.+)$/,
      ([, level]) => {
        const colors = {
          critical: "#a51c30",
          high: "#ff6b35",
          medium: "#ffb000",
          low: "#00ff41",
        };
        return {
          "border-left": `4px solid ${colors[level] || colors.medium}`,
          "padding-left": "1rem",
        };
      },
    ],

    // Glass morphism utilities
    [
      "glass-light",
      {
        background: "rgba(245, 245, 245, 0.7)",
        "backdrop-filter": "blur(10px)",
        "-webkit-backdrop-filter": "blur(10px)",
      },
    ],
    [
      "glass-dark",
      {
        background: "rgba(26, 26, 26, 0.7)",
        "backdrop-filter": "blur(10px)",
        "-webkit-backdrop-filter": "blur(10px)",
      },
    ],
  ],
  safelist: [
    // Commonly used dynamic classes
    ...["primary", "crimson", "gold", "digital", "outline"].map(
      (variant) => `nier-button-${variant}`
    ),
    ...["success", "warning", "error", "info"].map(
      (type) => `nier-badge-${type}`
    ),
    ...["dark", "crimson", "gold", "digital", "glass"].map(
      (type) => `nier-gradient-${type}`
    ),
    ...["critical", "high", "medium", "low"].map(
      (level) => `priority-${level}`
    ),
    "animate-fade-in",
    "animate-slide-up",
    "animate-digital-glow",
    "animate-crimson-pulse",
  ],
  // Add CSS layer for better organization
  layers: {
    "nier-base": 0,
    "nier-components": 1,
    "nier-utilities": 2,
  },
});
