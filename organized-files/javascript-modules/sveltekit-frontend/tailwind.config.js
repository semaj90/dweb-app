import typography from "@tailwindcss/typography";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{html,js,svelte,ts}"],
  theme: {
    extend: {
      colors: {
        "yorha-primary": "#d4af37",
        "yorha-secondary": "#c9aa71",
        "yorha-accent": "#48cae4",
        "yorha-bg-primary": "#1a1a1a",
        "yorha-bg-secondary": "#2d2d2d",
        "yorha-bg-tertiary": "#3a3a3a",
        "yorha-text-primary": "#f5f5f5",
        "yorha-text-secondary": "#cccccc",
        "yorha-text-muted": "#999999",
        "yorha-border": "#444444",
        "yorha-success": "#4ade80",
        "yorha-warning": "#fbbf24",
        "yorha-error": "#f87171",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "MS Gothic", "monospace"],
      },
      animation: {
        "scan-line": "scan-line 3s linear infinite",
        float: "float 6s linear infinite",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        glow: "glow 2s ease-in-out infinite alternate",
      },
      keyframes: {
        "scan-line": {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0) rotate(0deg)", opacity: "0" },
          "10%": { opacity: "1" },
          "90%": { opacity: "1" },
          "100%": {
            transform: "translateY(-100%) rotate(360deg)",
            opacity: "0",
          },
        },
        glow: {
          from: { boxShadow: "0 0 20px rgba(212, 175, 55, 0.2)" },
          to: { boxShadow: "0 0 30px rgba(212, 175, 55, 0.4)" },
        },
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [typography],
};
