// uno.config.ts or a file imported by it (e.g., src/lib/uno-shortcuts.ts)

// By exporting shortcuts as an array of arrays (or tuples),
// UnoCSS can correctly compose them. It processes the array in order,
// so 'btn' will be expanded when used in 'btn-primary'.

export const shortcuts = [
  // Base component styles
  [
    "btn",
    "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  ],
  ["card-base", "rounded-lg border bg-card text-card-foreground shadow-sm"],
  [
    "form-input",
    "flex h-10 w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
  ],
  ["form-label", "block text-sm font-medium text-foreground mb-1.5"],

  // Layout
  ["container", "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"],
  ["section", "py-12 md:py-16"],

  // Button variants that build upon the 'btn' base
  // Note: These now correctly reference the 'btn' shortcut defined above.
  [
    "btn-primary",
    "btn px-4 py-2 bg-primary text-primary-foreground hover:bg-primary/90",
  ],
  [
    "btn-secondary",
    "btn px-4 py-2 bg-secondary text-secondary-foreground hover:bg-secondary/80",
  ],
  [
    "btn-destructive",
    "btn px-4 py-2 bg-destructive text-destructive-foreground hover:bg-destructive/90",
  ],
  ["btn-ghost", "btn px-4 py-2 hover:bg-accent hover:text-accent-foreground"],
  ["btn-link", "btn text-primary underline-offset-4 hover:underline"],

  // Card variants
  [
    "card-interactive",
    "card-base transition-all duration-300 hover:shadow-lg hover:border-primary",
  ],

  // Nier-specific theme shortcuts
  ["nier-text-glow", "text-shadow-lg shadow-primary"],
  [
    "nier-border",
    "border-2 border-border relative before:content-[''] before:absolute before:-inset-0.5 before:bg-gradient-to-r before:from-primary before:via-transparent before:to-accent before:z-[-1] before:opacity-30",
  ],
];
