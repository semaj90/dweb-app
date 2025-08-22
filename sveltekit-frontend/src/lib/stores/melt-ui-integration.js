import { writable } from "svelte/store";

/**
 * Melt UI Integration - Fixed for Phase 2
 */

// UI Update Manager for real-time changes
export const uiUpdateManager = {
  updates: writable([]),

  queueUpdate(config) {
    this.updates.update((updates) => [...updates, config]);
    this.processUpdates();
  },

  processUpdates() {
    this.updates.subscribe((updates) => {
      updates.forEach((update) => {
        const elements = document.querySelectorAll(update.selector);
        elements.forEach((el) => {
          if (update.classes?.add) el.classList.add(...update.classes.add);
          if (update.classes?.remove)
            el.classList.remove(...update.classes.remove);
          if (update.attributes) {
            Object.entries(update.attributes).forEach(([key, value]) => {
              el.setAttribute(key, value);
            });
          }
        });
      });
      this.updates.set([]); // Clear processed updates
    });
  },
};

// Yorha Class Manager for legacy support
export const YorhaClassManager = {
  applyYorhaTheme(element, theme = "enhanced") {
    const themeClasses = {
      enhanced: ["yorha-enhanced", "yorha-glow", "yorha-shadow"],
      terminal: ["yorha-terminal", "yorha-mono", "yorha-green"],
      classic: ["yorha-classic", "yorha-border"],
    };

    const classes = themeClasses[theme] || themeClasses.enhanced;
    if (element?.classList?.add) {
      element.classList.add(...classes);
    }
  },
};

// Basic button implementation when Melt UI unavailable
function createBasicButton(options = {}) {
  return {
    elements: {
      root: {
        "data-melt-button": "",
        class: `btn btn-${options.variant || "default"}`,
        type: "button",
      },
    },
    states: {
      pressed: writable(false),
    },
  };
}

// Enhanced button creation with fallback
export async function createEnhancedButton(options = {}) {
  try {
    const meltUI = await import("melt");
    const { createButton } = meltUI;

    const button = createButton(options);
    return {
      ...button,
      enhanced: true,
      aiControlled: options.aiControlled || false,
    };
  } catch (error) {
    console.warn("Melt UI not available, using fallback");
    return createBasicButton(options);
  }
}

// Prop merging for Bits UI v2 compatibility
export function mergeBitsUIProps(meltProps, bitsProps, aiProps = {}) {
  return {
    ...meltProps,
    ...bitsProps,
    class: [
      meltProps?.class,
      bitsProps?.class,
      ...(aiProps.aiClasses || []),
      aiProps.yorhaClass,
    ]
      .filter(Boolean)
      .join(" "),
    "data-ai-enhanced": aiProps.aiClasses ? "true" : undefined,
  };
}
