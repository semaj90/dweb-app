// @ts-nocheck
// @ts-nocheck
import type { LayoutLoad } from "./$types";

export const load = async ({ url, fetch }: Parameters<LayoutLoad>[0]) => {
  // Initialize default user state
  let user = null;

  try {
    // Try to get user session if available
    // For now, we'll use a simple approach
    const savedUser =
      typeof window !== "undefined" ? localStorage.getItem("user") : null;
    if (savedUser) {
      user = JSON.parse(savedUser);
    }
  } catch (error) {
    console.warn("Failed to load user session:", error);
  }

  return {
    user,
    url: url.pathname,
  };
};
