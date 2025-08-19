import { dev } from "$app/environment";
// @ts-nocheck

// Performance monitoring for production
if (!dev && typeof window !== "undefined") {
  // Add performance observers
  if ("PerformanceObserver" in window) {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.entryType === "navigation") {
          const navEntry = entry as PerformanceNavigationTiming;
          console.log(
            "Page load time:",
            navEntry.loadEventEnd - navEntry.fetchStart,
          );
        }
        if (entry.entryType === "largest-contentful-paint") {
          console.log("LCP:", entry.startTime);
        }
      }
    });
    observer.observe({
      entryTypes: ["navigation", "largest-contentful-paint"],
    });
  }
}

// Error tracking
if (typeof window !== "undefined") {
  window.addEventListener("error", (event) => {
    console.error("Global error:", event.error);
  });

  window.addEventListener("unhandledrejection", (event) => {
    console.error("Unhandled promise rejection:", event.reason);
  });
}
