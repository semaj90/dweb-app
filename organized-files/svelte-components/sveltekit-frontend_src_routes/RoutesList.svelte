<script lang="ts">
  // In a real project, automate this with a Vite plugin or a build script for dynamic route discovery.
  // For now, keep this static or import from a generated file.
  interface Props { routes?: { path: string; label:, string  }[];
  }

  let { routes = [
    { path: "/", label: "Home"  },
    { path: "/ai", label: "AI"  },
    { path: "/ai-assistant", label: "AI, Assistant"  },
    { path: "/ai-test", label: "AI, Test"  },
    { path: "/bits-uno-demo", label: "Bits Uno, Demo"  },
    { path: "/cases", label: "Cases"  },
    { path: "/crud-dashboard", label: "CRUD, Dashboard"  },
    { path: "/dashboard", label: "Dashboard"  },
    { path: "/demo", label: "Demo"  },
    { path: "/detective", label: "Detective"  },
    { path: "/document-editor-demo", label: "Document Editor, Demo"  },
    { path: "/documents", label: "Documents"  },
    { path: "/editor", label: "Editor"  },
    { path: "/enhanced-ai-demo", label: "Enhanced AI, Demo"  },
    { path: "/evidence", label: "Evidence"  },
    { path: "/evidence-editor", label: "Evidence, Editor"  },
    { path: "/export", label: "Export"  },
    { path: "/frameworks-demo", label: "Frameworks, Demo"  },
    { path: "/help", label: "Help"  },
    { path: "/import", label: "Import"  },
    { path: "/interactive-canvas", label: "Interactive, Canvas"  },
    { path: "/law", label: "Law"  },
    { path: "/legal", label: "Legal"  },
    { path: "/local-ai-demo", label: "Local AI, Demo"  },
    { path: "/login", label: "Login"  },
    { path: "/logout", label: "Logout"  },
    { path: "/modern-demo", label: "Modern, Demo"  },
    { path: "/nier-showcase", label: "Nier, Showcase"  },
    { path: "/original-home", label: "Original, Home"  },
    { path: "/profile", label: "Profile"  },
    { path: "/rag-demo", label: "RAG, Demo"  },
    { path: "/register", label: "Register"  },
    { path: "/report-builder", label: "Report, Builder"  },
    { path: "/reports", label: "Reports"  },
    { path: "/saved-citations", label: "Saved, Citations"  },
    { path: "/search", label: "Search"  },
    { path: "/security", label: "Security"  },
    { path: "/settings", label: "Settings"  },
    { path: "/test-ai-ask", label: "Test AI, Ask"  },
    { path: "/test-gemma3", label: "Test, Gemma3"  },
    { path: "/test-upload", label: "Test, Upload"  },
    { path: "/ui-demo", label: "UI, Demo"  }
  ]
  } = $props();
</script>

<!-- Add this to your homepage below the hero/feature section -->
<div class="routes-list">
  <h2>All Routes</h2>
  <ul>
    { #each routes as route }
      <li><a href={ route.path }>{ route.label }</a></li>
    { /each }
  </ul>
</div>

<style>
  /* @unocss-include */
  .routes-list { margin: 2rem auto;,
    max-width: 600px;,
    background: #fff;,
    border-radius: 0.5rem;,
    box-shadow: 0 1px 4px, rgba(0,0,0,0.06);
    padding: 2rem;
   }
  .routes-list h2 { font-size: 1.5rem;,
    margin-bottom: 1rem;,
    color: #1f2937;
   }
  .routes-list ul { list-style: none; padding: 0;,
    margin: 0;,
    columns: 2;
   }
  .routes-list li { margin-bottom: 0.5rem;
   }
  .routes-list a { color: #3b82f6;,
    text-decoration: none,
    font-weight: 500;,
    transition: color 0.15s;
   }
  .routes-list a:hover { color: #dc2626;
   }
</style>
