import crypto from "crypto";
// Advanced Auto-Memory Store with 4D Search and Predictive Analytics
import { createMachine, assign } from "xstate";
import Fuse from "fuse.js";

// Auto-Memory State Machine
const autoMemoryMachine = createMachine({
  id: "autoMemory",
  initial: "idle",
  context: {
    memories: [],
    patterns: {},
    predictions: [],
    loading: false,
    error: null,
  },
  states: {
    idle: {
      on: {
        STORE_INTERACTION: "storing",
        SEARCH_4D: "searching",
        PREDICT_INTENT: "predicting",
      },
    },
    storing: {
      entry: assign({ loading: true }),
      invoke: {
        src: "storeInteraction",
        onDone: {
          target: "idle",
          actions: assign({
            memories: ({ context, event }) => [...context.memories, event.data],
            loading: false,
          }),
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => event.data,
            loading: false,
          }),
        },
      },
    },
    searching: {
      entry: assign({ loading: true }),
      invoke: {
        src: "search4D",
        onDone: {
          target: "idle",
          actions: assign({
            memories: ({ event }) => event.data.results,
            loading: false,
          }),
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => event.data,
            loading: false,
          }),
        },
      },
    },
    predicting: {
      entry: assign({ loading: true }),
      invoke: {
        src: "predictIntent",
        onDone: {
          target: "idle",
          actions: assign({
            predictions: ({ event }) => event.data.predictions,
            loading: false,
          }),
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => event.data,
            loading: false,
          }),
        },
      },
    },
    error: {
      on: {
        RETRY: "idle",
        CLEAR_ERROR: {
          target: "idle",
          actions: assign({ error: null }),
        },
      },
    },
  },
});

// Advanced Auto-Memory Store with Svelte 5 Runes
function createAutoMemoryStore() {
  const localMemories = $state([]);
  const userPatterns = $state({});
  const predictions = $state([]);
  const searchCache = $state(new Map());
  const connectionStatus = $state("disconnected");

  let fuseIndex = null;
  let ws = $state(null);

  const memoryStats = $derived({
    totalMemories: localMemories.length,
    uniqueTypes: [...new Set(localMemories.map((m) => m.interaction_type))]
      .length,
    recentMemories: localMemories.filter((m) => {
      const dayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
      return new Date(m.created_at) > dayAgo;
    }).length,
  });

  function updateFuseIndex() {
    const fuseOptions = {
      keys: ["content", "interaction_type"],
      threshold: 0.3,
      includeScore: true,
    };
    fuseIndex = new Fuse(localMemories, fuseOptions);
  }

  function connect() {
    try {
      ws = new WebSocket("ws://localhost:8001/ws/memory-stream/user_001");

      ws.onopen = () => {
        connectionStatus = "connected";
        console.log("✅ Auto-Memory connected");
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleRealtimeUpdate(data);
      };

      ws.onclose = () => {
        connectionStatus = "disconnected";
        setTimeout(connect, 3000);
      };
    } catch (error) {
      console.error("Failed to connect:", error);
    }
  }

  function handleRealtimeUpdate(data) {
    if (data.type === "analytics_update") {
      Object.assign(userPatterns, data.data.patterns || {});
    }
  }

  async function storeInteraction(interaction) {
    try {
      const enhancedInteraction = {
        ...interaction,
        temporal_context: {
          timestamp: new Date().toISOString(),
          hour: new Date().getHours(),
          day_of_week: new Date().getDay(),
        },
      };

      const response = await fetch("http://localhost:8001/store-interaction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(enhancedInteraction),
      });

      if (!response.ok) throw new Error("Store failed");

      const result = await response.json();
      localMemories.push({
        id: result.memory_id,
        ...enhancedInteraction,
        created_at: new Date().toISOString(),
      });

      updateFuseIndex();
      return result;
    } catch (error) {
      console.error("Store interaction failed:", error);
      throw error;
    }
  }

  async function search4D(query, options = {}) {
    try {
      const searchQuery = {
        user_id: "user_001",
        query,
        limit: options.limit || 10,
      };

      const response = await fetch("http://localhost:8001/search-4d", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(searchQuery),
      });

      if (response.ok) {
        return await response.json();
      } else {
        throw new Error("Remote search failed");
      }
    } catch (error) {
      // Fallback to local search
      if (fuseIndex) {
        const results = fuseIndex.search(query);
        return {
          results: results.map((r) => ({
            memory_id: r.item.id || crypto.randomUUID(),
            content: r.item.content,
            similarity_score: 1 - r.score,
            created_at: r.item.created_at,
          })),
          count: results.length,
          search_type: "local_fallback",
        };
      }
      return { results: [], count: 0 };
    }
  }

  function smartSearch(query) {
    if (!fuseIndex) return { results: [], suggestions: [] };

    const results = fuseIndex.search(query);
    return {
      results: results.map((r) => r.item),
      suggestions: [], // Simplified for now
    };
  }

  return {
    get memories() {
      return localMemories;
    },
    get patterns() {
      return userPatterns;
    },
    get predictions() {
      return predictions;
    },
    get stats() {
      return memoryStats;
    },
    get connectionStatus() {
      return connectionStatus;
    },

    connect,
    storeInteraction,
    search4D,
    smartSearch,

    initialize: () => {
      updateFuseIndex();
      connect();
    },

    disconnect: () => {
      if (ws) {
        ws.close();
        ws = null;
      }
    },
  };
}

const autoMemoryServices = {
  storeInteraction: async (context, event) => {
    const response = await fetch("http://localhost:8001/store-interaction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event.interaction),
    });

    if (!response.ok) throw new Error("Failed to store interaction");
    return await response.json();
  },

  search4D: async (context, event) => {
    const response = await fetch("http://localhost:8001/search-4d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event.query),
    });

    if (!response.ok) throw new Error("4D search failed");
    return await response.json();
  },

  predictIntent: async (context, event) => {
    const response = await fetch("http://localhost:8001/predict-intent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event.request),
    });

    if (!response.ok) throw new Error("Intent prediction failed");
    return await response.json();
  },
};

export { createAutoMemoryStore, autoMemoryMachine, autoMemoryServices };
