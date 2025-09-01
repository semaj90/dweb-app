
import { setup, createActor, assign, fromPromise } from "xstate";
import type {
  CanvasNode,
  CanvasConnection,
  InteractiveCanvasState,
} from "$lib/types/canvas";

export interface CanvasContext {
  nodes: CanvasNode[];
  connections: CanvasConnection[];
  selectedNode: string | null;
  draggedNode: string | null;
  canvasState: InteractiveCanvasState;
  error: string | null;
}

type CanvasEvent =
  | { type: "ADD_NODE"; node: CanvasNode }
  | { type: "REMOVE_NODE"; nodeId: string }
  | { type: "UPDATE_NODE"; nodeId: string; updates: Partial<CanvasNode> }
  | { type: "SELECT_NODE"; nodeId: string | null }
  | { type: "START_DRAG"; nodeId: string }
  | { type: "END_DRAG" }
  | { type: "ADD_CONNECTION"; connection: CanvasConnection }
  | { type: "REMOVE_CONNECTION"; connectionId: string }
  | { type: "SAVE_STATE" }
  | { type: "LOAD_STATE"; state: InteractiveCanvasState }
  | { type: "CLEAR_CANVAS" }
  | { type: "ERROR"; error: string }
  | { type: "CLEAR_ERROR" };

export const canvasSystemMachine = setup({
  types: {
    context: {} as CanvasContext,
    events: {} as CanvasEvent,
  },
  actions: {
    addNode: assign({
      nodes: ({ context, event }) => {
        if (event.type !== "ADD_NODE") return context.nodes;
        return [...context.nodes, event.node];
      },
    }),
    removeNode: assign({
      nodes: ({ context, event }) => {
        if (event.type !== "REMOVE_NODE") return context.nodes;
        return context.nodes.filter((node) => node.id !== event.nodeId);
      },
      connections: ({ context, event }) => {
        if (event.type !== "REMOVE_NODE") return context.connections;
        return context.connections.filter(
          (conn) =>
            conn.sourceId !== event.nodeId && conn.targetId !== event.nodeId,
        );
      },
    }),
    updateNode: assign({
      nodes: ({ context, event }) => {
        if (event.type !== "UPDATE_NODE") return context.nodes;
        return context.nodes.map((node) =>
          node.id === event.nodeId ? { ...node, ...event.updates } : node,
        );
      },
    }),
    selectNode: assign({
      selectedNode: ({ event }) => {
        if (event.type !== "SELECT_NODE") return null;
        return event.nodeId;
      },
    }),
    startDrag: assign({
      draggedNode: ({ event }) => {
        if (event.type !== "START_DRAG") return null;
        return event.nodeId;
      },
    }),
    endDrag: assign({
      draggedNode: () => null,
    }),
    addConnection: assign({
      connections: ({ context, event }) => {
        if (event.type !== "ADD_CONNECTION") return context.connections;
        return [...context.connections, event.connection];
      },
    }),
    removeConnection: assign({
      connections: ({ context, event }) => {
        if (event.type !== "REMOVE_CONNECTION") return context.connections;
        return context.connections.filter(
          (conn) => conn.id !== event.connectionId,
        );
      },
    }),
    clearCanvas: assign({
      nodes: () => [],
      connections: () => [],
      selectedNode: () => null,
      draggedNode: () => null,
    }),
    setError: assign({
      error: ({ event }) => {
        if (event.type !== "ERROR") return null;
        return event.error;
      },
    }),
    clearError: assign({
      error: () => null,
    }),
  },
  actors: {
    saveState: fromPromise(async ({ input }: { input: CanvasContext }) => {
      const state: InteractiveCanvasState = {
        nodes: input.nodes,
        connections: input.connections,
        viewport: { x: 0, y: 0, zoom: 1 },
      };

      // Save to localStorage or API
      if (typeof window !== "undefined") {
        localStorage.setItem("canvasState", JSON.stringify(state));
      }

      return state;
    }),
  },
}).createMachine({
  id: "canvasSystem",
  initial: "idle",
  context: {
    nodes: [],
    connections: [],
    selectedNode: null,
    draggedNode: null,
    canvasState: {
      nodes: [],
      connections: [],
      viewport: { x: 0, y: 0, zoom: 1 },
    },
    error: null,
  },
  states: {
    idle: {
      on: {
        ADD_NODE: {
          actions: ["addNode"],
        },
        REMOVE_NODE: {
          actions: ["removeNode"],
        },
        UPDATE_NODE: {
          actions: ["updateNode"],
        },
        SELECT_NODE: {
          actions: ["selectNode"],
        },
        START_DRAG: {
          target: "dragging",
          actions: ["startDrag"],
        },
        ADD_CONNECTION: {
          actions: ["addConnection"],
        },
        REMOVE_CONNECTION: {
          actions: ["removeConnection"],
        },
        SAVE_STATE: {
          target: "saving",
        },
        LOAD_STATE: {
          actions: assign({
            canvasState: ({ event }) => {
              if (event.type !== "LOAD_STATE") {
                return {
                  nodes: [],
                  connections: [],
                  viewport: { x: 0, y: 0, zoom: 1 },
                };
              }
              return event.state;
            },
          }),
        },
        CLEAR_CANVAS: {
          actions: ["clearCanvas"],
        },
        ERROR: {
          actions: ["setError"],
        },
        CLEAR_ERROR: {
          actions: ["clearError"],
        },
      },
    },
    dragging: {
      on: {
        END_DRAG: {
          target: "idle",
          actions: ["endDrag"],
        },
        UPDATE_NODE: {
          actions: ["updateNode"],
        },
      },
    },
    saving: {
      invoke: {
        src: "saveState",
        input: ({ context }) => context,
        onDone: {
          target: "idle",
          actions: assign({
            canvasState: ({ event }) => event.output,
          }),
        },
        onError: {
          target: "idle",
          actions: ["setError"],
        },
      },
    },
  },
});

export const createCanvasActor = () => createActor(canvasSystemMachine);
;