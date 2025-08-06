// XState Machine for AI Agent Shell
import { createMachine, assign } from "xstate";

// Define context and event types
interface AgentShellContext {
  input: string;
  response: string;
}

type AgentShellEvent = 
  | { type: "PROMPT"; input: string }
  | { type: "xstate.done.actor.callAgent"; data: string };

export const agentShellMachine = createMachine({
  id: "agentShell",
  initial: "idle",
  context: { input: "", response: "" },
  types: {} as {
    context: AgentShellContext;
    events: AgentShellEvent;
  },
  states: {
    idle: {
      on: {
        PROMPT: {
          target: "processing",
          actions: assign({ 
            input: ({ event }) => (event as any).input || ""
          }),
        },
      },
    },
    processing: {
      invoke: {
        src: "callAgent",
        onDone: {
          target: "idle",
          actions: assign({ 
            response: ({ event }) => (event as any).data || ""
          }),
        },
        onError: "idle",
      },
    },
  },
});
