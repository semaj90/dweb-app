// Agent Shell State Machine
// Manages the state of the agent-based system

import { createMachine, assign } from 'xstate';

export const agentShellMachine = createMachine({
  id: 'agentShell',
  initial: 'idle',
  context: {
    agents: [],
    tasks: [],
    currentTask: null,
    results: [],
    errors: [],
    status: 'idle'
  },
  states: {
    idle: {
      on: {
        START: 'initializing',
        ADD_TASK: {
          actions: assign({
            tasks: (context, event) => [...context.tasks, event.task]
          })
        }
      }
    },
    initializing: {
      invoke: {
        src: 'initializeAgents',
        onDone: {
          target: 'ready',
          actions: assign({
            agents: (context, event) => event.data,
            status: 'ready'
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            errors: (context, event) => [...context.errors, event.data],
            status: 'error'
          })
        }
      }
    },
    ready: {
      on: {
        PROCESS_TASK: 'processing',
        ADD_TASK: {
          actions: assign({
            tasks: (context, event) => [...context.tasks, event.task]
          })
        },
        STOP: 'stopping'
      }
    },
    processing: {
      entry: assign({
        currentTask: (context) => context.tasks[0],
        status: 'processing'
      }),
      invoke: {
        src: 'processTask',
        onDone: {
          target: 'taskComplete',
          actions: assign({
            results: (context, event) => [...context.results, event.data],
            tasks: (context) => context.tasks.slice(1)
          })
        },
        onError: {
          target: 'taskError',
          actions: assign({
            errors: (context, event) => [...context.errors, event.data]
          })
        }
      }
    },
    taskComplete: {
      always: [
        {
          target: 'processing',
          cond: (context) => context.tasks.length > 0
        },
        {
          target: 'ready'
        }
      ]
    },
    taskError: {
      on: {
        RETRY: 'processing',
        SKIP: {
          target: 'taskComplete',
          actions: assign({
            tasks: (context) => context.tasks.slice(1)
          })
        },
        STOP: 'stopping'
      }
    },
    stopping: {
      invoke: {
        src: 'cleanupAgents',
        onDone: 'stopped',
        onError: 'stopped'
      }
    },
    stopped: {
      type: 'final'
    },
    error: {
      on: {
        RETRY: 'initializing',
        RESET: 'idle'
      }
    }
  }
}, {
  services: {
    initializeAgents: async () => {
      // Initialize agents
      console.log('Initializing agents...');
      return [
        { id: 'agent1', type: 'analyzer', status: 'ready' },
        { id: 'agent2', type: 'processor', status: 'ready' },
        { id: 'agent3', type: 'validator', status: 'ready' }
      ];
    },
    processTask: async (context) => {
      // Process current task
      console.log('Processing task:', context.currentTask);
      return {
        taskId: context.currentTask?.id,
        result: 'Task completed successfully',
        timestamp: new Date().toISOString()
      };
    },
    cleanupAgents: async () => {
      // Cleanup agents
      console.log('Cleaning up agents...');
      return true;
    }
  }
});

export default agentShellMachine;