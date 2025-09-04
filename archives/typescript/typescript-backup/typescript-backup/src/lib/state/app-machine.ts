/**
 * Application state machine for global app lifecycle.
 * Provides a minimal, well-typed XState machine that can be imported and used
 * by other parts of the app without causing TypeScript errors.
 */

import { createMachine, assign } from 'xstate';

type AppContext = {
    initialized: boolean;
    error: string | null;
};

type AppEvent =
    | { type: 'INITIALIZE' }
    | { type: 'START' }
    | { type: 'STOP' }
    | { type: 'RETRY' }
    | { type: 'ERROR'; message: string };

export const appMachine = createMachine({
    id: 'app',
    initial: 'idle',
    context: {
        initialized: false,
        error: null,
    },
    states: {
        idle: {
            on: {
                INITIALIZE: 'initializing',
            },
        },
        initializing: {
            invoke: {
                src: 'initializeApp', // optional service name; provide implementation when interpreting the machine
                onDone: {
                    target: 'ready',
                    actions: assign({
                        initialized: () => true,
                        error: () => null,
                    }),
                },
                onError: {
                    target: 'failed',
                    actions: assign({
                        error: (_, e: any) => (e?.data?.message ?? String(e)),
                    }),
                },
            },
        },
        ready: {
            on: {
                START: 'running',
                STOP: 'idle',
            },
        },
        running: {
            on: {
                STOP: 'ready',
                ERROR: {
                    target: 'failed',
                    actions: assign({
                        error: (_, e: any) => e.message,
                    }),
                },
            },
        },
        failed: {
            on: {
                RETRY: 'initializing',
            },
        },
    },
});

export type { AppContext, AppEvent };
