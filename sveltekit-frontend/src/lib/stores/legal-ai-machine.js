import crypto from "crypto";
// ================================================================================
// XSTATE INTEGRATION WITH SVELTE 5 & GPU WORKER
// ================================================================================
// State Machines • GPU Threading • WebSocket • Service Worker Integration
// ================================================================================

import { createMachine, assign, interpret, spawn } from "xstate";
// Orphaned content: import {

import { writable, derived, get } from "svelte/store";
// Orphaned content: import {

// ============================================================================
// GPU SERVICE WORKER INTEGRATION
// ============================================================================

class GPUServiceWorkerClient {
    constructor() {
        this.worker = null;
        this.messageQueue = new Map();
        this.messageId = 0;
        this.connected = writable(false);
        this.status = writable('disconnected');
    }

    async initialize() {
        if (!browser || !('serviceWorker' in navigator)) {
            console.warn('Service Worker not supported');
            return false;
        }

        try {
            // Register service worker
            const registration = await navigator.serviceWorker.register('/service-worker.js', {
                scope: '/'
            });

            console.log('✅ Service Worker registered:', registration.scope);

            // Wait for active worker
            await this.waitForActiveWorker(registration);

            this.worker = registration.active;
            this.connected.set(true);
            this.status.set('connected');

            return true;
        } catch (error) {
            console.error('❌ Service Worker registration failed:', error);
            this.status.set('error');
            return false;
        }
    }

    async waitForActiveWorker(registration) {
        return new Promise((resolve) => {
            if (registration.active) {
                resolve();
                return;
            }

            const checkWorker = () => {
                if (registration.active) {
                    resolve();
                } else {
                    setTimeout(checkWorker, 100);
                }
            };
            checkWorker();
        });
    }

    async sendMessage(type, data) {
        if (!this.worker) {
            throw new Error('Service Worker not available');
        }

        const id = ++this.messageId;
        const messageChannel = new MessageChannel();

        return new Promise((resolve, reject) => {
            messageChannel.port1.onmessage = (event) => {
                const { success, result, error } = event.data;
                if (success) {
                    resolve(result);
                } else {
                    reject(new Error(error));
                }
            };

            this.worker.postMessage({
                type,
                data,
                id
            }, [messageChannel.port2]);

            // Timeout after 30 seconds
            setTimeout(() => {
                reject(new Error('GPU operation timeout'));
            }, 30000);
        });
    }

    async executeVectorSimilarity(vectorA, vectorB) {
        return this.sendMessage('GPU_VECTOR_SIMILARITY', { vectorA, vectorB });
    }

    async executeKMeans(dataPoints, k, maxIterations = 100) {
        return this.sendMessage('GPU_KMEANS', { dataPoints, k, maxIterations });
    }

    async computeEmbedding(tokens) {
        return this.sendMessage('GPU_EMBEDDING', { tokens });
    }

    async createVertexBuffer(id, data, usage = 'STATIC_DRAW') {
        return this.sendMessage('GPU_CREATE_BUFFER', { id, data, usage });
    }

    async getStatus() {
        return this.sendMessage('GPU_STATUS', {});
    }
}

// ============================================================================
// ENHANCED LEGAL AI STATE MACHINE
// ============================================================================

const legalAIConfig = {
    id: 'legalAI',
    initial: 'idle',
    context: {
        // User session
        userId: null,
        sessionId: null,
        
        // AI & Processing
        currentQuery: '',
        searchResults: [],
        chatHistory: [],
        documentAnalysis: null,
        
        // GPU & Performance
        gpuStatus: 'unknown',
        processingQueue: [],
        computeResults: new Map(),
        
        // WebSocket & Connectivity
        wsConnected: false,
        lastHeartbeat: null,
        connectionRetries: 0,
        
        // Error handling
        errors: [],
        retryCount: 0,
        
        // UI State
        activeTab: 'chat',
        sidebarOpen: true,
        notifications: [],
        
        // Legal specific
        activeCases: [],
        currentCase: null,
        evidenceUploads: [],
        precedentSearch: null
    },
    
    states: {
        idle: {
            entry: ['initializeSession', 'connectWebSocket'],
            on: {
                START_SEARCH: 'searching',
                START_CHAT: 'chatting',
                UPLOAD_DOCUMENT: 'uploadingDocument',
                ANALYZE_CASE: 'analyzingCase',
                CONNECT_GPU: 'initializingGPU',
                ERROR: 'error'
            }
        },
        
        initializingGPU: {
            entry: ['initializeGPUWorker'],
            invoke: {
                src: 'initializeGPU',
                onDone: {
                    target: 'idle',
                    actions: ['setGPUStatus']
                },
                onError: {
                    target: 'error',
                    actions: ['setError']
                }
            }
        },
        
        searching: {
            entry: ['startSearch'],
            invoke: {
                src: 'performSearch',
                onDone: {
                    target: 'idle',
                    actions: ['setSearchResults']
                },
                onError: {
                    target: 'error',
                    actions: ['setError']
                }
            },
            on: {
                CANCEL_SEARCH: 'idle',
                UPDATE_QUERY: {
                    actions: ['updateQuery']
                }
            }
        },
        
        chatting: {
            entry: ['startChat'],
            states: {
                idle: {
                    on: {
                        SEND_MESSAGE: 'processing',
                        CLEAR_CHAT: {
                            actions: ['clearChatHistory']
                        }
                    }
                },
                processing: {
                    entry: ['addUserMessage'],
                    invoke: {
                        src: 'processChat',
                        onDone: {
                            target: 'idle',
                            actions: ['addAIResponse']
                        },
                        onError: {
                            target: 'idle',
                            actions: ['setError']
                        }
                    }
                }
            },
            initial: 'idle',
            on: {
                STOP_CHAT: 'idle'
            }
        },
        
        uploadingDocument: {
            entry: ['startUpload'],
            invoke: {
                src: 'uploadDocument',
                onDone: {
                    target: 'analyzingDocument',
                    actions: ['setUploadResult']
                },
                onError: {
                    target: 'error',
                    actions: ['setError']
                }
            },
            on: {
                CANCEL_UPLOAD: 'idle'
            }
        },
        
        analyzingDocument: {
            entry: ['startDocumentAnalysis'],
            invoke: {
                src: 'analyzeDocument',
                onDone: {
                    target: 'idle',
                    actions: ['setDocumentAnalysis']
                },
                onError: {
                    target: 'error',
                    actions: ['setError']
                }
            }
        },
        
        analyzingCase: {
            entry: ['startCaseAnalysis'],
            states: {
                gatheringEvidence: {
                    invoke: {
                        src: 'gatherEvidence',
                        onDone: 'computingSimilarity'
                    }
                },
                computingSimilarity: {
                    invoke: {
                        src: 'computeVectorSimilarity',
                        onDone: 'searchingPrecedents'
                    }
                },
                searchingPrecedents: {
                    invoke: {
                        src: 'searchPrecedents',
                        onDone: 'generatingAnalysis'
                    }
                },
                generatingAnalysis: {
                    invoke: {
                        src: 'generateCaseAnalysis',
                        onDone: {
                            target: '#legalAI.idle',
                            actions: ['setCaseAnalysis']
                        }
                    }
                }
            },
            initial: 'gatheringEvidence'
        },
        
        error: {
            entry: ['logError'],
            on: {
                RETRY: [
                    {
                        target: 'idle',
                        cond: 'canRetry',
                        actions: ['incrementRetry']
                    },
                    {
                        target: 'criticalError'
                    }
                ],
                CLEAR_ERROR: 'idle'
            }
        },
        
        criticalError: {
            entry: ['handleCriticalError'],
            type: 'final'
        }
    }
};

// ============================================================================
// STATE MACHINE ACTIONS
// ============================================================================

const legalAIActions = {
    initializeSession: assign({
        sessionId: () => crypto.randomUUID(),
        userId: (context) => context.userId || `user_${Date.now()}`
    }),
    
    connectWebSocket: (context) => {
        // WebSocket connection will be handled by the WebSocket service
        console.log('Connecting WebSocket for session:', context.sessionId);
    },
    
    initializeGPUWorker: (context) => {
        console.log('Initializing GPU Worker...');
    },
    
    setGPUStatus: assign({
        gpuStatus: (_, event) => event.data.status
    }),
    
    startSearch: assign({
        currentQuery: (_, event) => event.query || ''
    }),
    
    setSearchResults: assign({
        searchResults: (_, event) => event.data.results || []
    }),
    
    updateQuery: assign({
        currentQuery: (_, event) => event.query
    }),
    
    startChat: (context) => {
        console.log('Starting chat session:', context.sessionId);
    },
    
    addUserMessage: assign({
        chatHistory: (context, event) => [
            ...context.chatHistory,
            {
                id: crypto.randomUUID(),
                role: 'user',
                content: event.message,
                timestamp: new Date(),
                sessionId: context.sessionId
            }
        ]
    }),
    
    addAIResponse: assign({
        chatHistory: (context, event) => [
            ...context.chatHistory,
            {
                id: crypto.randomUUID(),
                role: 'assistant',
                content: event.data.response,
                timestamp: new Date(),
                sessionId: context.sessionId,
                metadata: event.data.metadata
            }
        ]
    }),
    
    clearChatHistory: assign({
        chatHistory: []
    }),
    
    startUpload: (context) => {
        console.log('Starting document upload...');
    },
    
    setUploadResult: assign({
        evidenceUploads: (context, event) => [
            ...context.evidenceUploads,
            event.data
        ]
    }),
    
    startDocumentAnalysis: (context) => {
        console.log('Starting document analysis...');
    },
    
    setDocumentAnalysis: assign({
        documentAnalysis: (_, event) => event.data
    }),
    
    startCaseAnalysis: (context) => {
        console.log('Starting case analysis...');
    },
    
    setCaseAnalysis: assign({
        currentCase: (_, event) => ({
            ...event.data,
            analysisTimestamp: new Date()
        })
    }),
    
    setError: assign({
        errors: (context, event) => [
            ...context.errors,
            {
                id: crypto.randomUUID(),
                message: event.data?.message || 'Unknown error',
                timestamp: new Date(),
                context: event.data?.context
            }
        ]
    }),
    
    incrementRetry: assign({
        retryCount: (context) => context.retryCount + 1
    }),
    
    logError: (context, event) => {
        console.error('State machine error:', event.data);
    },
    
    handleCriticalError: (context) => {
        console.error('Critical error - system halted');
        // Send error report to monitoring service
    }
};

// ============================================================================
// STATE MACHINE SERVICES
// ============================================================================

const legalAIServices = {
    initializeGPU: async () => {
        const gpuClient = new GPUServiceWorkerClient();
        const success = await gpuClient.initialize();
        
        if (success) {
            const status = await gpuClient.getStatus();
            return { status: 'initialized', ...status };
        } else {
            throw new Error('GPU initialization failed');
        }
    },
    
    performSearch: async (context) => {
        const response = await fetch('/api/rag/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: context.currentQuery,
                sessionId: context.sessionId,
                options: {
                    includeVectorSimilarity: true,
                    includePrecedents: true
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }
        
        return response.json();
    },
    
    processChat: async (context, event) => {
        // Send to GPU for embedding computation
        const gpuClient = new GPUServiceWorkerClient();
        const tokens = event.message.split(' ');
        const embedding = await gpuClient.computeEmbedding(tokens);
        
        // Send to AI service
        const response = await fetch('/api/rag/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: event.message,
                context: context.chatHistory.slice(-10),
                sessionId: context.sessionId,
                embedding: Array.from(embedding)
            })
        });
        
        if (!response.ok) {
            throw new Error(`Chat failed: ${response.statusText}`);
        }
        
        return response.json();
    },
    
    uploadDocument: async (context, event) => {
        const formData = new FormData();
        formData.append('file', event.file);
        formData.append('caseId', context.currentCase?.id || 'general');
        formData.append('documentType', event.documentType || 'evidence');
        formData.append('sessionId', context.sessionId);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }
        
        return response.json();
    },
    
    analyzeDocument: async (context, event) => {
        const response = await fetch('/api/documents/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                documentId: event.documentId,
                analysisType: 'comprehensive',
                sessionId: context.sessionId
            })
        });
        
        if (!response.ok) {
            throw new Error(`Analysis failed: ${response.statusText}`);
        }
        
        return response.json();
    },
    
    gatherEvidence: async (context) => {
        // Gather all evidence for the case
        const evidence = context.evidenceUploads.filter(
            upload => upload.caseId === context.currentCase?.id
        );
        
        return { evidence };
    },
    
    computeVectorSimilarity: async (context, event) => {
        const gpuClient = new GPUServiceWorkerClient();
        
        // Compute similarity between evidence pieces
        const similarities = [];
        const evidence = event.data.evidence;
        
        for (let i = 0; i < evidence.length; i++) {
            for (let j = i + 1; j < evidence.length; j++) {
                const similarity = await gpuClient.executeVectorSimilarity(
                    evidence[i].embedding,
                    evidence[j].embedding
                );
                
                similarities.push({
                    doc1: evidence[i].id,
                    doc2: evidence[j].id,
                    similarity
                });
            }
        }
        
        return { similarities };
    },
    
    searchPrecedents: async (context, event) => {
        const response = await fetch('/api/legal/precedents', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                caseType: context.currentCase?.type,
                keywords: context.currentCase?.keywords,
                jurisdiction: context.currentCase?.jurisdiction,
                similarities: event.data.similarities
            })
        });
        
        return response.json();
    },
    
    generateCaseAnalysis: async (context, event) => {
        const response = await fetch('/api/legal/analyze-case', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                caseId: context.currentCase?.id,
                evidence: event.data.evidence,
                precedents: event.data.precedents,
                similarities: event.data.similarities,
                sessionId: context.sessionId
            })
        });
        
        return response.json();
    }
};

// ============================================================================
// STATE MACHINE GUARDS
// ============================================================================

const legalAIGuards = {
    canRetry: (context) => context.retryCount < 3,
    
    hasGPUSupport: () => {
        return browser && 'serviceWorker' in navigator && 'WebGL2RenderingContext' in window;
    },
    
    isConnected: (context) => context.wsConnected,
    
    hasActiveCase: (context) => context.currentCase !== null
};

// ============================================================================
// MACHINE CREATION & STORE INTEGRATION
// ============================================================================

export function createLegalAIMachine() {
    return createMachine(legalAIConfig, {
        actions: legalAIActions,
        services: legalAIServices,
        guards: legalAIGuards
    });
}

// Svelte store integration
export function useLegalAI() {
    const machine = createLegalAIMachine();
    const { state, send, service } = useMachine(machine);
    
    // Derived stores for specific state slices
    const currentState = derived(state, ($state) => $state.value);
    const context = derived(state, ($state) => $state.context);
    const canTransition = derived(state, ($state) => (event) => $state.can(event));
    
    // Specific context slices
    const chatHistory = derived(context, ($context) => $context.chatHistory);
    const searchResults = derived(context, ($context) => $context.searchResults);
    const currentCase = derived(context, ($context) => $context.currentCase);
    const gpuStatus = derived(context, ($context) => $context.gpuStatus);
    const errors = derived(context, ($context) => $context.errors);
    
    // Action creators
    const actions = {
        startSearch: (query) => send({ type: 'START_SEARCH', query }),
        startChat: () => send({ type: 'START_CHAT' }),
        sendMessage: (message) => send({ type: 'SEND_MESSAGE', message }),
        uploadDocument: (file, documentType) => send({ 
            type: 'UPLOAD_DOCUMENT', 
            file, 
            documentType 
        }),
        analyzeCase: (caseData) => send({ type: 'ANALYZE_CASE', caseData }),
        connectGPU: () => send({ type: 'CONNECT_GPU' }),
        retry: () => send({ type: 'RETRY' }),
        clearError: () => send({ type: 'CLEAR_ERROR' }),
        clearChat: () => send({ type: 'CLEAR_CHAT' })
    };
    
    // Initialize GPU on mount
    if (browser) {
        actions.connectGPU();
    }
    
    return {
        // State
        state,
        currentState,
        context,
        canTransition,
        
        // Context slices
        chatHistory,
        searchResults,
        currentCase,
        gpuStatus,
        errors,
        
        // Actions
        send,
        ...actions,
        
        // Service
        service
    };
}

// ============================================================================
// WEBSOCKET INTEGRATION WITH XSTATE
// ============================================================================

export function createWebSocketService(url) {
    return createMachine({
        id: 'websocket',
        initial: 'disconnected',
        context: {
            url,
            socket: null,
            reconnectAttempts: 0,
            maxReconnectAttempts: 5,
            messages: [],
            subscriptions: []
        },
        states: {
            disconnected: {
                on: {
                    CONNECT: 'connecting'
                }
            },
            connecting: {
                invoke: {
                    src: 'connectWebSocket',
                    onDone: {
                        target: 'connected',
                        actions: ['setSocket']
                    },
                    onError: {
                        target: 'reconnecting',
                        actions: ['incrementReconnectAttempts']
                    }
                }
            },
            connected: {
                entry: ['resetReconnectAttempts'],
                invoke: {
                    src: 'listenToSocket',
                    onError: 'reconnecting'
                },
                on: {
                    SEND_MESSAGE: {
                        actions: ['sendMessage']
                    },
                    SUBSCRIBE: {
                        actions: ['addSubscription']
                    },
                    DISCONNECT: 'disconnected'
                }
            },
            reconnecting: {
                after: {
                    3000: [
                        {
                            target: 'connecting',
                            cond: 'canReconnect'
                        },
                        {
                            target: 'failed'
                        }
                    ]
                }
            },
            failed: {
                on: {
                    RETRY: 'connecting'
                }
            }
        }
    }, {
        services: {
            connectWebSocket: (context) => {
                return new Promise((resolve, reject) => {
                    const socket = new WebSocket(context.url);
                    
                    socket.onopen = () => resolve(socket);
                    socket.onerror = () => reject(new Error('WebSocket connection failed'));
                    
                    setTimeout(() => reject(new Error('Connection timeout')), 10000);
                });
            },
            
            listenToSocket: (context) => {
                return new Promise((resolve, reject) => {
                    const { socket } = context;
                    
                    socket.onmessage = (event) => {
                        try {
                            const message = JSON.parse(event.data);
                            // Handle message based on type
                            handleWebSocketMessage(message);
                        } catch (error) {
                            console.error('Failed to parse WebSocket message:', error);
                        }
                    };
                    
                    socket.onclose = () => reject(new Error('WebSocket closed'));
                    socket.onerror = () => reject(new Error('WebSocket error'));
                });
            }
        },
        
        actions: {
            setSocket: assign({
                socket: (_, event) => event.data
            }),
            
            incrementReconnectAttempts: assign({
                reconnectAttempts: (context) => context.reconnectAttempts + 1
            }),
            
            resetReconnectAttempts: assign({
                reconnectAttempts: 0
            }),
            
            sendMessage: (context, event) => {
                if (context.socket && context.socket.readyState === WebSocket.OPEN) {
                    context.socket.send(JSON.stringify(event.message));
                }
            },
            
            addSubscription: assign({
                subscriptions: (context, event) => [
                    ...context.subscriptions,
                    event.channel
                ]
            })
        },
        
        guards: {
            canReconnect: (context) => context.reconnectAttempts < context.maxReconnectAttempts
        }
    });
}

function handleWebSocketMessage(message) {
    // Handle different message types
    switch (message.type) {
        case 'gpu_result':
            // Handle GPU computation result
            console.log('GPU Result:', message.data);
            break;
        case 'som_update':
            // Handle SOM training update
            console.log('SOM Update:', message.data);
            break;
        case 'xstate_event':
            // Handle XState event from server
            console.log('XState Event:', message.data);
            break;
        default:
            console.log('Unknown message type:', message.type);
    }
}

// ============================================================================
// USAGE EXAMPLE IN SVELTE COMPONENT
// ============================================================================

/*
<script>
import { useLegalAI, createWebSocketService } from '$lib/stores/legal-ai-machine.js';
// Orphaned content: import { useMachine
    
    // Initialize the main legal AI machine
    const {
        currentState,
        chatHistory,
        searchResults,
        gpuStatus,
        startSearch,
        sendMessage,
        uploadDocument
    } = useLegalAI();
    
    // Initialize WebSocket service
    const wsService = createWebSocketService('ws://localhost:8094/ws');
    const { state: wsState, send: wsSend } = useMachine(wsService);
    
    // Connect WebSocket on mount
    import { onMount } from 'svelte';
    onMount(() => {
        wsSend({ type: 'CONNECT' });
    });
    
    // Reactive statements
    $: isSearching = $currentState === 'searching';
    $: isChatting = typeof $currentState === 'object' && $currentState.chatting;
    $: gpuReady = $gpuStatus === 'initialized';
    $: wsConnected = $wsState.matches('connected');
</script>

<div class="legal-ai-interface">
    <div class="status-bar">
        <span class="status-item" class:active={gpuReady}>
            🎮 GPU: {$gpuStatus}
        </span>
        <span class="status-item" class:active={wsConnected}>
            🔗 WebSocket: {$wsState.value}
        </span>
        <span class="status-item">
            🧠 State: {$currentState}
        </span>
    </div>
    
    {#if isSearching}
        <div class="search-indicator">
            <span>🔍 Searching...</span>
        </div>
    {/if}
    
    <div class="chat-container">
        {#each $chatHistory as message}
            <div class="message" class:user={message.role === 'user'}>
                {message.content}
            </div>
        {/each}
    </div>
    
    <button on:click={() => startSearch('contract law')}>
        Search Contract Law
    </button>
    
    <button on:click={() => sendMessage('What is consideration in contracts?')}>
        Ask Question
    </button>
</div>
*/

export default {
    createLegalAIMachine,
    useLegalAI,
    createWebSocketService,
    GPUServiceWorkerClient
};
