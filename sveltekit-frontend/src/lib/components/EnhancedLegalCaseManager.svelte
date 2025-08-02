<!-- Enhanced Legal Case Manager with Production Features -->
<script lang="ts">
    import { onMount, tick } from 'svelte';
    import { writable, derived } from 'svelte/store';
    import { page } from '$app/stores';
    import { goto } from '$app/navigation';
    import { dev } from '$app/environment';

    // Import subcomponents
    import CaseInfoForm from './subcomponents/CaseInfoForm.svelte';
    import DocumentUploadForm from './subcomponents/DocumentUploadForm.svelte';
    import EvidenceAnalysisForm from './subcomponents                    {#if recognition}
                        <button
                            on:click={toggleVoiceListening}
                            class="p-2 rounded-lg border border-gray-300 dark:border-gray-600
                                   hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors
                                   {isListening ? 'bg-red-50 border-red-300 text-red-600' : ''}"
                            title="Toggle voice commands"
                            aria-label="Toggle voice commands"
                        >
                            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clip-rule="evenodd" />
                            </svg>
                        </button>
                    {/if}ysisForm.svelte';
    import AIAnalysisForm from './subcomponents/AIAnalysisForm.svelte';
    import ReviewSubmitForm from './subcomponents/ReviewSubmitForm.svelte';
    import ProgressIndicator from './subcomponents/ProgressIndicator.svelte';
    import LoadingSpinner from './subcomponents/LoadingSpinner.svelte';

    // Import enhanced services
    import { ocrProcessor } from '$lib/services/enhanced-ocr-processor';
    import { caseStore, type CaseData } from '$lib/stores/caseStore';
    import { notificationStore } from '$lib/stores/notificationStore';
    import { analyticsStore } from '$lib/stores/analyticsStore';

    // Types
    interface StepConfig {
        id: string;
        title: string;
        description: string;
        component: any;
        required: boolean;
        estimatedTime: number; // in minutes
    }

    interface ValidationResult {
        isValid: boolean;
        errors: string[];
        warnings: string[];
    }

    // Reactive state
    let currentStep = writable(0);
    let isProcessing = writable(false);
    let autoSaveEnabled = writable(true);
    let validationResults = writable<Record<number, ValidationResult>>({});
    let processingQueue = writable<string[]>([]);

    // Case data with enhanced tracking
    let caseData = writable<CaseData>({
        id: '',
        title: '',
        description: '',
        clientInfo: {
            name: '',
            email: '',
            phone: '',
            address: ''
        },
        documents: [],
        evidence: [],
        aiAnalysis: null,
        status: 'draft',
        priority: 'medium',
        tags: [],
        metadata: {
            createdAt: new Date(),
            updatedAt: new Date(),
            version: 1,
            workflow: 'standard'
        }
    });

    // Step configuration
    const steps: StepConfig[] = [
        {
            id: 'case-info',
            title: 'Case Information',
            description: 'Basic case details and client information',
            component: CaseInfoForm,
            required: true,
            estimatedTime: 5
        },
        {
            id: 'document-upload',
            title: 'Document Upload',
            description: 'Upload and process case documents',
            component: DocumentUploadForm,
            required: true,
            estimatedTime: 10
        },
        {
            id: 'evidence-analysis',
            title: 'Evidence Analysis',
            description: 'Analyze and categorize evidence',
            component: EvidenceAnalysisForm,
            required: false,
            estimatedTime: 15
        },
        {
            id: 'ai-analysis',
            title: 'AI Analysis',
            description: 'AI-powered case analysis and recommendations',
            component: AIAnalysisForm,
            required: false,
            estimatedTime: 8
        },
        {
            id: 'review-submit',
            title: 'Review & Submit',
            description: 'Final review and case submission',
            component: ReviewSubmitForm,
            required: true,
            estimatedTime: 5
        }
    ];

    // Derived stores
    const totalSteps = derived(currentStep, () => steps.length);
    const progressPercentage = derived(currentStep, ($step) =>
        Math.round(($step / (steps.length - 1)) * 100)
    );
    const currentStepConfig = derived(currentStep, ($step) => steps[$step]);
    const isFirstStep = derived(currentStep, ($step) => $step === 0);
    const isLastStep = derived(currentStep, ($step) => $step === steps.length - 1);
    const estimatedTimeRemaining = derived(currentStep, ($step) =>
        steps.slice($step + 1).reduce((sum, step) => sum + step.estimatedTime, 0)
    );

    // Auto-save functionality
    let autoSaveTimeout: NodeJS.Timeout;
    const AUTOSAVE_DELAY = 3000; // 3 seconds

    $: if ($autoSaveEnabled && $caseData) {
        clearTimeout(autoSaveTimeout);
        autoSaveTimeout = setTimeout(saveProgress, AUTOSAVE_DELAY);
    }

    // Methods
    async function saveProgress(): Promise<void> {
        try {
            await caseStore.updateCase($caseData);
            notificationStore.addNotification({
                type: 'info',
                message: 'Progress auto-saved',
                timeout: 2000
            });
        } catch (error) {
            console.error('Auto-save failed:', error);
            notificationStore.addNotification({
                type: 'error',
                message: 'Auto-save failed. Please save manually.',
                timeout: 5000
            });
        }
    }

    async function validateCurrentStep(): Promise<ValidationResult> {
        const stepConfig = $currentStepConfig;
        const result: ValidationResult = {
            isValid: true,
            errors: [],
            warnings: []
        };

        switch (stepConfig.id) {
            case 'case-info':
                if (!$caseData.title.trim()) {
                    result.errors.push('Case title is required');
                }
                if (!$caseData.clientInfo.name.trim()) {
                    result.errors.push('Client name is required');
                }
                if (!$caseData.clientInfo.email.trim()) {
                    result.warnings.push('Client email is recommended');
                }
                break;

            case 'document-upload':
                if ($caseData.documents.length === 0) {
                    result.errors.push('At least one document is required');
                }
                break;

            case 'evidence-analysis':
                if ($caseData.evidence.length === 0) {
                    result.warnings.push('No evidence items found');
                }
                break;

            case 'ai-analysis':
                if (!$caseData.aiAnalysis) {
                    result.warnings.push('AI analysis not completed');
                }
                break;

            case 'review-submit':
                // Final validation
                if (!$caseData.title || !$caseData.clientInfo.name) {
                    result.errors.push('Required fields missing');
                }
                break;
        }

        result.isValid = result.errors.length === 0;

        // Update validation store
        validationResults.update(results => ({
            ...results,
            [$currentStep]: result
        }));

        return result;
    }

    async function nextStep(): Promise<void> {
        isProcessing.set(true);

        try {
            // Validate current step
            const validation = await validateCurrentStep();

            if (!validation.isValid) {
                notificationStore.addNotification({
                    type: 'error',
                    message: `Please fix errors: ${validation.errors.join(', ')}`,
                    timeout: 5000
                });
                return;
            }

            // Show warnings if any
            if (validation.warnings.length > 0) {
                notificationStore.addNotification({
                    type: 'warning',
                    message: `Warnings: ${validation.warnings.join(', ')}`,
                    timeout: 4000
                });
            }

            // Save progress
            await saveProgress();

            // Track analytics
            analyticsStore.trackEvent('case_step_completed', {
                step: $currentStep,
                stepId: $currentStepConfig.id,
                caseId: $caseData.id
            });

            // Move to next step
            if ($currentStep < steps.length - 1) {
                currentStep.update(n => n + 1);

                // Smooth scroll to top
                await tick();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }

        } catch (error) {
            console.error('Error advancing to next step:', error);
            notificationStore.addNotification({
                type: 'error',
                message: 'Failed to advance to next step',
                timeout: 5000
            });
        } finally {
            isProcessing.set(false);
        }
    }

    async function previousStep(): Promise<void> {
        if ($currentStep > 0) {
            currentStep.update(n => n - 1);
            await tick();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }

    async function jumpToStep(stepIndex: number): Promise<void> {
        if (stepIndex >= 0 && stepIndex < steps.length) {
            // Validate all previous steps
            let canJump = true;

            for (let i = 0; i < stepIndex; i++) {
                currentStep.set(i);
                const validation = await validateCurrentStep();

                if (!validation.isValid && steps[i].required) {
                    notificationStore.addNotification({
                        type: 'error',
                        message: `Cannot skip required step: ${steps[i].title}`,
                        timeout: 5000
                    });
                    canJump = false;
                    break;
                }
            }

            if (canJump) {
                currentStep.set(stepIndex);
                await tick();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        }
    }

    async function submitCase(): Promise<void> {
        isProcessing.set(true);

        try {
            // Final validation
            const validation = await validateCurrentStep();

            if (!validation.isValid) {
                throw new Error(`Validation failed: ${validation.errors.join(', ')}`);
            }

            // Update case status
            caseData.update(data => ({
                ...data,
                status: 'submitted',
                metadata: {
                    ...data.metadata,
                    submittedAt: new Date()
                }
            }));

            // Submit to backend
            const response = await fetch('/api/cases/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify($caseData)
            });

            if (!response.ok) {
                throw new Error('Failed to submit case');
            }

            const result = await response.json();

            // Track analytics
            analyticsStore.trackEvent('case_submitted', {
                caseId: result.id,
                stepCount: steps.length,
                documentCount: $caseData.documents.length,
                evidenceCount: $caseData.evidence.length
            });

            // Show success notification
            notificationStore.addNotification({
                type: 'success',
                message: 'Case submitted successfully!',
                timeout: 5000
            });

            // Redirect to case view
            await goto(`/cases/${result.id}`);

        } catch (error) {
            console.error('Case submission failed:', error);
            notificationStore.addNotification({
                type: 'error',
                message: 'Failed to submit case. Please try again.',
                timeout: 5000
            });
        } finally {
            isProcessing.set(false);
        }
    }

    async function resetCase(): Promise<void> {
        if (confirm('Are you sure you want to reset all case data? This cannot be undone.')) {
            caseData.set({
                id: '',
                title: '',
                description: '',
                clientInfo: {
                    name: '',
                    email: '',
                    phone: '',
                    address: ''
                },
                documents: [],
                evidence: [],
                aiAnalysis: null,
                status: 'draft',
                priority: 'medium',
                tags: [],
                metadata: {
                    createdAt: new Date(),
                    updatedAt: new Date(),
                    version: 1,
                    workflow: 'standard'
                }
            });

            currentStep.set(0);
            validationResults.set({});

            notificationStore.addNotification({
                type: 'info',
                message: 'Case data reset',
                timeout: 3000
            });
        }
    }

    // Voice commands setup (if supported)
    let recognition: SpeechRecognition | null = null;
    let isListening = false;

    function setupVoiceCommands(): void {
        if (typeof window !== 'undefined' && 'webkitSpeechRecognition' in window) {
            recognition = new (window as any).webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onresult = (event: SpeechRecognitionEvent) => {
                const command = event.results[0][0].transcript.toLowerCase();
                handleVoiceCommand(command);
            };

            recognition.onerror = () => {
                isListening = false;
            };

            recognition.onend = () => {
                isListening = false;
            };
        }
    }

    function handleVoiceCommand(command: string): void {
        if (command.includes('next')) {
            nextStep();
        } else if (command.includes('previous') || command.includes('back')) {
            previousStep();
        } else if (command.includes('save')) {
            saveProgress();
        } else if (command.includes('submit')) {
            submitCase();
        }
    }

    function toggleVoiceListening(): void {
        if (!recognition) return;

        if (isListening) {
            recognition.stop();
            isListening = false;
        } else {
            recognition.start();
            isListening = true;
        }
    }

    // Keyboard shortcuts
    function handleKeydown(event: KeyboardEvent): void {
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case 'ArrowRight':
                    event.preventDefault();
                    nextStep();
                    break;
                case 'ArrowLeft':
                    event.preventDefault();
                    previousStep();
                    break;
                case 's':
                    event.preventDefault();
                    saveProgress();
                    break;
                case 'Enter':
                    if ($isLastStep) {
                        event.preventDefault();
                        submitCase();
                    }
                    break;
            }
        }
    }

    // Lifecycle
    onMount(async () => {
        // Initialize OCR processor
        ocrProcessor.on('initialized', (message) => {
            console.log('OCR Service:', message);
        });

        ocrProcessor.on('processing:start', (data) => {
            processingQueue.update(queue => [...queue, data.filename]);
        });

        ocrProcessor.on('processing:complete', (result) => {
            processingQueue.update(queue =>
                queue.filter(filename => filename !== result.metadata.filename)
            );
        });

        // Setup voice commands
        setupVoiceCommands();

        // Track page view
        analyticsStore.trackPageView('/case/new');

        // Check for case ID in URL (edit mode)
        const caseId = $page.url.searchParams.get('id');
        if (caseId) {
            try {
                const existingCase = await caseStore.getCase(caseId);
                if (existingCase) {
                    caseData.set(existingCase);
                    notificationStore.addNotification({
                        type: 'info',
                        message: 'Loaded existing case for editing',
                        timeout: 3000
                    });
                }
            } catch (error) {
                console.error('Failed to load existing case:', error);
            }
        }
    });

    // Reactive statement for step validation
    $: {
        if ($currentStep >= 0) {
            validateCurrentStep();
        }
    }
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="legal-case-manager min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Header with progress -->
    <div class="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <div class="flex items-center space-x-4">
                    <h1 class="text-xl font-semibold text-gray-900 dark:text-white">
                        Legal Case Manager
                    </h1>
                    <div class="text-sm text-gray-500 dark:text-gray-400">
                        Step {$currentStep + 1} of {$totalSteps}
                    </div>
                </div>

                <div class="flex items-center space-x-4">
                    <!-- Voice control button -->
                    {#if recognition}
                        <button
                            on:click={toggleVoiceListening}
                            class="p-2 rounded-lg border border-gray-300 dark:border-gray-600
                                   hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors
                                   {isListening ? 'bg-red-50 border-red-300 text-red-600' : ''}"
                            title="Toggle voice commands"
                        >
                            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clip-rule="evenodd" />
                            </svg>
                        </button>
                    {/if}

                    <!-- Auto-save toggle -->
                    <label class="flex items-center space-x-2 text-sm">
                        <input
                            type="checkbox"
                            bind:checked={$autoSaveEnabled}
                            class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span class="text-gray-600 dark:text-gray-400">Auto-save</span>
                    </label>

                    <!-- Estimated time -->
                    {#if $estimatedTimeRemaining > 0}
                        <div class="text-sm text-gray-500 dark:text-gray-400">
                            ~{$estimatedTimeRemaining} min remaining
                        </div>
                    {/if}
                </div>
            </div>
        </div>
    </div>

    <!-- Progress indicator -->
    <ProgressIndicator
        {steps}
        currentStep={$currentStep}
        validationResults={$validationResults}
        on:step-click={(e) => jumpToStep(e.detail)}
    />

    <!-- Processing queue indicator -->
    {#if $processingQueue.length > 0}
        <div class="bg-blue-50 dark:bg-blue-900/20 border-b border-blue-200 dark:border-blue-800">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
                <div class="flex items-center space-x-3">
                    <LoadingSpinner size="sm" />
                    <span class="text-sm text-blue-700 dark:text-blue-300">
                        Processing {$processingQueue.length} file(s): {$processingQueue.join(', ')}
                    </span>
                </div>
            </div>
        </div>
    {/if}

    <!-- Main content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            <!-- Step header -->
            <div class="border-b border-gray-200 dark:border-gray-700 px-6 py-4">
                <h2 class="text-lg font-medium text-gray-900 dark:text-white">
                    {$currentStepConfig.title}
                </h2>
                <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    {$currentStepConfig.description}
                </p>
            </div>

            <!-- Step content -->
            <div class="p-6">
                {#if $isProcessing}
                    <div class="flex items-center justify-center py-12">
                        <LoadingSpinner size="lg" />
                        <span class="ml-3 text-gray-600 dark:text-gray-400">Processing...</span>
                    </div>
                {:else}
                    <svelte:component
                        this={$currentStepConfig.component}
                        bind:caseData={$caseData}
                        validationResult={$validationResults[$currentStep]}
                        on:data-changed={() => {
                            caseData.update(data => ({ ...data, metadata: { ...data.metadata, updatedAt: new Date() } }));
                        }}
                        on:request-validation={validateCurrentStep}
                    />
                {/if}
            </div>

            <!-- Navigation footer -->
            <div class="border-t border-gray-200 dark:border-gray-700 px-6 py-4">
                <div class="flex items-center justify-between">
                    <div class="flex space-x-3">
                        <button
                            on:click={previousStep}
                            disabled={$isFirstStep || $isProcessing}
                            class="px-4 py-2 border border-gray-300 dark:border-gray-600
                                   rounded-md shadow-sm text-sm font-medium
                                   text-gray-700 dark:text-gray-300
                                   bg-white dark:bg-gray-700
                                   hover:bg-gray-50 dark:hover:bg-gray-600
                                   disabled:opacity-50 disabled:cursor-not-allowed
                                   transition-colors"
                        >
                            Previous
                        </button>

                        <button
                            on:click={resetCase}
                            disabled={$isProcessing}
                            class="px-4 py-2 border border-red-300 dark:border-red-600
                                   rounded-md shadow-sm text-sm font-medium
                                   text-red-700 dark:text-red-300
                                   bg-white dark:bg-gray-700
                                   hover:bg-red-50 dark:hover:bg-red-900/20
                                   disabled:opacity-50 disabled:cursor-not-allowed
                                   transition-colors"
                        >
                            Reset
                        </button>
                    </div>

                    <div class="flex space-x-3">
                        <button
                            on:click={saveProgress}
                            disabled={$isProcessing}
                            class="px-4 py-2 border border-gray-300 dark:border-gray-600
                                   rounded-md shadow-sm text-sm font-medium
                                   text-gray-700 dark:text-gray-300
                                   bg-white dark:bg-gray-700
                                   hover:bg-gray-50 dark:hover:bg-gray-600
                                   disabled:opacity-50 disabled:cursor-not-allowed
                                   transition-colors"
                        >
                            Save Progress
                        </button>

                        {#if $isLastStep}
                            <button
                                on:click={submitCase}
                                disabled={$isProcessing}
                                class="px-4 py-2 border border-transparent
                                       rounded-md shadow-sm text-sm font-medium
                                       text-white bg-blue-600
                                       hover:bg-blue-700
                                       disabled:opacity-50 disabled:cursor-not-allowed
                                       transition-colors"
                            >
                                Submit Case
                            </button>
                        {:else}
                            <button
                                on:click={nextStep}
                                disabled={$isProcessing}
                                class="px-4 py-2 border border-transparent
                                       rounded-md shadow-sm text-sm font-medium
                                       text-white bg-blue-600
                                       hover:bg-blue-700
                                       disabled:opacity-50 disabled:cursor-not-allowed
                                       transition-colors"
                            >
                                Next
                            </button>
                        {/if}
                    </div>
                </div>

                <!-- Keyboard shortcuts help -->
                <div class="mt-4 text-xs text-gray-500 dark:text-gray-400">
                    <span class="font-medium">Keyboard shortcuts:</span>
                    Ctrl+→ Next • Ctrl+← Previous • Ctrl+S Save • Ctrl+Enter Submit
                </div>
            </div>
        </div>
    </main>
</div>

<style>
    .legal-case-manager {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }

    /* Smooth transitions for all interactive elements */
    .legal-case-manager button,
    .legal-case-manager input {
        transition: all 0.2s ease-in-out;
    }
</style>
