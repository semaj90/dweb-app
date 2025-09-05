<!-- Progress Indicator Component -->
<script lang="ts">
  interface Props {
    steps: Array<{
    currentStep: number
    validationResults: Record<number, {
  }
  let {
    steps,
    currentStep,
    validationResults
  } = $props();



    import { createEventDispatcher } from 'svelte';

            id: string
        title: string
        description: string
        required: boolean
        estimatedTime: number
    }>;
                isValid: boolean
        errors: string[];
        warnings: string[];
    }> = {};

    const dispatch = createEventDispatcher<{
        'step-click': number;
    }>();

    function handleStepClick(stepIndex: number): void {
        if (stepIndex <= currentStep || !steps[stepIndex].required) {
            dispatch('step-click', stepIndex);
        }
    }

    function getStepStatus(stepIndex: number): 'completed' | 'current' | 'pending' | 'error' | 'warning' {
        if (stepIndex < currentStep) {
            const validation = validationResults[stepIndex];
            if (validation) {
                if (!validation.isValid) return 'error';
                if (validation.warnings.length > 0) return 'warning';
            }
            return 'completed';
        } else if (stepIndex === currentStep) {
            return 'current';
        } else {
            return 'pending';
        }
    }

    function getStepIcon(stepIndex: number): string {
        const status = getStepStatus(stepIndex);

        switch (status) {
            case 'completed':
                return 'check';
            case 'current':
                return 'current';
            case 'error':
                return 'exclamation';
            case 'warning':
                return 'exclamation-triangle';
            default:
                return 'circle';
        }
    }

    let progressPercentage = $derived(Math.round((currentStep / (steps.length - 1)) * 100););
</script>

<div class="progress-indicator bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
    <!-- Progress bar -->
    <div class="w-full bg-gray-200 dark:bg-gray-700 h-1">
        <div
            class="h-1 bg-blue-600 transition-all duration-500 ease-out"
            style="width: {progressPercentage}%"
        />
    </div>

    <!-- Steps navigation -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <nav aria-label="Progress">
            <ol class="flex items-center justify-between space-x-2 lg:space-x-4">
                {#each steps as step, index}
                    {@const status = getStepStatus(index)}
                    {@const icon = getStepIcon(index)}
                    {@const isClickable = index <= currentStep || !step.required}

                    <li class="flex-1 min-w-0">
                        <button
                            onclick={() => handleStepClick(index)}
                            disabled={!isClickable}
                            class="group flex items-center w-full text-left
                                   {isClickable ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700' : 'cursor-not-allowed'}
                                   rounded-lg p-2 transition-colors duration-200"
                            aria-current={status === 'current' ? 'step' : undefined}
                        >
                            <!-- Step indicator -->
                            <div class="flex-shrink-0 relative">
                                <div class="flex items-center justify-center w-8 h-8 rounded-full
                                           {status === 'completed' ? 'bg-green-100 dark:bg-green-900' : ''}
                                           {status === 'current' ? 'bg-blue-100 dark:bg-blue-900 border-2 border-blue-600' : ''}
                                           {status === 'error' ? 'bg-red-100 dark:bg-red-900' : ''}
                                           {status === 'warning' ? 'bg-yellow-100 dark:bg-yellow-900' : ''}
                                           {status === 'pending' ? 'bg-gray-100 dark:bg-gray-700 border-2 border-gray-300 dark:border-gray-600' : ''}">

                                    {#if icon === 'check'}
                                        <svg class="w-4 h-4 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
                                            <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                                        </svg>
                                    {:else if icon === 'current'}
                                        <div class="w-3 h-3 rounded-full bg-blue-600"></div>
                                    {:else if icon === 'exclamation'}
                                        <svg class="w-4 h-4 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
                                            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                                        </svg>
                                    {:else if icon === 'exclamation-triangle'}
                                        <svg class="w-4 h-4 text-yellow-600 dark:text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                                            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                                        </svg>
                                    {:else}
                                        <div class="w-3 h-3 rounded-full bg-gray-400 dark:bg-gray-500"></div>
                                    {/if}
                                </div>

                                <!-- Step number badge for smaller screens -->
                                <div class="absolute -top-1 -right-1 w-4 h-4 bg-gray-100 dark:bg-gray-700 rounded-full
                                           flex items-center justify-center text-xs font-medium text-gray-600 dark:text-gray-400
                                           lg:hidden">
                                    {index + 1}
                                </div>
                            </div>

                            <!-- Step content -->
                            <div class="ml-3 min-w-0 flex-1">
                                <div class="flex items-center space-x-2">
                                    <p class="text-sm font-medium text-gray-900 dark:text-white truncate
                                             {status === 'current' ? 'text-blue-600 dark:text-blue-400' : ''}">
                                        {step.title}
                                    </p>

                                    {#if step.required}
                                        <span class="inline-flex items-center px-1.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                                            Required
                                        </span>
                                    {/if}

                                    <!-- Estimated time badge -->
                                    <span class="hidden lg:inline-flex items-center px-1.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200">
                                        ~{step.estimatedTime}m
                                    </span>
                                </div>

                                <p class="hidden lg:block text-xs text-gray-500 dark:text-gray-400 truncate mt-1">
                                    {step.description}
                                </p>

                                <!-- Validation messages -->
                                {#if validationResults[index]}
                                    {@const validation = validationResults[index]}
                                    {#if validation.errors.length > 0}
                                        <div class="mt-1 text-xs text-red-600 dark:text-red-400">
                                            {validation.errors.length} error{validation.errors.length !== 1 ? 's' : ''}
                                        </div>
                                    {:else if validation.warnings.length > 0}
                                        <div class="mt-1 text-xs text-yellow-600 dark:text-yellow-400">
                                            {validation.warnings.length} warning{validation.warnings.length !== 1 ? 's' : ''}
                                        </div>
                                    {/if}
                                {/if}
                            </div>
                        </button>

                        <!-- Connector line -->
                        {#if index < steps.length - 1}
                            <div class="hidden lg:block absolute top-1/2 right-0 transform translate-x-2 -translate-y-1/2">
                                <div class="w-4 h-0.5 bg-gray-300 dark:bg-gray-600"></div>
                            </div>
                        {/if}
                    </li>
                {/each}
            </ol>
        </nav>

        <!-- Mobile step counter -->
        <div class="lg:hidden mt-3 text-center">
            <span class="text-sm text-gray-500 dark:text-gray-400">
                Step {currentStep + 1} of {steps.length}: {steps[currentStep].title}
            </span>
        </div>
    </div>
</div>

<style>
    .progress-indicator li {
        position: relative
    }
</style>
