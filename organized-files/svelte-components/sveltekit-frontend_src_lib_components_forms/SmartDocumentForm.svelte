<!-- Smart Document Form with OCR Auto-Population -->
<script lang="ts">
  interface Props {
    title?: any;
    description?: any;
    formSchema: FormField[] ;
    enableOCR?: any;
    enableSmartSuggestions?: any;
    documentTypes: string[] ;
  }
  let {
    title = "Smart Document Form",
    description = "Upload a document for automatic field extraction and population",
    formSchema = [],
    enableOCR = true,
    enableSmartSuggestions = true,
    documentTypes = ['legal_document', 'contract', 'form']
  } = $props();



	import { createEventDispatcher, onMount } from 'svelte';
	import { Button } from 'bits-ui';
	import { Card, CardContent, CardHeader, CardTitle } from 'bits-ui';
	import { Badge } from 'bits-ui';
	import { Progress } from 'bits-ui';
	import { Input } from 'bits-ui';
	import { Textarea } from 'bits-ui';
	import { Label } from 'bits-ui';
	import { ocrService, type ExtractedField, type FormField, type FieldType } from '$lib/services/ocrService';
	import { enhancedRAG } from '$lib/services/enhancedRAG';
	import { fade, fly, scale } from 'svelte/transition';
	import { writable } from 'svelte/store';

						
	const dispatch = createEventDispatcher<{
		submit: { formData: Record<string, any>; extractedFields: ExtractedField[] };
		fieldChange: { fieldName: string value: string confidence?: number };
		ocrComplete: { result: any extractedFields: ExtractedField[] };
	}>();

	// Component state
	let fileInput: HTMLInputElement
	let uploadedFile: File | null = null;
	let populatedFields: FormField[] = [...formSchema];
	let isProcessing = false;
	let showPreview = false;
	let selectedDocumentType = 'auto';

	// OCR stores
	let processing = $derived(ocrService.processing$)
	let progress = $derived(ocrService.progress$)
	let ocrResult = $derived(ocrService.currentResult$)
	let extractedFields = $derived(ocrService.extractedFields$)

	// Form validation
	const formErrors = writable<Record<string, string>>({});
	let isFormValid = false;

	// Smart suggestions
	let activeSuggestions: Record<string, string[]> = {};
	let suggestionLoading: Record<string, boolean> = {};

	// Default form schema if none provided
	onMount(() => {
		if (formSchema.length === 0) {
			populatedFields = [
				{ name: 'client_name', type: 'name', label: 'Client Name', required: true },
				{ name: 'case_number', type: 'case_number', label: 'Case Number', required: false },
				{ name: 'document_date', type: 'date', label: 'Document Date', required: true },
				{ name: 'jurisdiction', type: 'text_block', label: 'Jurisdiction', required: false },
				{ name: 'contact_email', type: 'email', label: 'Contact Email', required: true },
				{ name: 'contact_phone', type: 'phone', label: 'Contact Phone', required: false },
				{ name: 'description', type: 'text_block', label: 'Description', required: true },
				{ name: 'notes', type: 'text_block', label: 'Additional Notes', required: false }
			];
		}
	});

	// Handle file upload
	const handleFileUpload = async () => {
		if (!uploadedFile || !enableOCR) return;

		try {
			isProcessing = true;

			const result = await ocrService.processDocument(uploadedFile, {
				documentType: selectedDocumentType as any,
				extractFields: true,
				qualityEnhancement: true
			});

			// Auto-populate form fields
			populatedFields = ocrService.autoPopulateForm($extractedFields, populatedFields);

			// Generate smart suggestions for incomplete fields
			if (enableSmartSuggestions) {
				await generateSmartSuggestions(result.text);
			}

			showPreview = true;
			dispatch('ocrComplete', { result, extractedFields: $extractedFields });

		} catch (error) {
			console.error('OCR processing failed:', error);
		} finally {
			isProcessing = false;
		}
	};

	// Generate smart suggestions for incomplete fields
	const generateSmartSuggestions = async (documentText: string) => {
		for (const field of populatedFields) {
			if (!field.value && enableSmartSuggestions) {
				try {
					suggestionLoading[field.name] = true;
					const suggestions = await ocrService.getSuggestions(field.name, field.type, documentText);
					activeSuggestions[field.name] = suggestions;
				} catch (error) {
					console.warn(`Failed to generate suggestions for ${field.name}:`, error);
				} finally {
					suggestionLoading[field.name] = false;
				}
			}
		}
		activeSuggestions = { ...activeSuggestions }; // Trigger reactivity
	};

	// Handle field value changes
	const handleFieldChange = (fieldName: string, value: string, confidence?: number) => {
		const fieldIndex = populatedFields.findIndex(f => f.name === fieldName);
		if (fieldIndex !== -1) {
			populatedFields[fieldIndex].value = value;
			populatedFields[fieldIndex].confidence = confidence;
			
			// Clear suggestions once user makes a selection
			delete activeSuggestions[fieldName];
			activeSuggestions = { ...activeSuggestions };
		}

		// Validate field
		validateField(fieldName, value);
		
		dispatch('fieldChange', { fieldName, value, confidence });
	};

	// Apply suggestion to field
	const applySuggestion = (fieldName: string, suggestion: string) => {
		handleFieldChange(fieldName, suggestion, 0.8);
	};

	// Field validation
	const validateField = (fieldName: string, value: string) => {
		const field = populatedFields.find(f => f.name === fieldName);
		if (!field) return;

		const errors = { ...$formErrors };

		// Required field validation
		if (field.required && !value.trim()) {
			errors[fieldName] = 'This field is required';
		} else if (field.validation) {
			try {
				field.validation.parse(value);
				delete errors[fieldName];
			} catch (error: any) {
				errors[fieldName] = error.errors?.[0]?.message || 'Invalid value';
			}
		} else {
			delete errors[fieldName];
		}

		formErrors.set(errors);
		isFormValid = Object.keys(errors).length === 0 && 
			populatedFields.filter(f => f.required).every(f => f.value?.trim());
	};

	// Form submission
	const handleSubmit = () => {
		// Final validation
		populatedFields.forEach(field => {
			if (field.value) validateField(field.name, field.value);
		});

		if (isFormValid) {
			const formData = populatedFields.reduce((acc, field) => {
				acc[field.name] = field.value || '';
				return acc;
			}, {} as Record<string, any>);

			dispatch('submit', { formData, extractedFields: $extractedFields });
		}
	};

	// Get field type icon
	const getFieldTypeIcon = (type: FieldType) => {
		switch (type) {
			case 'name': return '👤';
			case 'email': return '📧';
			case 'phone': return '📞';
			case 'date': return '📅';
			case 'address': return '📍';
			case 'case_number': return '📋';
			case 'monetary_amount': return '💰';
			default: return '📝';
		}
	};

	// Get confidence color
	const getConfidenceColor = (confidence?: number) => {
		if (!confidence) return 'bg-gray-500';
		if (confidence >= 0.9) return 'bg-green-500';
		if (confidence >= 0.7) return 'bg-yellow-500';
		return 'bg-red-500';
	};

	// File drop handling
	const handleDrop = (event: DragEvent) => {
		event.preventDefault();
		const files = event.dataTransfer?.files;
		if (files && files.length > 0) {
			uploadedFile = files[0];
			handleFileUpload();
		}
	};

	const handleDragOver = (event: DragEvent) => {
		event.preventDefault();
	};
</script>

<div class="smart-document-form max-w-4xl mx-auto p-6 space-y-6">
	<!-- Header -->
	<div class="text-center">
		<h1 class="text-2xl font-bold text-yorha-text-primary mb-2">{title}</h1>
		<p class="text-yorha-text-secondary">{description}</p>
	</div>

	<!-- File Upload Section -->
	{#if enableOCR}
		<Card>
			<CardHeader>
				<CardTitle class="flex items-center space-x-2">
					<span>📄</span>
					<span>Document Upload & Processing</span>
				</CardTitle>
			</CardHeader>
			<CardContent class="space-y-4">
				<!-- Document Type Selection -->
				<div class="flex items-center space-x-4">
					<Label class="text-sm font-medium">Document Type:</Label>
					<select 
						bind:value={selectedDocumentType}
						class="px-3 py-2 bg-yorha-bg-secondary border border-yorha-border rounded-md text-yorha-text-primary"
					>
						<option value="auto">Auto-detect</option>
						{#each documentTypes as type}
							<option value={type}>{type.replace('_', ' ').toUpperCase()}</option>
						{/each}
					</select>
				</div>

				<!-- File Drop Zone -->
				<div 
					class="border-2 border-dashed border-yorha-border rounded-lg p-8 text-center transition-colors duration-200 hover:border-yorha-primary hover:bg-yorha-bg-secondary/50"
					class:border-yorha-primary={uploadedFile}
					ondrop={handleDrop}
					ondragover={handleDragOver}
					role="button"
					tabindex="0"
				>
					{#if uploadedFile}
						<div class="flex items-center justify-center space-x-3">
							<span class="text-2xl">📄</span>
							<div>
								<p class="font-medium text-yorha-text-primary">{uploadedFile.name}</p>
								<p class="text-sm text-yorha-text-secondary">
									{(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
								</p>
							</div>
						</div>
					{:else}
						<div class="space-y-2">
							<span class="text-4xl">📁</span>
							<p class="text-yorha-text-primary">Drop your document here or click to browse</p>
							<p class="text-sm text-yorha-text-secondary">Supports PDF, PNG, JPG, TIFF</p>
						</div>
					{/if}

					<input
						bind:this={fileInput}
						type="file"
						accept=".pdf,.png,.jpg,.jpeg,.tiff"
						class="hidden"
						onchange={(e) => {
							const files = e.target?.files;
							if (files && files.length > 0) {
								uploadedFile = files[0];
								handleFileUpload();
							}
						}}
					/>
					
					{#if !uploadedFile}
						<Button 
							variant="outline" 
							class="mt-4"
							onclick={() => fileInput.click()}
						>
							Browse Files
						</Button>
					{/if}
				</div>

				<!-- Processing Status -->
				{#if $processing}
					<div class="space-y-2" transition:fade>
						<div class="flex items-center justify-between">
							<span class="text-sm text-yorha-text-secondary">Processing document...</span>
							<span class="text-sm text-yorha-text-secondary">{Math.round($progress)}%</span>
						</div>
						<Progress value={$progress} class="h-2" />
					</div>
				{/if}

				<!-- OCR Results Preview -->
				{#if $ocrResult && showPreview}
					<div class="bg-yorha-bg-secondary rounded-md p-4 border border-yorha-border" transitionfly={{ y: 20 }}>
						<div class="flex items-center justify-between mb-2">
							<h4 class="font-medium text-yorha-text-primary">Extraction Results</h4>
							<Badge class="bg-yorha-success text-yorha-bg-primary">
								{$extractedFields.length} fields found
							</Badge>
						</div>
						<div class="text-xs text-yorha-text-secondary">
							Confidence: {Math.round($ocrResult.confidence)}% | 
							Processing Time: {$ocrResult.processingTime}ms |
							Document Type: {$ocrResult.metadata.documentType}
						</div>
					</div>
				{/if}
			</CardContent>
		</Card>
	{/if}

	<!-- Form Fields -->
	<Card>
		<CardHeader>
			<CardTitle class="flex items-center space-x-2">
				<span>📝</span>
				<span>Form Fields</span>
				{#if enableOCR && $extractedFields.length > 0}
					<Badge class="bg-yorha-accent text-yorha-bg-primary">
						Auto-populated
					</Badge>
				{/if}
			</CardTitle>
		</CardHeader>
		<CardContent>
			<form on:submit|preventDefault={handleSubmit} class="space-y-6">
				<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
					{#each populatedFields as field (field.name)}
						<div class="space-y-2" transition:fade>
							<!-- Field Label -->
							<div class="flex items-center justify-between">
								<Label class="flex items-center space-x-2">
									<span class="text-lg">{getFieldTypeIcon(field.type)}</span>
									<span>{field.label}</span>
									{#if field.required}
										<span class="text-yorha-danger">*</span>
									{/if}
								</Label>
								
								<!-- Confidence Indicator -->
								{#if field.confidence}
									<div class="flex items-center space-x-1">
										<div class={`w-2 h-2 rounded-full ${getConfidenceColor(field.confidence)}`}></div>
										<span class="text-xs text-yorha-text-secondary">
											{Math.round(field.confidence * 100)}%
										</span>
									</div>
								{/if}
							</div>

							<!-- Field Input -->
							{#if field.type === 'text_block' && field.name.includes('notes')}
								<Textarea
									bind:value={field.value}
									placeholder={`Enter ${field.label.toLowerCase()}...`}
									class="min-h-[80px] bg-yorha-bg-secondary border-yorha-border text-yorha-text-primary"
									oninput={(e) => handleFieldChange(field.name, e.target.value)}
								/>
							{:else}
								<Input
									type={field.type === 'email' ? 'email' : field.type === 'phone' ? 'tel' : 'text'}
									bind:value={field.value}
									placeholder={`Enter ${field.label.toLowerCase()}...`}
									class="bg-yorha-bg-secondary border-yorha-border text-yorha-text-primary"
									class:border-yorha-danger={$formErrors[field.name]}
									class:border-yorha-success={field.confidence && field.confidence > 0.8}
									oninput={(e) => handleFieldChange(field.name, e.target.value)}
								/>
							{/if}

							<!-- Field Error -->
							{#if $formErrors[field.name]}
								<p class="text-xs text-yorha-danger" transition:scale>
									{$formErrors[field.name]}
								</p>
							{/if}

							<!-- Smart Suggestions -->
							{#if activeSuggestions[field.name] && activeSuggestions[field.name].length > 0}
								<div class="space-y-1" transitionfly={{ y: -10 }}>
									<p class="text-xs text-yorha-text-secondary">Suggestions:</p>
									<div class="flex flex-wrap gap-1">
										{#each activeSuggestions[field.name] as suggestion}
											<Button
												variant="outline"
												size="sm"
												class="text-xs h-6 px-2"
												onclick={() => applySuggestion(field.name, suggestion)}
											>
												{suggestion}
											</Button>
										{/each}
									</div>
								</div>
							{/if}

							<!-- Loading Suggestions -->
							{#if suggestionLoading[field.name]}
								<div class="flex items-center space-x-2 text-xs text-yorha-text-secondary">
									<div class="animate-spin w-3 h-3 border border-yorha-accent border-t-transparent rounded-full"></div>
									<span>Generating suggestions...</span>
								</div>
							{/if}
						</div>
					{/each}
				</div>

				<!-- Form Actions -->
				<div class="flex items-center justify-between pt-6 border-t border-yorha-border">
					<div class="flex items-center space-x-4">
						<Badge class={isFormValid ? 'bg-yorha-success' : 'bg-yorha-warning'}>
							{isFormValid ? 'Ready to Submit' : 'Incomplete'}
						</Badge>
						
						{#if enableOCR && $extractedFields.length > 0}
							<span class="text-xs text-yorha-text-secondary">
								{populatedFields.filter(f => f.value).length} / {populatedFields.length} fields completed
							</span>
						{/if}
					</div>

					<div class="flex items-center space-x-3">
						<Button 
							variant="outline"
							onclick={() => {
								populatedFields = populatedFields.map(f => ({ ...f, value: '' }));
								formErrors.set({});
							}}
						>
							Clear All
						</Button>
						
						<Button 
							type="submit"
							disabled={!isFormValid}
							class="bg-yorha-primary hover:bg-yorha-primary/80 disabled:opacity-50"
						>
							Submit Form
						</Button>
					</div>
				</div>
			</form>
		</CardContent>
	</Card>

	<!-- Extracted Fields Preview -->
	{#if $extractedFields.length > 0 && showPreview}
		<Card>
			<CardHeader>
				<CardTitle class="flex items-center space-x-2">
					<span>🔍</span>
					<span>Extracted Data</span>
					<Button
						variant="ghost"
						size="sm"
						onclick={() => showPreview = !showPreview}
					>
						{showPreview ? 'Hide' : 'Show'}
					</Button>
				</CardTitle>
			</CardHeader>
			<CardContent>
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
					{#each $extractedFields as field (field.fieldName)}
						<div class="bg-yorha-bg-secondary rounded p-3 border border-yorha-border">
							<div class="flex items-center justify-between mb-1">
								<span class="text-sm font-medium text-yorha-text-primary">
									{field.fieldName.replace('_', ' ')}
								</span>
								<Badge 
									class={
										field.validationStatus === 'valid' ? 'bg-yorha-success' :
										field.validationStatus === 'invalid' ? 'bg-yorha-danger' :
										'bg-yorha-warning'
									}
								>
									{field.validationStatus}
								</Badge>
							</div>
							<p class="text-sm text-yorha-text-secondary mb-1">{field.value}</p>
							<div class="flex items-center justify-between text-xs text-yorha-text-tertiary">
								<span>{field.fieldType}</span>
								<span>{Math.round(field.confidence * 100)}%</span>
							</div>
						</div>
					{/each}
				</div>
			</CardContent>
		</Card>
	{/if}
</div>

<style>
	.smart-document-form {
		background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
		min-height: 100vh;
	}
</style>