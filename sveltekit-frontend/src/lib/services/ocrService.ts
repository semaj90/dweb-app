// @ts-nocheck
/**
 * OCR Service with Auto-Population
 * Handles document OCR, field extraction, and form auto-population
 */

import { writable, type Writable } from "svelte/store";
// Orphaned content: import {
createWorker, type Worker, PSM
import { z } from 'zod';

// OCR Types
export interface OCRResult {
	id: string;
	text: string;
	confidence: number;
	boundingBoxes: BoundingBox[];
	extractedFields: ExtractedField[];
	metadata: OCRMetadata;
	processingTime: number;
}

export interface BoundingBox {
	x: number;
	y: number;
	width: number;
	height: number;
	text: string;
	confidence: number;
}

export interface ExtractedField {
	fieldName: string;
	value: string;
	confidence: number;
	boundingBox: BoundingBox;
	fieldType: FieldType;
	validationStatus: 'valid' | 'invalid' | 'needs_review';
}

export type FieldType = 
	| 'name' 
	| 'date' 
	| 'address' 
	| 'phone' 
	| 'email' 
	| 'case_number' 
	| 'jurisdiction'
	| 'monetary_amount'
	| 'legal_entity'
	| 'signature'
	| 'checkbox'
	| 'text_block';

export interface OCRMetadata {
	filename: string;
	fileSize: number;
	dimensions: { width: number; height: number };
	pageCount: number;
	language: string;
	documentType: string;
	processingDate: number;
}

export interface FormField {
	name: string;
	type: FieldType;
	label: string;
	value?: string;
	confidence?: number;
	required: boolean;
	validation?: z.ZodSchema;
	suggestions?: string[];
}

// Field extraction patterns for different document types
const FIELD_PATTERNS = {
	legal_document: {
		case_number: /case\s*(?:no\.?|number)?\s*:?\s*([A-Z0-9-]+)/i,
		date: /(?:date|filed|executed):\s*(\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2})/i,
		jurisdiction: /(?:jurisdiction|court|county):\s*([^,\n]+)/i,
		plaintiff: /plaintiff:\s*([^,\n]+)/i,
		defendant: /defendant:\s*([^,\n]+)/i,
		attorney: /attorney\s*(?:for)?:\s*([^,\n]+)/i,
		amount: /(?:amount|damages?):\s*\$?([\d,]+\.?\d*)/i
	},
	contract: {
		party_1: /(?:party|contractor)\s*(?:1|one|first)?:\s*([^,\n]+)/i,
		party_2: /(?:party|contractor)\s*(?:2|two|second)?:\s*([^,\n]+)/i,
		effective_date: /effective\s*date:\s*(\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2})/i,
		term: /term:\s*([^,\n]+)/i,
		amount: /(?:amount|payment|fee):\s*\$?([\d,]+\.?\d*)/i,
		signature_1: /signature.*?([A-Za-z\s]{2,30})/i,
		signature_2: /signature.*?([A-Za-z\s]{2,30})/i
	},
	form: {
		name: /(?:name|full\s*name):\s*([A-Za-z\s]{2,50})/i,
		address: /(?:address|street):\s*([^,\n]{5,100})/i,
		city: /city:\s*([A-Za-z\s]{2,50})/i,
		state: /state:\s*([A-Za-z]{2,20})/i,
		zip: /(?:zip|postal):\s*(\d{5}(?:-\d{4})?)/i,
		phone: /(?:phone|tel):\s*(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})/i,
		email: /email:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/i,
		ssn: /(?:ssn|social):\s*(\d{3}-?\d{2}-?\d{4})/i
	}
};

// Validation schemas
const fieldValidationSchemas = {
	email: z.string().email('Invalid email format'),
	phone: z.string().regex(/^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$/, 'Invalid phone format'),
	date: z.string().regex(/^\d{1,2}\/\d{1,2}\/\d{4}$|^\d{4}-\d{2}-\d{2}$/, 'Invalid date format'),
	monetary_amount: z.string().regex(/^\$?[\d,]+\.?\d*$/, 'Invalid monetary amount'),
	case_number: z.string().min(3, 'Case number too short'),
	name: z.string().min(2, 'Name too short').max(100, 'Name too long')
};

export class OCRService {
	private worker: Worker | null = null;
	private isInitialized = false;

	// Reactive stores
	public processing$: Writable<boolean> = writable(false);
	public progress$: Writable<number> = writable(0);
	public currentResult$: Writable<OCRResult | null> = writable(null);
	public extractedFields$: Writable<ExtractedField[]> = writable([]);

	constructor() {
		this.initializeWorker();
	}

	private async initializeWorker(): Promise<void> {
		try {
			console.log('🔧 Initializing OCR worker...');
			
			this.worker = await createWorker('eng', 1, {
				logger: (m) => {
					if (m.status === 'recognizing text') {
						this.progress$.set(m.progress * 100);
					}
				}
			});

			// Configure worker for better legal document recognition
			await this.worker.setParameters({
				tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?-()[]{}/@#$%^&*+=<>|\\~`"\' \n\t',
				preserve_interword_spaces: '1',
				tessedit_pageseg_mode: PSM.AUTO_OSD // Automatic page segmentation with OSD
			});

			this.isInitialized = true;
			console.log('✅ OCR worker initialized');
		} catch (error) {
			console.error('❌ OCR worker initialization failed:', error);
		}
	}

	/**
	 * Main OCR processing method
	 */
	public async processDocument(
		file: File,
		options: {
			documentType?: 'legal_document' | 'contract' | 'form' | 'auto';
			extractFields?: boolean;
			language?: string;
			qualityEnhancement?: boolean;
		} = {}
	): Promise<OCRResult> {
		if (!this.isInitialized || !this.worker) {
			throw new Error('OCR worker not initialized');
		}

		const startTime = Date.now();
		this.processing$.set(true);
		this.progress$.set(0);

		try {
			console.log(`🔍 Processing document: ${file.name}`);

			// Step 1: Image preprocessing (if enabled)
			let processedImage = file;
			if (options.qualityEnhancement) {
				processedImage = await this.enhanceImageQuality(file);
			}

			// Step 2: OCR recognition
			const { data } = await this.worker.recognize(processedImage);
			
			// Step 3: Extract bounding boxes
			const boundingBoxes = this.extractBoundingBoxes(data);
			
			// Step 4: Determine document type
			const documentType = options.documentType === 'auto' 
				? await this.detectDocumentType(data.text)
				: options.documentType || 'form';

			// Step 5: Extract fields based on document type
			const extractedFields = options.extractFields !== false 
				? await this.extractFields(data.text, documentType, boundingBoxes)
				: [];

			// Step 6: Build result
			const result: OCRResult = {
				id: `ocr_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
				text: data.text,
				confidence: data.confidence,
				boundingBoxes,
				extractedFields,
				metadata: {
					filename: file.name,
					fileSize: file.size,
					dimensions: { width: (data as any).words?.[0]?.bbox?.x1 || 0, height: (data as any).words?.[0]?.bbox?.y1 || 0 },
					pageCount: 1, // TODO: Multi-page support
					language: options.language || 'eng',
					documentType,
					processingDate: Date.now()
				},
				processingTime: Date.now() - startTime
			};

			// Update stores
			this.currentResult$.set(result);
			this.extractedFields$.set(extractedFields);
			this.processing$.set(false);
			this.progress$.set(100);

			console.log(`✅ OCR completed in ${result.processingTime}ms with ${extractedFields.length} fields extracted`);
			return result;

		} catch (error) {
			console.error('❌ OCR processing failed:', error);
			this.processing$.set(false);
			throw error;
		}
	}

	/**
	 * Enhanced image preprocessing for better OCR accuracy
	 */
	private async enhanceImageQuality(file: File): Promise<File> {
		return new Promise((resolve) => {
			const canvas = document.createElement('canvas');
			const ctx = canvas.getContext('2d')!;
			const img = new Image();

			img.onload = () => {
				// Set canvas size
				canvas.width = img.width;
				canvas.height = img.height;

				// Draw original image
				ctx.drawImage(img, 0, 0);

				// Apply enhancements
				const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
				const data = imageData.data;

				// Increase contrast and brightness
				for (let i = 0; i < data.length; i += 4) {
					// Contrast adjustment
					data[i] = Math.min(255, Math.max(0, (data[i] - 128) * 1.2 + 128));     // Red
					data[i + 1] = Math.min(255, Math.max(0, (data[i + 1] - 128) * 1.2 + 128)); // Green
					data[i + 2] = Math.min(255, Math.max(0, (data[i + 2] - 128) * 1.2 + 128)); // Blue
					
					// Brightness adjustment
					data[i] = Math.min(255, data[i] + 10);
					data[i + 1] = Math.min(255, data[i + 1] + 10);
					data[i + 2] = Math.min(255, data[i + 2] + 10);
				}

				ctx.putImageData(imageData, 0, 0);

				// Convert back to blob
				canvas.toBlob((blob) => {
					if (blob) {
						const enhancedFile = new File([blob], file.name, { type: file.type });
						resolve(enhancedFile);
					} else {
						resolve(file); // Fallback to original
					}
				}, file.type);
			};

			img.onerror = () => resolve(file); // Fallback to original
			img.src = URL.createObjectURL(file);
		});
	}

	/**
	 * Extract bounding box information
	 */
	private extractBoundingBoxes(data: any): BoundingBox[] {
		const boxes: BoundingBox[] = [];

		if (data.words) {
			for (const word of data.words) {
				boxes.push({
					x: word.bbox.x0,
					y: word.bbox.y0,
					width: word.bbox.x1 - word.bbox.x0,
					height: word.bbox.y1 - word.bbox.y0,
					text: word.text,
					confidence: word.confidence
				});
			}
		}

		return boxes;
	}

	/**
	 * Detect document type using LLM
	 */
	private async detectDocumentType(text: string): Promise<string> {
		try {
			const prompt = `
Analyze this document text and determine its type:

Text: "${text.substring(0, 1000)}..."

Classify as one of:
- legal_document (court filings, legal briefs, motions)
- contract (agreements, contracts, terms of service)
- form (applications, intake forms, questionnaires)

Return only the classification: legal_document, contract, or form`;

			const response = await fetch('http://localhost:11434/api/generate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: 'gemma3-legal',
					prompt,
					stream: false
				})
			});

			const data = await response.json();
			const result = data.response.trim().toLowerCase();
			
			return ['legal_document', 'contract', 'form'].includes(result) ? result : 'form';
		} catch (error) {
			console.warn('Document type detection failed:', error);
			return 'form';
		}
	}

	/**
	 * Extract structured fields from OCR text
	 */
	private async extractFields(
		text: string, 
		documentType: string, 
		boundingBoxes: BoundingBox[]
	): Promise<ExtractedField[]> {
		const fields: ExtractedField[] = [];
		const patterns = FIELD_PATTERNS[documentType as keyof typeof FIELD_PATTERNS] || FIELD_PATTERNS.form;

		// Pattern-based extraction
		for (const [fieldName, pattern] of Object.entries(patterns)) {
			const match = text.match(pattern);
			if (match && match[1]) {
				const value = match[1].trim();
				const fieldType = this.determineFieldType(fieldName, value);
				const boundingBox = this.findBoundingBox(value, boundingBoxes);
				
				fields.push({
					fieldName,
					value,
					confidence: boundingBox?.confidence || 0.8,
					boundingBox: boundingBox || { x: 0, y: 0, width: 0, height: 0, text: value, confidence: 0.8 },
					fieldType,
					validationStatus: this.validateField(fieldType, value)
				});
			}
		}

		// LLM-based enhancement for missed fields
		const enhancedFields = await this.enhanceFieldsWithLLM(text, documentType, fields);
		
		return [...fields, ...enhancedFields];
	}

	/**
	 * Determine field type based on field name and value
	 */
	private determineFieldType(fieldName: string, value: string): FieldType {
		// Email detection
		if (fieldName.includes('email') || value.includes('@')) return 'email';
		
		// Phone detection
		if (fieldName.includes('phone') || /\d{3}[-.\s]?\d{3}[-.\s]?\d{4}/.test(value)) return 'phone';
		
		// Date detection
		if (fieldName.includes('date') || /\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2}/.test(value)) return 'date';
		
		// Monetary amount detection
		if (fieldName.includes('amount') || /^\$?[\d,]+\.?\d*$/.test(value)) return 'monetary_amount';
		
		// Case number detection
		if (fieldName.includes('case') || fieldName.includes('number')) return 'case_number';
		
		// Name detection
		if (fieldName.includes('name') || fieldName.includes('plaintiff') || fieldName.includes('defendant')) return 'name';
		
		// Address detection
		if (fieldName.includes('address') || fieldName.includes('street')) return 'address';
		
		// Default to text block
		return 'text_block';
	}

	/**
	 * Find bounding box for extracted text
	 */
	private findBoundingBox(text: string, boundingBoxes: BoundingBox[]): BoundingBox | null {
		// Find the bounding box that contains this text
		for (const box of boundingBoxes) {
			if (box.text.toLowerCase().includes(text.toLowerCase()) || 
				text.toLowerCase().includes(box.text.toLowerCase())) {
				return box;
			}
		}
		return null;
	}

	/**
	 * Validate extracted field value
	 */
	private validateField(fieldType: FieldType, value: string): 'valid' | 'invalid' | 'needs_review' {
		const schema = fieldValidationSchemas[fieldType as keyof typeof fieldValidationSchemas];
		
		if (!schema) return 'needs_review';
		
		try {
			schema.parse(value);
			return 'valid';
		} catch {
			return 'invalid';
		}
	}

	/**
	 * Enhance field extraction using LLM
	 */
	private async enhanceFieldsWithLLM(
		text: string, 
		documentType: string, 
		existingFields: ExtractedField[]
	): Promise<ExtractedField[]> {
		try {
			const existingFieldNames = existingFields.map((f: any) => f.fieldName);
			
			const prompt = `
Analyze this ${documentType} and extract additional structured fields that may have been missed:

Document text: "${text.substring(0, 2000)}..."

Already extracted: ${existingFieldNames.join(', ')}

Find additional fields like names, dates, addresses, phone numbers, email addresses, case numbers, monetary amounts, signatures, etc.

Return JSON array:
[
  {"fieldName": "field1", "value": "extracted_value", "fieldType": "name", "confidence": 0.85},
  {"fieldName": "field2", "value": "extracted_value", "fieldType": "date", "confidence": 0.92}
]`;

			const response = await fetch('http://localhost:11434/api/generate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: 'gemma3-legal',
					prompt,
					stream: false,
					format: 'json'
				})
			});

			const data = await response.json();
			const enhancedFields = JSON.parse(data.response);

			return enhancedFields.map((field: any) => ({
				fieldName: field.fieldName,
				value: field.value,
				confidence: field.confidence || 0.7,
				boundingBox: { x: 0, y: 0, width: 0, height: 0, text: field.value, confidence: field.confidence },
				fieldType: field.fieldType || 'text_block',
				validationStatus: this.validateField(field.fieldType, field.value)
			}));

		} catch (error) {
			console.warn('LLM field enhancement failed:', error);
			return [];
		}
	}

	/**
	 * Auto-populate form fields from extracted data
	 */
	public autoPopulateForm(extractedFields: ExtractedField[], formSchema: FormField[]): FormField[] {
		const populatedFields = [...formSchema];

		for (const formField of populatedFields) {
			// Find matching extracted field
			const match = extractedFields.find((extracted: any) => extracted.fieldName.toLowerCase().includes(formField.name.toLowerCase()) ||
				formField.name.toLowerCase().includes(extracted.fieldName.toLowerCase()) ||
				extracted.fieldType === formField.type
			);

			if (match && match.validationStatus !== 'invalid') {
				formField.value = match.value;
				formField.confidence = match.confidence;
				
				// Add suggestions for review
				if (match.validationStatus === 'needs_review') {
					formField.suggestions = [match.value, `Review: ${match.value}`];
				}
			}
		}

		return populatedFields;
	}

	/**
	 * Get suggestions for incomplete fields
	 */
	public async getSuggestions(fieldName: string, fieldType: FieldType, context: string): Promise<string[]> {
		try {
			const prompt = `
Based on this document context, suggest possible values for the field "${fieldName}" of type "${fieldType}":

Context: "${context.substring(0, 500)}..."

Provide 3-5 realistic suggestions as a JSON array: ["suggestion1", "suggestion2", ...]`;

			const response = await fetch('http://localhost:11434/api/generate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: 'gemma3-legal',
					prompt,
					stream: false,
					format: 'json'
				})
			});

			const data = await response.json();
			return JSON.parse(data.response);
		} catch (error) {
			console.warn('Suggestion generation failed:', error);
			return [];
		}
	}

	/**
	 * Cleanup resources
	 */
	public async destroy(): Promise<void> {
		if (this.worker) {
			await this.worker.terminate();
			this.worker = null;
		}
		this.isInitialized = false;
		console.log('🔧 OCR service destroyed');
	}
}

// Singleton instance
export const ocrService = new OCRService();