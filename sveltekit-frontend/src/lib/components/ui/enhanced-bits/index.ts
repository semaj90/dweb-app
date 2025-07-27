// Enhanced Bits UI v2 + Svelte 5 + UnoCSS Integration
// Optimized for Legal AI with NieR theming

export { default as Button } from './Button.svelte';
export { default as Dialog } from './Dialog.svelte';
export { default as Select } from './Select.svelte';
export { default as Input } from './Input.svelte';
export { default as Card } from './Card.svelte';

// Demo components
export { default as EnhancedBitsDemo } from './EnhancedBitsDemo.svelte';
export { default as VectorIntelligenceDemo } from './VectorIntelligenceDemo.svelte';

// Types and utilities
export type * from './types';
export * from '../enhanced/button-variants';
export { cn, legalCn, confidenceClass, priorityClass } from '$lib/utils/cn';

// Legal AI specific types
export interface EvidenceItem {
	id: string;
	title: string;
	type: 'document' | 'image' | 'video' | 'audio' | 'transcript';
	priority: 'critical' | 'high' | 'medium' | 'low';
	confidence: number;
	metadata?: Record<string, unknown>;
	createdAt: Date;
	updatedAt: Date;
}

export interface CaseData {
	id: string;
	title: string;
	type: string;
	status: 'active' | 'closed' | 'pending';
	evidence: EvidenceItem[];
	priority: 'critical' | 'high' | 'medium' | 'low';
	assignedTo?: string;
}

export interface AIAnalysis {
	confidence: number;
	entities: Array<{
		text: string;
		type: string;
		confidence: number;
	}>;
	themes: Array<{
		topic: string;
		weight: number;
	}>;
	summary: string;
}

// Legal domain constants
export const EVIDENCE_TYPES = [
	'document',
	'image', 
	'video',
	'audio',
	'transcript'
] as const;

export const PRIORITY_LEVELS = [
	'critical',
	'high', 
	'medium',
	'low'
] as const;

export const CASE_TYPES = [
	'criminal',
	'civil',
	'corporate',
	'employment',
	'intellectual_property'
] as const;