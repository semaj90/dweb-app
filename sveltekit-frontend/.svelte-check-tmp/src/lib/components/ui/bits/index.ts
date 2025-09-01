// Modern Bits-UI Components for Legal AI App
// Built with Svelte 5 + bits-ui primitives

// Core Components
export { default as BitsDialog } from '../dialog/BitsDialog.svelte';
export { default as BitsSelect } from '../select/BitsSelect.svelte';  
export { default as BitsInput } from '../input/BitsInput.svelte';
export { default as BitsCombobox } from '../combobox/BitsCombobox.svelte';
export { default as BitsDatePicker } from '../date-picker/BitsDatePicker.svelte';
export { default as BitsDataTable } from '../data-table/BitsDataTable.svelte';

// Toast System
export { default as BitsToast } from '../toast/BitsToast.svelte';
export { default as ToastProvider } from '../toast/ToastProvider.svelte';

// Legal Domain Components
export { default as CaseTimeline } from '../../legal/CaseTimeline.svelte';
export { default as ChainOfCustodyTracker } from '../../legal/ChainOfCustodyTracker.svelte';
export { default as LegalPrecedentCard } from '../../legal/LegalPrecedentCard.svelte';
export { default as CriminalProfile } from '../../legal/CriminalProfile.svelte';

// Demo Component
export { default as BitsUIDemo } from '../BitsUIDemo.svelte';

// Re-export existing Svelte 5 compatible components
export { default as Button } from '../button/Button.svelte';

// Types
export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface ComboboxOption {
  value: string;
  label: string;
  description?: string;
  category?: string;
  disabled?: boolean;
  metadata?: Record<string, any>;
}

export interface DataTableColumn<T = any> {
  key: string;
  label: string;
  sortable?: boolean;
  filterable?: boolean;
  render?: (value: any, row: T) => string;
  class?: string;
  width?: string;
}

export interface InputVariant {
  variant: 'default' | 'legal' | 'search' | 'ai';
}

export interface ComponentSize {
  size: 'sm' | 'md' | 'lg';
}

// Component status and capabilities
export const BITS_UI_STATUS = {
  components: {
    BitsDialog: { 
      status: 'stable',
      features: ['accessibility', 'keyboard-nav', 'focus-trap', 'legal-styling'],
      dependencies: ['bits-ui/Dialog']
    },
    BitsSelect: {
      status: 'stable', 
      features: ['searchable', 'keyboard-nav', 'legal-options', 'form-integration'],
      dependencies: ['bits-ui/Select']
    },
    BitsInput: {
      status: 'stable',
      features: ['variants', 'icons', 'validation', 'legal-styling'],
      dependencies: []
    },
    BitsCombobox: {
      status: 'stable',
      features: ['searchable', 'creatable', 'categories', 'multiple-selection', 'legal-entities'],
      dependencies: ['bits-ui/Combobox']
    },
    BitsDatePicker: {
      status: 'stable',
      features: ['time-picker', 'deadline-warnings', 'legal-deadlines', 'accessibility'],
      dependencies: ['bits-ui/DatePicker']
    },
    BitsDataTable: {
      status: 'stable',
      features: ['sorting', 'filtering', 'pagination', 'selection', 'export', 'evidence-management'],
      dependencies: []
    },
    BitsToast: {
      status: 'stable',
      features: ['variants', 'actions', 'legal-notifications', 'case-updates'],
      dependencies: ['bits-ui/Toast']
    },
    Button: {
      status: 'stable',
      features: ['variants', 'sizes', 'loading-states', 'svelte5-props'],
      dependencies: []
    }
  },
  legalComponents: {
    CaseTimeline: {
      status: 'stable',
      features: ['timeline-visualization', 'event-tracking', 'legal-milestones'],
      dependencies: []
    },
    ChainOfCustodyTracker: {
      status: 'stable',
      features: ['evidence-integrity', 'transfer-tracking', 'custody-verification'],
      dependencies: []
    },
    LegalPrecedentCard: {
      status: 'stable',
      features: ['precedent-analysis', 'relevance-scoring', 'citation-tracking'],
      dependencies: []
    },
    CriminalProfile: {
      status: 'stable',
      features: ['criminal-records', 'risk-assessment', 'biometric-data', 'warrant-tracking'],
      dependencies: []
    }
  },
  svelte5: {
    compatible: true,
    patterns: ['$props()', '$bindable()', '$derived', '$state()'],
    migration: 'complete'
  },
  bitsUI: {
    version: '2.x',
    primitives: ['Dialog', 'Select', 'Button', 'Input', 'Combobox', 'DatePicker', 'Toast'],
    accessibility: 'WCAG 2.1 AA compliant'
  }
} as const;