import type { CommonProps } from '$lib/types/common-props';
// YoRHa Table and Grid Components
export { default as YoRHaTable } from './YoRHaTable.svelte';
export { default as YoRHaDataGrid } from './YoRHaDataGrid.svelte';

// Table utilities and stores
export { 
  tableManager,
  createTableStats,
  formatTableData,
  exportTableData,
  legalAITableConfigs,
  type TableState,
  type TableNotification
} from '$lib/stores/tables';

// Table-related types for components
export interface YoRHaTableColumn {
  key: string;
  title: string;
  sortable?: boolean;
  width?: string;
  align?: 'left' | 'center' | 'right';
  type?: 'text' | 'number' | 'date' | 'status' | 'action';
}

export interface YoRHaTableRow {
  id: string | number;
  [key: string]: any;
}

export interface YoRHaGridColumn {
  key: string;
  title: string;
  width?: number;
  minWidth?: number;
  maxWidth?: number;
  resizable?: boolean;
  sortable?: boolean;
  filterable?: boolean;
  editable?: boolean;
  type?: 'text' | 'number' | 'date' | 'boolean' | 'select' | 'action';
  options?: Array<{ label: string; value: any }>;
  formatter?: (value: any, row: any) => string;
  validator?: (value: any) => boolean | string;
}

// Legal AI specific table configurations and utilities
export const yorhaTablePresets = {
  // Legal cases table preset
  legalCases: {
    columns: [
      { key: 'case_number', title: 'CASE ID', sortable: true, width: '120px' },
      { key: 'title', title: 'CASE TITLE', sortable: true, width: '300px' },
      { key: 'status', title: 'STATUS', sortable: true, type: 'status' as const, width: '120px' },
      { key: 'priority', title: 'PRIORITY', sortable: true, type: 'status' as const, width: '100px' },
      { key: 'assigned_to', title: 'ASSIGNED TO', sortable: true, width: '150px' },
      { key: 'created_at', title: 'CREATED', sortable: true, type: 'date' as const, width: '120px' },
      { key: 'actions', title: 'ACTIONS', type: 'action' as const, width: '150px' }
    ] as YoRHaTableColumn[],
    config: {
      selectable: true,
      sortable: true,
      pagination: true,
      pageSize: 25,
      hover: true,
      striped: true,
      bordered: true,
      glitchEffect: false
    }
  },

  // Evidence table preset
  evidence: {
    columns: [
      { key: 'evidence_id', title: 'EVIDENCE ID', sortable: true, width: '140px' },
      { key: 'title', title: 'TITLE', sortable: true, width: '250px' },
      { key: 'type', title: 'TYPE', sortable: true, type: 'status' as const, width: '100px' },
      { key: 'case_id', title: 'CASE', sortable: true, width: '120px' },
      { key: 'collected_by', title: 'COLLECTED BY', sortable: true, width: '150px' },
      { key: 'date_collected', title: 'COLLECTED', sortable: true, type: 'date' as const, width: '120px' },
      { key: 'status', title: 'STATUS', sortable: true, type: 'status' as const, width: '100px' },
      { key: 'actions', title: 'ACTIONS', type: 'action' as const, width: '150px' }
    ] as YoRHaTableColumn[],
    config: {
      selectable: true,
      sortable: true,
      pagination: true,
      pageSize: 50,
      hover: true,
      striped: true,
      bordered: true,
      dense: false
    }
  },

  // Documents table preset
  documents: {
    columns: [
      { key: 'filename', title: 'FILENAME', sortable: true, width: '200px' },
      { key: 'type', title: 'TYPE', sortable: true, type: 'status' as const, width: '100px' },
      { key: 'size', title: 'SIZE', sortable: true, type: 'number' as const, width: '100px' },
      { key: 'case_id', title: 'CASE', sortable: true, width: '120px' },
      { key: 'uploaded_by', title: 'UPLOADED BY', sortable: true, width: '150px' },
      { key: 'upload_date', title: 'UPLOADED', sortable: true, type: 'date' as const, width: '120px' },
      { key: 'status', title: 'STATUS', sortable: true, type: 'status' as const, width: '100px' },
      { key: 'actions', title: 'ACTIONS', type: 'action' as const, width: '150px' }
    ] as YoRHaTableColumn[],
    config: {
      selectable: true,
      sortable: true,
      pagination: true,
      pageSize: 20,
      hover: true,
      striped: true,
      bordered: true
    }
  },

  // Users table preset
  users: {
    columns: [
      { key: 'username', title: 'USERNAME', sortable: true, width: '150px' },
      { key: 'full_name', title: 'FULL NAME', sortable: true, width: '200px' },
      { key: 'role', title: 'ROLE', sortable: true, type: 'status' as const, width: '120px' },
      { key: 'email', title: 'EMAIL', sortable: true, width: '200px' },
      { key: 'status', title: 'STATUS', sortable: true, type: 'status' as const, width: '100px' },
      { key: 'last_login', title: 'LAST LOGIN', sortable: true, type: 'date' as const, width: '140px' },
      { key: 'actions', title: 'ACTIONS', type: 'action' as const, width: '150px' }
    ] as YoRHaTableColumn[],
    config: {
      selectable: true,
      sortable: true,
      pagination: true,
      pageSize: 30,
      hover: true,
      striped: true,
      bordered: true
    }
  },

  // Audit log table preset
  auditLog: {
    columns: [
      { key: 'timestamp', title: 'TIMESTAMP', sortable: true, type: 'date' as const, width: '140px' },
      { key: 'user', title: 'USER', sortable: true, width: '120px' },
      { key: 'action', title: 'ACTION', sortable: true, type: 'status' as const, width: '150px' },
      { key: 'resource', title: 'RESOURCE', sortable: true, width: '150px' },
      { key: 'resource_id', title: 'RESOURCE ID', sortable: true, width: '120px' },
      { key: 'ip_address', title: 'IP ADDRESS', sortable: true, width: '130px' },
      { key: 'status', title: 'STATUS', sortable: true, type: 'status' as const, width: '100px' }
    ] as YoRHaTableColumn[],
    config: {
      selectable: false,
      sortable: true,
      pagination: true,
      pageSize: 100,
      hover: true,
      striped: true,
      bordered: true,
      dense: true
    }
  }
};

// Grid presets for more complex data management
export const yorhaGridPresets = {
  // Editable case management grid
  caseManagement: {
    columns: [
      { key: 'case_number', title: 'CASE ID', sortable: true, filterable: true, width: 120 },
      { key: 'title', title: 'TITLE', sortable: true, filterable: true, editable: true, width: 300 },
      { key: 'description', title: 'DESCRIPTION', editable: true, width: 400 },
      { 
        key: 'status', 
        title: 'STATUS', 
        sortable: true, 
        filterable: true, 
        editable: true, 
        type: 'select' as const,
        options: [
          { label: 'ACTIVE', value: 'active' },
          { label: 'INACTIVE', value: 'inactive' },
          { label: 'CLOSED', value: 'closed' },
          { label: 'PENDING', value: 'pending' }
        ],
        width: 120
      },
      {
        key: 'priority',
        title: 'PRIORITY',
        sortable: true,
        filterable: true,
        editable: true,
        type: 'select' as const,
        options: [
          { label: 'LOW', value: 'low' },
          { label: 'MEDIUM', value: 'medium' },
          { label: 'HIGH', value: 'high' },
          { label: 'CRITICAL', value: 'critical' }
        ],
        width: 100
      },
      { key: 'assigned_to', title: 'ASSIGNED TO', sortable: true, filterable: true, editable: true, width: 150 },
      { key: 'created_at', title: 'CREATED', sortable: true, type: 'date' as const, width: 120 },
      { key: 'actions', title: 'ACTIONS', type: 'action' as const, width: 150 }
    ] as YoRHaGridColumn[],
    config: {
      editable: true,
      selectable: true,
      multiSelect: true,
      sortable: true,
      filterable: true,
      resizable: true,
      virtualScroll: false,
      rowHeight: 50,
      maxHeight: 600
    }
  },

  // Evidence analysis grid
  evidenceAnalysis: {
    columns: [
      { key: 'evidence_id', title: 'ID', sortable: true, filterable: true, width: 100 },
      { key: 'filename', title: 'FILENAME', sortable: true, filterable: true, width: 200 },
      { key: 'type', title: 'TYPE', sortable: true, filterable: true, width: 100 },
      { key: 'ai_processed', title: 'AI ANALYZED', sortable: true, type: 'boolean' as const, width: 120 },
      { key: 'confidence_score', title: 'CONFIDENCE', sortable: true, type: 'number' as const, width: 120 },
      { key: 'relevance_score', title: 'RELEVANCE', sortable: true, type: 'number' as const, width: 120 },
      { key: 'tags', title: 'TAGS', filterable: true, width: 200 },
      { key: 'status', title: 'STATUS', sortable: true, filterable: true, type: 'select' as const, width: 120 },
      { key: 'actions', title: 'ACTIONS', type: 'action' as const, width: 200 }
    ] as YoRHaGridColumn[],
    config: {
      editable: false,
      selectable: true,
      multiSelect: true,
      sortable: true,
      filterable: true,
      resizable: true,
      virtualScroll: true,
      rowHeight: 45,
      maxHeight: 700
    }
  }
};

// Utility functions for YoRHa tables
export function createLegalAITable(preset: keyof typeof yorhaTablePresets) {
  return yorhaTablePresets[preset];
}

export function createLegalAIGrid(preset: keyof typeof yorhaGridPresets) {
  return yorhaGridPresets[preset];
}

// Status formatters for legal AI
export const statusFormatters = {
  caseStatus: (status: string) => {
    const statusMap: Record<string, string> = {
      'active': 'ACTIVE',
      'inactive': 'INACTIVE', 
      'closed': 'CLOSED',
      'pending': 'PENDING',
      'archived': 'ARCHIVED'
    };
    return statusMap[status?.toLowerCase()] || status?.toUpperCase() || 'UNKNOWN';
  },

  priority: (priority: string) => {
    const priorityMap: Record<string, string> = {
      'low': 'LOW',
      'medium': 'MEDIUM',
      'high': 'HIGH',
      'critical': 'CRITICAL'
    };
    return priorityMap[priority?.toLowerCase()] || priority?.toUpperCase() || 'NORMAL';
  },

  evidenceType: (type: string) => {
    const typeMap: Record<string, string> = {
      'document': 'DOCUMENT',
      'image': 'IMAGE',
      'video': 'VIDEO',
      'audio': 'AUDIO',
      'digital': 'DIGITAL',
      'physical': 'PHYSICAL'
    };
    return typeMap[type?.toLowerCase()] || type?.toUpperCase() || 'OTHER';
  },

  userRole: (role: string) => {
    const roleMap: Record<string, string> = {
      'admin': 'ADMINISTRATOR',
      'prosecutor': 'PROSECUTOR',
      'detective': 'DETECTIVE',
      'analyst': 'ANALYST',
      'user': 'USER'
    };
    return roleMap[role?.toLowerCase()] || role?.toUpperCase() || 'USER';
  }
};

// Import for default export
import { 
  tableManager as importedTableManager, 
  createTableStats as importedCreateTableStats 
} from '$lib/stores/tables';

// Export all components and utilities
export default {
  YoRHaTable: () => import('./YoRHaTable.svelte'),
  YoRHaDataGrid: () => import('./YoRHaDataGrid.svelte'),
  tableManager: importedTableManager,
  createTableStats: importedCreateTableStats,
  yorhaTablePresets,
  yorhaGridPresets,
  statusFormatters
};