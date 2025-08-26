import { writable } from 'svelte/store';
import type { Report } from '$lib/types';
import { cache } from '$lib/client/cache';

interface ReportStore {
  items: Report[];
  currentReport: Report | null;
  loading: boolean;
  error: string | null;
  saving: boolean;
  lastSaved: Date | null;
}

const initialState: ReportStore = {
  items: [],
  currentReport: null,
  loading: false,
  error: null,
  saving: false,
  lastSaved: null
};

function createReportStore() {
  const { subscribe, set, update } = writable<ReportStore>(initialState);

  return {
    subscribe,
    
    // Load report for a case
    load: async (caseId: number) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        // Try cache first
        const cached = cache.reports.get(caseId);
        if (cached) {
          update(state => ({
            ...state,
            currentReport: cached,
            loading: false
          }));
        }

        // Fetch from API
        const response = await fetch(`/api/cases/${caseId}/report`);
        if (!response.ok) throw new Error('Failed to load report');
        
        const data = await response.json();
        const report = data.data;

        // Update cache
        if (report) {
          cache.reports.add(report);
        }

        update(state => ({
          ...state,
          currentReport: report,
          loading: false
        }));

        return report;

      } catch (error) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message
        }));
      }
    },

    // Create or update report
    save: async (caseId: number, reportData: Partial<Report>) => {
      update(state => ({ ...state, saving: true, error: null }));

      try {
        const response = await fetch(`/api/cases/${caseId}/report`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(reportData)
        });

        if (!response.ok) throw new Error('Failed to save report');

        const data = await response.json();
        const savedReport = data.data;

        // Update cache
        cache.reports.update(savedReport.id, savedReport);

        update(state => ({
          ...state,
          currentReport: savedReport,
          saving: false,
          lastSaved: new Date()
        }));

        return savedReport;

      } catch (error) {
        update(state => ({
          ...state,
          saving: false,
          error: error.message
        }));
        throw error;
      }
    },

    // Auto-save with debouncing
    autoSave: (() => {
      let timeoutId: NodeJS.Timeout;
      
      return async (caseId: number, reportData: Partial<Report>, delay = 2000) => {
        clearTimeout(timeoutId);
        
        timeoutId = setTimeout(async () => {
          try {
            await this.save(caseId, reportData);
          } catch (error) {
            console.warn('Auto-save failed:', error);
          }
        }, delay);
      };
    })(),

    // Update report content
    updateContent: (updates: Partial<Report>) => {
      update(state => ({
        ...state,
        currentReport: state.currentReport 
          ? { ...state.currentReport, ...updates }
          : null
      }));
    },

    // Update summary
    updateSummary: (summary: string) => {
      update(state => ({
        ...state,
        currentReport: state.currentReport 
          ? { ...state.currentReport, summary }
          : null
      }));
    },

    // Update document content
    updateDoc: (doc: any) => {
      update(state => ({
        ...state,
        currentReport: state.currentReport 
          ? { ...state.currentReport, doc }
          : null
      }));
    },

    // Generate summary using AI
    generateSummary: async (caseId: number, evidenceText: string) => {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch('/api/llm/summarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: evidenceText,
            type: 'case'
          })
        });

        if (!response.ok) throw new Error('Failed to generate summary');

        const data = await response.json();
        const summary = data.data.summary;

        update(state => ({
          ...state,
          currentReport: state.currentReport 
            ? { ...state.currentReport, summary }
            : null,
          loading: false
        }));

        return summary;

      } catch (error) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message
        }));
        throw error;
      }
    },

    // Generate full report
    generateReport: async (caseId: number, template?: string) => {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch('/api/reports/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ caseId, template })
        });

        if (!response.ok) throw new Error('Failed to generate report');

        const data = await response.json();
        const report = data.data;

        // Update cache and store
        cache.reports.update(report.id, report);
        update(state => ({
          ...state,
          currentReport: report,
          loading: false,
          lastSaved: new Date()
        }));

        return report;

      } catch (error) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message
        }));
        throw error;
      }
    },

    // Export report
    export: async (format: 'pdf' | 'docx' | 'html' = 'pdf') => {
      const state = writable.get();
      if (!state.currentReport) {
        throw new Error('No report to export');
      }

      try {
        const response = await fetch(`/api/reports/${state.currentReport.id}/export`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ format })
        });

        if (!response.ok) throw new Error('Failed to export report');

        // Handle file download
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${state.currentReport.id}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

      } catch (error) {
        update(state => ({
          ...state,
          error: error.message
        }));
        throw error;
      }
    },

    // Set current report
    setCurrent: (report: Report | null) => {
      update(state => ({
        ...state,
        currentReport: report
      }));
    },

    // Clear error
    clearError: () => {
      update(state => ({
        ...state,
        error: null
      }));
    },

    // Reset store
    reset: () => {
      set(initialState);
    }
  };
}

export const reportStore = createReportStore();