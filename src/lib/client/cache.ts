import loki from 'lokijs';
import Fuse from 'fuse.js';

// Initialize LokiJS database
const db = new loki('legal_ai_cache.db');

// Collections
const evidenceCollection = db.addCollection('evidence', { 
  indices: ['id', 'filename', 'caseId'] 
});

const casesCollection = db.addCollection('cases', { 
  indices: ['id', 'caseNumber', 'userId'] 
});

const reportsCollection = db.addCollection('reports', { 
  indices: ['id', 'caseId'] 
});

const searchHistoryCollection = db.addCollection('searchHistory', { 
  indices: ['timestamp', 'userId'] 
});

// Cache management
export const cache = {
  // Evidence operations
  evidence: {
    set: (list: any[]) => {
      evidenceCollection.clear();
      list.forEach(item => evidenceCollection.insert(item));
    },

    get: (caseId?: number) => {
      if (caseId) {
        return evidenceCollection.find({ caseId });
      }
      return evidenceCollection.find();
    },

    add: (item: any) => {
      evidenceCollection.insert(item);
    },

    update: (id: number, updates: any) => {
      const item = evidenceCollection.findOne({ id });
      if (item) {
        Object.assign(item, updates);
        evidenceCollection.update(item);
      }
    },

    remove: (id: number) => {
      evidenceCollection.removeWhere({ id });
    },

    search: (query: string, options?: any) => {
      const items = evidenceCollection.find();
      const fuse = new Fuse(items, {
        keys: ['filename', 'tags', 'fileType'],
        threshold: 0.3,
        includeScore: true,
        ...options
      });
      return fuse.search(query).map(result => ({
        ...(result.item as Record<string, any>),
        _score: result.score
      }));
    }
  },

  // Cases operations
  cases: {
    set: (list: any[]) => {
      casesCollection.clear();
      list.forEach(item => casesCollection.insert(item));
    },

    get: (userId?: number) => {
      if (userId) {
        return casesCollection.find({ userId });
      }
      return casesCollection.find();
    },

    add: (item: any) => {
      casesCollection.insert(item);
    },

    update: (id: number, updates: any) => {
      const item = casesCollection.findOne({ id });
      if (item) {
        Object.assign(item, updates);
        casesCollection.update(item);
      }
    },

    remove: (id: number) => {
      casesCollection.removeWhere({ id });
    },

    search: (query: string, options?: any) => {
      const items = casesCollection.find();
      const fuse = new Fuse(items, {
        keys: ['title', 'description', 'caseNumber'],
        threshold: 0.3,
        includeScore: true,
        ...options
      });
      return fuse.search(query).map(result => ({
        ...(result.item as Record<string, any>),
        _score: result.score
      }));
    }
  },

  // Reports operations
  reports: {
    set: (list: any[]) => {
      reportsCollection.clear();
      list.forEach(item => reportsCollection.insert(item));
    },

    get: (caseId?: number) => {
      if (caseId) {
        return reportsCollection.findOne({ caseId });
      }
      return reportsCollection.find();
    },

    add: (item: any) => {
      reportsCollection.insert(item);
    },

    update: (id: number, updates: any) => {
      const item = reportsCollection.findOne({ id });
      if (item) {
        Object.assign(item, updates);
        reportsCollection.update(item);
      }
    },

    remove: (id: number) => {
      reportsCollection.removeWhere({ id });
    },

    search: (query: string, options?: any) => {
      const items = reportsCollection.find();
      const fuse = new Fuse(items, {
        keys: ['title', 'summary'],
        threshold: 0.3,
        includeScore: true,
        ...options
      });
      return fuse.search(query).map(result => ({
        ...(result.item as Record<string, any>),
        _score: result.score
      }));
    }
  },

  // Search history
  searchHistory: {
    add: (query: string, results: number, userId: number) => {
      searchHistoryCollection.insert({
        query,
        results,
        userId,
        timestamp: new Date()
      });

      // Keep only last 100 searches
      const all = searchHistoryCollection.find();
      if (all.length > 100) {
        const oldest = all.sort((a, b) => a.timestamp - b.timestamp).slice(0, all.length - 100);
        oldest.forEach(item => searchHistoryCollection.remove(item));
      }
    },

    get: (userId: number, limit = 10) => {
      return searchHistoryCollection
        .find({ userId })
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, limit);
    },

    getPopular: (userId: number, limit = 5) => {
      const history = searchHistoryCollection.find({ userId });
      const queryCount = {};
      
      history.forEach(item => {
        queryCount[item.query] = (queryCount[item.query] || 0) + 1;
      });

      return Object.entries(queryCount)
        .sort((a, b) => (b[1] as number) - (a[1] as number))
        .slice(0, limit)
        .map(([query, count]) => ({ query, count }));
    }
  },

  // Global operations
  clear: () => {
    evidenceCollection.clear();
    casesCollection.clear();
    reportsCollection.clear();
    searchHistoryCollection.clear();
  },

  stats: () => ({
    evidence: evidenceCollection.count(),
    cases: casesCollection.count(),
    reports: reportsCollection.count(),
    searches: searchHistoryCollection.count()
  }),

  // Advanced search across all collections
  globalSearch: (query: string, options?: any) => {
    const evidenceResults = cache.evidence.search(query, options);
    const caseResults = cache.cases.search(query, options);
    const reportResults = cache.reports.search(query, options);

    return {
      evidence: evidenceResults,
      cases: caseResults,
      reports: reportResults,
      total: evidenceResults.length + caseResults.length + reportResults.length
    };
  }
};

// Auto-persist to localStorage (optional)
export const persistCache = {
  save: () => {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('legal_ai_cache', JSON.stringify(db.serialize()));
    }
  },

  load: () => {
    if (typeof localStorage !== 'undefined') {
      const saved = localStorage.getItem('legal_ai_cache');
      if (saved) {
        try {
          db.loadJSON(saved);
        } catch (error) {
          console.warn('Failed to load cache from localStorage:', error);
        }
      }
    }
  },

  clear: () => {
    if (typeof localStorage !== 'undefined') {
      localStorage.removeItem('legal_ai_cache');
    }
    cache.clear();
  }
};

// Initialize cache on import
if (typeof window !== 'undefined') {
  persistCache.load();
  
  // Auto-save every 30 seconds
  setInterval(persistCache.save, 30000);
  
  // Save on page unload
  window.addEventListener('beforeunload', persistCache.save);
}