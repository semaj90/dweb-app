// Re-export Document API service from canonical location
// Added to satisfy imports expecting '$lib/api/documentApi'
export * from '../services/documentApi';
import { DocumentApiService } from '../services/documentApi';
export default new DocumentApiService();
