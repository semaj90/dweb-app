import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Mock evidence data
const mockEvidence = [
  {
    id: 'EVD-001',
    caseId: 'CASE-2024-001',
    title: 'Software License Agreement',
    type: 'document',
    content: 'Software licensing agreement between parties...',
    uploadedAt: '2024-01-15T10:30:00Z',
    fileSize: '2.4 MB',
    tags: ['contract', 'software', 'licensing']
  },
  {
    id: 'EVD-002',
    caseId: 'CASE-2024-001',
    title: 'Email Correspondence',
    type: 'communication',
    content: 'Email chain regarding contract terms...',
    uploadedAt: '2024-01-16T14:15:00Z',
    fileSize: '156 KB',
    tags: ['email', 'communication', 'negotiation']
  },
  {
    id: 'EVD-003',
    caseId: 'CASE-2024-002',
    title: 'Employment Contract',
    type: 'document',
    content: 'Original employment agreement...',
    uploadedAt: '2024-01-10T09:45:00Z',
    fileSize: '1.8 MB',
    tags: ['employment', 'contract', 'terms']
  }
];

export const GET: RequestHandler = async ({ url }) => {
  try {
    const caseId = url.searchParams.get('caseId');
    const type = url.searchParams.get('type');
    const search = url.searchParams.get('search');
    
    let filteredEvidence = [...mockEvidence];
    
    if (caseId) {
      filteredEvidence = filteredEvidence.filter(e => e.caseId === caseId);
    }
    
    if (type) {
      filteredEvidence = filteredEvidence.filter(e => e.type === type);
    }
    
    if (search) {
      const searchLower = search.toLowerCase();
      filteredEvidence = filteredEvidence.filter(e => 
        e.title.toLowerCase().includes(searchLower) ||
        e.content.toLowerCase().includes(searchLower) ||
        e.tags.some(tag => tag.toLowerCase().includes(searchLower))
      );
    }

    return json({
      evidence: filteredEvidence,
      total: filteredEvidence.length,
      filters: { caseId, type, search }
    });
  } catch (error) {
    return json(
      { error: 'Failed to fetch evidence' },
      { status: 500 }
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { caseId, title, type, content, tags = [] } = await request.json();
    
    if (!caseId || !title || !type) {
      return json(
        { error: 'Case ID, title, and type are required' },
        { status: 400 }
      );
    }

    const newEvidence = {
      id: `EVD-${String(mockEvidence.length + 1).padStart(3, '0')}`,
      caseId,
      title,
      type,
      content: content || '',
      uploadedAt: new Date().toISOString(),
      fileSize: '0 KB',
      tags
    };

    mockEvidence.push(newEvidence);
    
    return json(newEvidence, { status: 201 });
  } catch (error) {
    return json(
      { error: 'Failed to create evidence' },
      { status: 500 }
    );
  }
};