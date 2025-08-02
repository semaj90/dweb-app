import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Mock data for development - replace with real database in production
const mockCases = [
  {
    id: 'CASE-2024-001',
    title: 'Contract Dispute - Smith vs. Tech Corp',
    description: 'Breach of contract claim regarding software licensing agreement',
    status: 'active',
    priority: 'high',
    createdAt: '2024-01-15T10:00:00Z',
    updatedAt: '2024-01-20T14:30:00Z',
    evidenceCount: 12,
    documentsCount: 8
  },
  {
    id: 'CASE-2024-002', 
    title: 'Employment Dispute - Johnson vs. Manufacturing Inc',
    description: 'Wrongful termination and workplace harassment claims',
    status: 'pending',
    priority: 'medium',
    createdAt: '2024-01-10T09:15:00Z',
    updatedAt: '2024-01-18T16:45:00Z',
    evidenceCount: 6,
    documentsCount: 15
  },
  {
    id: 'CASE-2024-003',
    title: 'Intellectual Property - Patent Infringement',
    description: 'Technology patent dispute in mobile app development',
    status: 'closed',
    priority: 'high',
    createdAt: '2024-01-05T11:30:00Z',
    updatedAt: '2024-01-25T10:00:00Z',
    evidenceCount: 20,
    documentsCount: 25
  }
];

export const GET: RequestHandler = async ({ url }) => {
  try {
    const search = url.searchParams.get('search');
    const status = url.searchParams.get('status');
    const priority = url.searchParams.get('priority');
    
    let filteredCases = [...mockCases];
    
    // Apply filters
    if (search) {
      const searchLower = search.toLowerCase();
      filteredCases = filteredCases.filter(c => 
        c.title.toLowerCase().includes(searchLower) ||
        c.description.toLowerCase().includes(searchLower)
      );
    }
    
    if (status) {
      filteredCases = filteredCases.filter(c => c.status === status);
    }
    
    if (priority) {
      filteredCases = filteredCases.filter(c => c.priority === priority);
    }

    return json({
      cases: filteredCases,
      total: filteredCases.length,
      filters: { search, status, priority }
    });
  } catch (error) {
    return json(
      { error: 'Failed to fetch cases' },
      { status: 500 }
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { title, description, priority = 'medium' } = await request.json();
    
    if (!title || !description) {
      return json(
        { error: 'Title and description are required' },
        { status: 400 }
      );
    }

    const newCase = {
      id: `CASE-2024-${String(mockCases.length + 1).padStart(3, '0')}`,
      title,
      description,
      status: 'active',
      priority,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      evidenceCount: 0,
      documentsCount: 0
    };

    mockCases.push(newCase);
    
    return json(newCase, { status: 201 });
  } catch (error) {
    return json(
      { error: 'Failed to create case' },
      { status: 500 }
    );
  }
};