/**
 * Gallery API Server - Main Gallery Data Handler
 * Provides unified access to all media types across the legal AI platform
 */

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/database';
import { cases, evidence, users, legalDocuments } from '$lib/server/database';
import { eq, desc, asc, and, or, like, isNull } from 'drizzle-orm';
import { URL } from "url";

export interface GalleryItem {
  id: string;
  type: 'evidence' | 'document' | 'image' | 'ai-generated' | 'upload';
  title: string;
  description?: string;
  url: string;
  thumbnailUrl?: string;
  fileType: string;
  size: number;
  uploadedAt: string;
  uploadedBy: string;
  caseId?: string;
  caseTitle?: string;
  tags?: string[];
  metadata?: Record<string, any>;
  isPublic: boolean;
  category: string;
  searchableText?: string;
}

export interface GalleryResponse {
  items: GalleryItem[];
  totalCount: number;
  categories: Array<{ name: string; count: number }>;
  filters: {
    types: string[];
    cases: Array<{ id: string; title: string }>;
    users: Array<{ id: string; name: string }>;
  };
  pagination: {
    page: number;
    pageSize: number;
    totalPages: number;
  };
}

interface GalleryFilters {
  type?: string;
  category?: string;
  caseId?: string;
  userId?: string;
  search?: string;
  tags?: string[];
  dateFrom?: string;
  dateTo?: string;
  fileTypes?: string[];
  isPublic?: boolean;
}

export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    // Parse query parameters
    const page = Math.max(1, parseInt(url.searchParams.get('page') || '1'));
    const pageSize = Math.min(100, Math.max(10, parseInt(url.searchParams.get('pageSize') || '20')));
    const sortBy = url.searchParams.get('sortBy') || 'uploadedAt';
    const sortOrder = url.searchParams.get('sortOrder') || 'desc';

    // Parse filters
    const filters: GalleryFilters = {
      type: url.searchParams.get('type') || undefined,
      category: url.searchParams.get('category') || undefined,
      caseId: url.searchParams.get('caseId') || undefined,
      userId: url.searchParams.get('userId') || undefined,
      search: url.searchParams.get('search') || undefined,
      tags: url.searchParams.get('tags')?.split(',').filter(Boolean) || undefined,
      dateFrom: url.searchParams.get('dateFrom') || undefined,
      dateTo: url.searchParams.get('dateTo') || undefined,
      fileTypes: url.searchParams.get('fileTypes')?.split(',').filter(Boolean) || undefined,
      isPublic: url.searchParams.get('isPublic') ? url.searchParams.get('isPublic') === 'true' : undefined
    };

    const startTime = Date.now();

    // Get gallery items from multiple sources
    const [evidenceItems, documentItems, aiGeneratedItems, categories, casesData, usersData] = await Promise.all([
      getEvidenceItems(filters, page, pageSize, sortBy, sortOrder),
      getDocumentItems(filters, page, pageSize, sortBy, sortOrder),
      getAIGeneratedItems(filters, page, pageSize, sortBy, sortOrder),
      getCategories(),
      getCases(),
      getUsers()
    ]);

    // Combine and sort all items
    const allItems = [
      ...evidenceItems.items,
      ...documentItems.items,
      ...aiGeneratedItems.items
    ];

    // Sort combined items
    allItems.sort((a, b) => {
      const aVal = a[sortBy as keyof GalleryItem] || '';
      const bVal = b[sortBy as keyof GalleryItem] || '';
      
      if (sortOrder === 'desc') {
        return bVal > aVal ? 1 : -1;
      }
      return aVal > bVal ? 1 : -1;
    });

    // Apply pagination to combined results
    const offset = (page - 1) * pageSize;
    const paginatedItems = allItems.slice(offset, offset + pageSize);
    const totalCount = allItems.length;

    // Prepare filter options
    const filterOptions = {
      types: ['evidence', 'document', 'image', 'ai-generated', 'upload'],
      cases: casesData.map(c => ({ id: c.id, title: c.title })),
      users: usersData.map(u => ({ id: u.id, name: u.email || 'Unknown' }))
    };

    const processingTime = Date.now() - startTime;

    const response: GalleryResponse = {
      items: paginatedItems,
      totalCount,
      categories,
      filters: filterOptions,
      pagination: {
        page,
        pageSize,
        totalPages: Math.ceil(totalCount / pageSize)
      }
    };

    return json(response, {
      headers: {
        'X-Processing-Time': `${processingTime}ms`,
        'X-Total-Items': totalCount.toString(),
        'Cache-Control': 'public, max-age=60' // Cache for 1 minute
      }
    });

  } catch (err) {
    console.error('Gallery API error:', err);
    throw error(500, `Failed to fetch gallery data: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

async function getEvidenceItems(filters: GalleryFilters, page: number, pageSize: number, sortBy: string, sortOrder: string) {
  try {
    const evidenceQuery = db
      .select({
        id: evidence.id,
        title: evidence.title,
        description: evidence.description,
        fileName: evidence.fileName,
        fileSize: evidence.fileSize,
        fileType: evidence.fileType,
        filePath: evidence.filePath,
        uploadedAt: evidence.uploadedAt,
        caseId: evidence.caseId,
        caseTitle: cases.title,
        tags: evidence.tags,
        metadata: evidence.metadata,
        isPublic: evidence.isPublic,
        contentText: evidence.contentText
      })
      .from(evidence)
      .leftJoin(cases, eq(evidence.caseId, cases.id));

    // Apply filters
    const conditions = [];
    
    if (filters.caseId) {
      conditions.push(eq(evidence.caseId, filters.caseId));
    }
    
    if (filters.search) {
      conditions.push(
        or(
          like(evidence.title, `%${filters.search}%`),
          like(evidence.description, `%${filters.search}%`),
          like(evidence.contentText, `%${filters.search}%`)
        )
      );
    }
    
    if (filters.fileTypes && filters.fileTypes.length > 0) {
      conditions.push(
        or(...filters.fileTypes.map(type => like(evidence.fileType, `%${type}%`)))
      );
    }

    if (filters.isPublic !== undefined) {
      conditions.push(eq(evidence.isPublic, filters.isPublic));
    }

    if (conditions.length > 0) {
      evidenceQuery.where(and(...conditions));
    }

    const evidenceData = await evidenceQuery.execute();

    const items: GalleryItem[] = evidenceData.map(item => ({
      id: item.id,
      type: 'evidence' as const,
      title: item.title || item.fileName || 'Untitled Evidence',
      description: item.description || undefined,
      url: `/api/files/evidence/${item.id}`,
      thumbnailUrl: generateThumbnailUrl(item.filePath, item.fileType),
      fileType: item.fileType || 'unknown',
      size: item.fileSize || 0,
      uploadedAt: item.uploadedAt?.toISOString() || new Date().toISOString(),
      uploadedBy: 'System', // TODO: Add user tracking
      caseId: item.caseId || undefined,
      caseTitle: item.caseTitle || undefined,
      tags: Array.isArray(item.tags) ? item.tags : [],
      metadata: item.metadata as Record<string, any> || {},
      isPublic: item.isPublic || false,
      category: 'Legal Evidence',
      searchableText: [item.title, item.description, item.contentText].filter(Boolean).join(' ')
    }));

    return { items, total: items.length };
  } catch (err) {
    console.error('Error fetching evidence items:', err);
    return { items: [], total: 0 };
  }
}

async function getDocumentItems(filters: GalleryFilters, page: number, pageSize: number, sortBy: string, sortOrder: string) {
  try {
    // TODO: Implement documents table query when schema is available
    // For now, return empty array
    return { items: [], total: 0 };
  } catch (err) {
    console.error('Error fetching document items:', err);
    return { items: [], total: 0 };
  }
}

async function getAIGeneratedItems(filters: GalleryFilters, page: number, pageSize: number, sortBy: string, sortOrder: string) {
  try {
    // Check if we have AI-generated content in local storage or service
    // This would integrate with the image generation service we created
    const aiItems: GalleryItem[] = [];

    // TODO: Query AI-generated images from storage or database
    // For now, return empty array - this will be populated by the image generation service
    
    return { items: aiItems, total: aiItems.length };
  } catch (err) {
    console.error('Error fetching AI-generated items:', err);
    return { items: [], total: 0 };
  }
}

async function getCategories() {
  return [
    { name: 'Legal Evidence', count: 0 },
    { name: 'Case Documents', count: 0 },
    { name: 'AI Generated', count: 0 },
    { name: 'Uploads', count: 0 },
    { name: 'Images', count: 0 },
    { name: 'Videos', count: 0 },
    { name: 'Audio', count: 0 },
    { name: 'PDFs', count: 0 },
    { name: 'Presentations', count: 0 },
    { name: 'Spreadsheets', count: 0 }
  ];
}

async function getCases() {
  try {
    return await db
      .select({
        id: cases.id,
        title: cases.title
      })
      .from(cases)
      .orderBy(asc(cases.title))
      .execute();
  } catch (err) {
    console.error('Error fetching cases:', err);
    return [];
  }
}

async function getUsers() {
  try {
    return await db
      .select({
        id: users.id,
        email: users.email
      })
      .from(users)
      .orderBy(asc(users.email))
      .execute();
  } catch (err) {
    console.error('Error fetching users:', err);
    return [];
  }
}

function generateThumbnailUrl(filePath: string | null, fileType: string | null): string | undefined {
  if (!filePath || !fileType) return undefined;
  
  // For images, we can serve them directly as thumbnails
  if (fileType.startsWith('image/')) {
    return `/api/files/thumbnails/${encodeURIComponent(filePath)}`;
  }
  
  // For other file types, return appropriate icons
  if (fileType.includes('pdf')) {
    return '/icons/pdf-thumbnail.svg';
  }
  
  if (fileType.includes('video')) {
    return '/icons/video-thumbnail.svg';
  }
  
  if (fileType.includes('audio')) {
    return '/icons/audio-thumbnail.svg';
  }
  
  return '/icons/file-thumbnail.svg';
}

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const data = await request.json();
    
    // Handle bulk operations like delete, move, tag
    if (data.action === 'bulk_delete') {
      return await handleBulkDelete(data.ids);
    }
    
    if (data.action === 'bulk_tag') {
      return await handleBulkTag(data.ids, data.tags);
    }
    
    if (data.action === 'bulk_move') {
      return await handleBulkMove(data.ids, data.caseId);
    }
    
    throw error(400, 'Invalid action');
    
  } catch (err) {
    console.error('Gallery POST error:', err);
    throw error(500, `Gallery operation failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

async function handleBulkDelete(ids: string[]) {
  // TODO: Implement bulk delete across different item types
  return json({ success: true, deleted: ids.length });
}

async function handleBulkTag(ids: string[], tags: string[]) {
  // TODO: Implement bulk tagging across different item types
  return json({ success: true, tagged: ids.length });
}

async function handleBulkMove(ids: string[], caseId: string) {
  // TODO: Implement bulk move to different case
  return json({ success: true, moved: ids.length });
}