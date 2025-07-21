import { legalDocuments } from "$lib/server/db/unified-schema";
import type { RequestEvent } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { and, desc, eq, like, or } from "drizzle-orm";
import { db } from "$lib/server/db/index";

// Sample documents for when database is not available
const sampleDocuments = [
  {
    id: "doc-1",
    title: "Criminal Case Brief - State v. Johnson",
    content:
      "# Criminal Case Brief\n\n## Case Overview\nState v. Johnson involves charges of armed robbery and assault...",
    documentType: "brief",
    status: "draft",
    version: 1,
    citations: [
      {
        id: "cite-1",
        text: "Miranda v. Arizona, 384 U.S. 436 (1966)",
        source: "384 U.S. 436",
        type: "case",
      },
    ],
    tags: ["criminal", "robbery", "assault"],
    wordCount: 1247,
    createdAt: new Date("2024-01-15").toISOString(),
    updatedAt: new Date("2024-01-15").toISOString(),
  },
  {
    id: "doc-2",
    title: "Motion to Suppress Evidence",
    content:
      "# Motion to Suppress Evidence\n\n## Introduction\nDefendant moves this Court to suppress evidence obtained...",
    documentType: "motion",
    status: "review",
    version: 2,
    citations: [
      {
        id: "cite-2",
        text: "Fourth Amendment protection against unreasonable searches",
        source: "U.S. Const. amend. IV",
        type: "statute",
      },
    ],
    tags: ["motion", "evidence", "suppression"],
    wordCount: 892,
    createdAt: new Date("2024-01-16").toISOString(),
    updatedAt: new Date("2024-01-17").toISOString(),
  },
  {
    id: "doc-3",
    title: "Evidence Analysis Report",
    content:
      "# Evidence Analysis Report\n\n## Executive Summary\nThis report analyzes the physical evidence collected...",
    documentType: "evidence",
    status: "approved",
    version: 1,
    citations: [],
    tags: ["evidence", "analysis", "forensics"],
    wordCount: 2134,
    createdAt: new Date("2024-01-18").toISOString(),
    updatedAt: new Date("2024-01-18").toISOString(),
  },
];

// GET /api/documents - List documents with filtering and pagination
export async function GET({ url }: RequestEvent) {
  try {
    const caseId = url.searchParams.get("caseId");
    const documentType = url.searchParams.get("type");
    const status = url.searchParams.get("status");
    const search = url.searchParams.get("search");
    const limit = parseInt(url.searchParams.get("limit") || "10");
    const offset = parseInt(url.searchParams.get("offset") || "0");

    // Try to fetch from database first
    try {
      const conditions = [];

      if (caseId) {
        conditions.push(eq(legalDocuments.caseId, caseId));
      }
      if (documentType) {
        conditions.push(eq(legalDocuments.documentType, documentType));
      }
      if (status) {
        conditions.push(eq(legalDocuments.status, status));
      }
      if (search) {
        conditions.push(
          or(
            like(legalDocuments.title, `%${search}%`),
            like(legalDocuments.content, `%${search}%`),
          ),
        );
      }
      const documents = await db
        .select()
        .from(legalDocuments)
        .where(conditions.length > 0 ? and(...conditions) : undefined)
        .orderBy(desc(legalDocuments.updatedAt))
        .limit(limit)
        .offset(offset);

      return json({
        success: true,
        documents,
        pagination: {
          limit,
          offset,
          total: documents.length,
        },
      });
    } catch (dbError) {
      console.warn("Database query failed, using sample data:", dbError);

      // Filter sample documents based on query parameters
      let filteredDocuments = sampleDocuments;

      if (caseId) {
        filteredDocuments = filteredDocuments.filter((doc) =>
          doc.id.includes(caseId),
        );
      }
      if (documentType) {
        filteredDocuments = filteredDocuments.filter(
          (doc) => doc.documentType === documentType,
        );
      }
      if (status) {
        filteredDocuments = filteredDocuments.filter(
          (doc) => doc.status === status,
        );
      }
      if (search) {
        filteredDocuments = filteredDocuments.filter(
          (doc) =>
            doc.title.toLowerCase().includes(search.toLowerCase()) ||
            doc.content.toLowerCase().includes(search.toLowerCase()),
        );
      }
      // Apply pagination
      const paginatedDocuments = filteredDocuments.slice(
        offset,
        offset + limit,
      );

      return json({
        success: true,
        documents: paginatedDocuments,
        pagination: {
          limit,
          offset,
          total: filteredDocuments.length,
        },
      });
    }
  } catch (error) {
    console.error("Error fetching documents:", error);
    return json(
      {
        success: false,
        error: "Failed to fetch documents",
        documents: [],
      },
      { status: 500 },
    );
  }
}
// POST /api/documents - Create a new document
export async function POST({ request }: RequestEvent) {
  try {
    const body = await request.json();
    const {
      title,
      content,
      documentType,
      caseId,
      userId,
      citations = [],
      tags = [],
      metadata = {},
    } = body;

    // Validate required fields
    if (!title || !content || !documentType || !userId) {
      return json(
        {
          success: false,
          error:
            "Missing required fields: title, content, documentType, userId",
        },
        { status: 400 },
      );
    }
    // Try to insert into database
    try {
      const newDocument = await db
        .insert(legalDocuments)
        .values({
          title,
          content,
          documentType,
          caseId: caseId || null,
          userId,
          citations: JSON.stringify(citations),
          tags: JSON.stringify(tags),
          metadata,
          wordCount: content.split(/\s+/).length,
          status: "draft",
          version: 1,
        })
        .returning();

      return json({
        success: true,
        document: newDocument[0],
      });
    } catch (dbError) {
      console.warn("Database insert failed, returning mock response:", dbError);

      // Return mock response for development
      const mockDocument = {
        id: `doc-${Date.now()}`,
        title,
        content,
        documentType,
        caseId,
        userId,
        citations,
        tags,
        metadata,
        status: "draft",
        version: 1,
        wordCount: content.split(/\s+/).length,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };

      return json({
        success: true,
        document: mockDocument,
      });
    }
  } catch (error) {
    console.error("Error creating document:", error);
    return json(
      {
        success: false,
        error: "Failed to create document",
      },
      { status: 500 },
    );
  }
}
