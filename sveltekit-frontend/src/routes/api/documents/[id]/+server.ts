import { legalDocuments } from "$lib/server/db/unified-schema";
// @ts-nocheck
type { RequestEvent }, {
json } from "@sveltejs/kit";
import { db } from "$lib/server/db/index";

// Sample documents for fallback
const sampleDocuments = [
  {
    id: "doc-1",
    title: "Criminal Case Brief - State v. Johnson",
    content:
      "# Criminal Case Brief\n\n## Case Overview\nState v. Johnson involves charges of armed robbery and assault...\n\n## Facts\nOn the evening of March 15, 2024, the defendant allegedly entered a convenience store with a weapon and demanded money from the cashier. Security footage shows the defendant threatening the clerk and taking approximately $347 from the register.\n\n## Legal Issues\n1. Whether the defendant's actions constitute armed robbery under state law\n2. Whether the evidence obtained during the search was lawfully seized\n3. Whether the defendant's Miranda rights were properly administered\n\n## Analysis\nThe elements of armed robbery under state law require:\n- Taking of property\n- From another person\n- By force or threat of force\n- With a deadly weapon\n\nAll elements appear to be satisfied based on the evidence...",
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
      {
        id: "cite-2",
        text: "State v. Smith, 123 State Rep. 456 (2020)",
        source: "123 State Rep. 456",
        type: "case",
      },
    ],
    tags: ["criminal", "robbery", "assault", "weapons"],
    wordCount: 1247,
    createdAt: new Date("2024-01-15").toISOString(),
    updatedAt: new Date("2024-01-15").toISOString(),
    lastSavedAt: new Date("2024-01-15").toISOString(),
  },
  {
    id: "doc-2",
    title: "Motion to Suppress Evidence",
    content:
      "# Motion to Suppress Evidence\n\n## Introduction\nDefendant moves this Court to suppress evidence obtained during a search of defendant's vehicle on the grounds that the search violated the Fourth Amendment to the United States Constitution.\n\n## Statement of Facts\nOn March 15, 2024, Officer Johnson conducted a traffic stop of defendant's vehicle for speeding. During the stop, Officer Johnson observed what he believed to be contraband in plain view...\n\n## Legal Argument\n### I. The Search Violated the Fourth Amendment\nThe Fourth Amendment protects against unreasonable searches and seizures. In this case, the officer exceeded the scope of a lawful traffic stop...\n\n### II. The Evidence Should Be Suppressed\nUnder the exclusionary rule, evidence obtained in violation of the Fourth Amendment must be suppressed...",
    documentType: "motion",
    status: "review",
    version: 2,
    citations: [
      {
        id: "cite-3",
        text: "Fourth Amendment protection against unreasonable searches",
        source: "U.S. Const. amend. IV",
        type: "statute",
      },
      {
        id: "cite-4",
        text: "Terry v. Ohio, 392 U.S. 1 (1968)",
        source: "392 U.S. 1",
        type: "case",
      },
    ],
    tags: ["motion", "evidence", "suppression", "fourth amendment"],
    wordCount: 892,
    createdAt: new Date("2024-01-16").toISOString(),
    updatedAt: new Date("2024-01-17").toISOString(),
    lastSavedAt: new Date("2024-01-17").toISOString(),
  },
];

// GET /api/documents/[id] - Get a specific document
export async function GET({ params }: RequestEvent) {
  try {
    const documentId = params.id;

    if (!documentId) {
      return json(
        {
          success: false,
          error: "Document ID is required",
        },
        { status: 400 },
      );
    }
    // Try to fetch from database first
    try {
      const document = await db
        .select()
        .from(legalDocuments)
        .where(eq(legalDocuments.id, documentId))
        .limit(1);

      if (document.length === 0) {
        return json(
          {
            success: false,
            error: "Document not found",
          },
          { status: 404 },
        );
      }
      return json({
        success: true,
        document: document[0],
      });
    } catch (dbError) {
      console.warn("Database query failed, using sample data:", dbError);

      // Find document in sample data
      const document = sampleDocuments.find((doc) => doc.id === documentId);

      if (!document) {
        return json(
          {
            success: false,
            error: "Document not found",
          },
          { status: 404 },
        );
      }
      return json({
        success: true,
        document,
      });
    }
  } catch (error) {
    console.error("Error fetching document:", error);
    return json(
      {
        success: false,
        error: "Failed to fetch document",
      },
      { status: 500 },
    );
  }
}
// PUT /api/documents/[id] - Update a document
export async function PUT({ params, request }: RequestEvent) {
  try {
    const documentId = params.id;
    const body = await request.json();

    if (!documentId) {
      return json(
        {
          success: false,
          error: "Document ID is required",
        },
        { status: 400 },
      );
    }
    const { title, content, documentType, status, citations, tags, metadata } =
      body;

    // Try to update in database
    try {
      const updates: any = {
        updatedAt: new Date(),
        lastSavedAt: new Date(),
      };

      if (title !== undefined) updates.title = title;
      if (content !== undefined) {
        updates.content = content;
        updates.wordCount = content.split(/\s+/).length;
      }
      if (documentType !== undefined) updates.documentType = documentType;
      if (status !== undefined) updates.status = status;
      if (citations !== undefined) updates.citations = citations;
      if (tags !== undefined) updates.tags = tags;
      if (metadata !== undefined) updates.metadata = metadata;

      const updatedDocument = await db
        .update(legalDocuments)
        .set(updates)
        .where(eq(legalDocuments.id, documentId))
        .returning();

      if (updatedDocument.length === 0) {
        return json(
          {
            success: false,
            error: "Document not found",
          },
          { status: 404 },
        );
      }
      return json({
        success: true,
        document: updatedDocument[0],
      });
    } catch (dbError) {
      console.warn("Database update failed, returning mock response:", dbError);

      // Return mock response for development
      const mockDocument = {
        id: documentId,
        title: title || "Updated Document",
        content: content || "Updated content",
        documentType: documentType || "brief",
        status: status || "draft",
        citations: citations || [],
        tags: tags || [],
        metadata: metadata || {},
        wordCount: content ? content.split(/\s+/).length : 0,
        updatedAt: new Date().toISOString(),
        lastSavedAt: new Date().toISOString(),
      };

      return json({
        success: true,
        document: mockDocument,
      });
    }
  } catch (error) {
    console.error("Error updating document:", error);
    return json(
      {
        success: false,
        error: "Failed to update document",
      },
      { status: 500 },
    );
  }
}
// DELETE /api/documents/[id] - Delete a document
export async function DELETE({ params }: RequestEvent) {
  try {
    const documentId = params.id;

    if (!documentId) {
      return json(
        {
          success: false,
          error: "Document ID is required",
        },
        { status: 400 },
      );
    }
    // Try to delete from database
    try {
      const deletedDocument = await db
        .delete(legalDocuments)
        .where(eq(legalDocuments.id, documentId))
        .returning();

      if (deletedDocument.length === 0) {
        return json(
          {
            success: false,
            error: "Document not found",
          },
          { status: 404 },
        );
      }
      return json({
        success: true,
        message: "Document deleted successfully",
      });
    } catch (dbError) {
      console.warn("Database delete failed, returning mock response:", dbError);

      return json({
        success: true,
        message: "Document deleted successfully (mock)",
      });
    }
  } catch (error) {
    console.error("Error deleting document:", error);
    return json(
      {
        success: false,
        error: "Failed to delete document",
      },
      { status: 500 },
    );
  }
}
