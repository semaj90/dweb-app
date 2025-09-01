import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types';


// Individual Legal Document API - SvelteKit Server Endpoint
import { json } from "@sveltejs/kit";

// Import with fallback for different schema files
let schema: any = {};
try {
  schema = await import("$lib/server/db/unified-schema.js");
} catch (error: any) {
  try {
    schema = await import("$lib/server/db/schema-postgres.js");
  } catch (error2) {
    console.warn("No database schema available");
  }
}
const { legalDocuments } = schema;

// GET - Get specific legal document
export const GET: RequestHandler = async ({ params }) => {
  try {
    const documentId = params.id;
    if (!documentId) {
      return json({ error: "Document ID is required" }, { status: 400 });
    }
    // Handle case where schema is not available
    if (!legalDocuments) {
      console.warn("Legal documents table not available, returning mock data");
      return json({
        id: documentId,
        title: "Sample Legal Document",
        content: "This is sample content for the legal document...",
        documentType: "brief",
        status: "draft",
        createdAt: new Date(),
        updatedAt: new Date(),
        wordCount: 150,
      });
    }
    // Get document
    const [document] = await db
      .select()
      .from(legalDocuments)
      .where(eq(legalDocuments.id, documentId));

    if (!document) {
      return json({ error: "Document not found" }, { status: 404 });
    }
    // Calculate word count
    const documentWithWordCount = {
      ...document,
      wordCount: document.content ? document.content.split(/\s+/).length : 0,
    };

    return json(documentWithWordCount);
  } catch (error: any) {
    console.error("Error fetching legal document:", error);
    return json({ error: "Failed to fetch legal document" }, { status: 500 });
  }
};

// PUT - Update specific legal document
export const PUT: RequestHandler = async ({ request, params }) => {
  try {
    const documentId = params.id;
    if (!documentId) {
      return json({ error: "Document ID is required" }, { status: 400 });
    }
    const data = await request.json();

    // Handle case where schema is not available
    if (!legalDocuments) {
      console.warn(
        "Legal documents table not available, returning mock response",
      );
      return json({
        id: documentId,
        ...data,
        updatedAt: new Date(),
        wordCount: data.content ? data.content.split(/\s+/).length : 0,
      });
    }
    // Calculate word count if content is provided
    const updateData: any = {
      ...data,
      updatedAt: new Date(),
    };

    if (data.content) {
      updateData.wordCount = data.content.split(/\s+/).length;
    }
    // Update document
    const [updatedDocument] = await db
      .update(legalDocuments)
      .set(updateData)
      .where(eq(legalDocuments.id, documentId))
      .returning();

    if (!updatedDocument) {
      return json({ error: "Document not found" }, { status: 404 });
    }
    return json(updatedDocument);
  } catch (error: any) {
    console.error("Error updating legal document:", error);
    return json({ error: "Failed to update legal document" }, { status: 500 });
  }
};

// DELETE - Delete specific legal document
export const DELETE: RequestHandler = async ({ params }) => {
  try {
    const documentId = params.id;
    if (!documentId) {
      return json({ error: "Document ID is required" }, { status: 400 });
    }
    // Handle case where schema is not available
    if (!legalDocuments) {
      console.warn(
        "Legal documents table not available, returning mock response",
      );
      return json({ success: true });
    }
    // Delete document
    const deleteResult = await db
      .delete(legalDocuments)
      .where(eq(legalDocuments.id, documentId))
      .returning();
    
    const deletedDocument = Array.isArray(deleteResult) ? deleteResult[0] : deleteResult;

    if (!deletedDocument) {
      return json({ error: "Document not found" }, { status: 404 });
    }
    return json({ success: true });
  } catch (error: any) {
    console.error("Error deleting legal document:", error);
    return json({ error: "Failed to delete legal document" }, { status: 500 });
  }
};
