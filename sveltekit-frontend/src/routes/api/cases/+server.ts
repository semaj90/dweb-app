import type { RequestHandler } from '@sveltejs/kit';
import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types.js";
import { z } from "zod";

// Enhanced case schema
const caseSchema = z.object({
  id: z.string().optional(),
  title: z.string().min(1, "Case title is required"),
  caseNumber: z.string().min(1, "Case number is required"),
  description: z.string().min(10, "Description must be at least 10 characters"),
  incidentDate: z.string().optional(),
  location: z.string().optional(),
  priority: z.enum(["low", "medium", "high", "urgent"]).default("medium"),
  status: z.enum(["open", "closed", "pending", "archived", "under_review"]).default("open"),
  category: z.string().min(1, "Category is required"),
  dangerScore: z.number().min(0).max(10).default(0),
  estimatedValue: z.number().optional(),
  jurisdiction: z.string().optional(),
  leadProsecutor: z.string().optional(),
  assignedTeam: z.array(z.string()).default([]),
  tags: z.array(z.string()).default([]),
  metadata: z.record(z.unknown()).default({})
});

// In-memory storage for demo (will be replaced with Drizzle)
let casesStore: any[] = [];

// Initialize with sample data
initializeSampleData();

// GET - List cases with advanced search and filtering
export const GET: RequestHandler = async ({ url }) => {
  const startTime = Date.now();
  
  try {
    // Extract query parameters
    const search = url.searchParams.get("search");
    const status = url.searchParams.get("status");
    const priority = url.searchParams.get("priority");
    const category = url.searchParams.get("category");
    const limit = parseInt(url.searchParams.get("limit") || "50");
    const offset = parseInt(url.searchParams.get("offset") || "0");
    const sortBy = url.searchParams.get("sortBy") || "createdAt";
    const sortOrder = url.searchParams.get("sortOrder") || "desc";

    console.log("🔍 Cases search request:", { search, status, priority, category, limit, offset });

    // Start with all cases
    let results = [...casesStore];

    // Apply filters
    if (status) {
      results = results.filter(c => c.status === status);
    }
    if (priority) {
      results = results.filter(c => c.priority === priority);
    }
    if (category) {
      results = results.filter(c => c.category === category);
    }

    // Apply search if provided
    if (search && search.trim()) {
      const searchTerm = search.toLowerCase();
      results = results.filter(c => 
        c.title.toLowerCase().includes(searchTerm) ||
        c.description.toLowerCase().includes(searchTerm) ||
        c.caseNumber.toLowerCase().includes(searchTerm) ||
        c.location?.toLowerCase().includes(searchTerm) ||
        c.tags.some((tag: string) => tag.toLowerCase().includes(searchTerm))
      );
      console.log(`🎯 Search "${search}" returned ${results.length} results`);
    }

    // Sort results
    results.sort((a, b) => {
      const aVal = a[sortBy] || "";
      const bVal = b[sortBy] || "";
      
      if (sortOrder === "desc") {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      } else {
        return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
      }
    });

    // Apply pagination
    const total = results.length;
    const paginatedResults = results.slice(offset, offset + limit);

    // Add analytics
    const analytics = {
      totalCases: casesStore.length,
      statusBreakdown: getStatusBreakdown(),
      priorityBreakdown: getPriorityBreakdown(),
      categoryBreakdown: getCategoryBreakdown()
    };

    const response = {
      success: true,
      data: paginatedResults,
      pagination: {
        total,
        limit,
        offset,
        hasNext: offset + limit < total,
        hasPrev: offset > 0
      },
      analytics,
      search: search ? {
        term: search,
        resultsCount: results.length,
        processingTime: Date.now() - startTime
      } : null,
      processingTime: Date.now() - startTime
    };

    console.log(`✅ Cases search completed: ${paginatedResults.length}/${total} results in ${Date.now() - startTime}ms`);
    
    return json(response);

  } catch (error) {
    console.error("❌ Cases search error:", error);
    return json({
      success: false,
      error: "Failed to search cases",
      message: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 });
  }
};

// POST - Create new case
export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();
  
  try {
    const body = await request.json();
    console.log("📝 Creating new case:", { title: body.title, category: body.category });

    // Validate input
    const validationResult = caseSchema.safeParse(body);
    if (!validationResult.success) {
      return json({
        success: false,
        error: "Invalid case data",
        details: validationResult.error.flatten()
      }, { status: 400 });
    }

    const caseData = validationResult.data;

    // Generate ID and timestamps
    const newCase = {
      ...caseData,
      id: crypto.randomUUID(),
      caseNumber: caseData.caseNumber || generateCaseNumber(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      userId: "current-user-id" // TODO: Get from auth context
    };

    // Add to store
    casesStore.push(newCase);

    console.log("✅ Case created successfully:", {
      id: newCase.id,
      caseNumber: newCase.caseNumber,
      processingTime: Date.now() - startTime
    });

    return json({
      success: true,
      message: "Case created successfully",
      data: newCase
    }, { status: 201 });

  } catch (error) {
    console.error("❌ Case creation error:", error);
    return json({
      success: false,
      error: "Failed to create case",
      message: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 });
  }
};

// Helper functions
function initializeSampleData() {
  if (casesStore.length === 0) {
    const sampleCases = [
      {
        id: crypto.randomUUID(),
        title: "Armed Robbery at Downtown Bank",
        caseNumber: "CR-2024-001",
        description: "Armed robbery occurred at First National Bank on Main Street. Three suspects involved, weapons recovered.",
        incidentDate: "2024-01-15",
        location: "123 Main Street, Downtown",
        priority: "high",
        status: "open",
        category: "violent_crime",
        dangerScore: 8,
        estimatedValue: 250000,
        jurisdiction: "Metro PD",
        leadProsecutor: "Sarah Johnson",
        assignedTeam: ["Det. Mike Smith", "Det. Jane Doe"],
        tags: ["armed_robbery", "weapons", "bank", "multiple_suspects"],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        userId: "user-1"
      },
      {
        id: crypto.randomUUID(),
        title: "Insurance Fraud Investigation", 
        caseNumber: "CR-2024-002",
        description: "Suspected insurance fraud involving falsified medical claims and staged accidents.",
        incidentDate: "2024-02-01",
        location: "Various locations",
        priority: "medium",
        status: "under_review",
        category: "white_collar",
        dangerScore: 2,
        estimatedValue: 500000,
        jurisdiction: "State AG",
        leadProsecutor: "Robert Chen",
        assignedTeam: ["Analyst Kate Wilson"],
        tags: ["fraud", "insurance", "medical", "staged"],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        userId: "user-2"
      },
      {
        id: crypto.randomUUID(),
        title: "Drug Trafficking Network",
        caseNumber: "CR-2024-003", 
        description: "Large-scale drug trafficking operation involving multiple states and international connections.",
        incidentDate: "2024-01-20",
        location: "Multi-state operation",
        priority: "urgent",
        status: "open",
        category: "drug_crime",
        dangerScore: 9,
        estimatedValue: 2000000,
        jurisdiction: "Federal DEA",
        leadProsecutor: "Michael Torres",
        assignedTeam: ["Agent Lisa Park", "Agent Tom Brady", "Analyst Jennifer Adams"],
        tags: ["drugs", "trafficking", "multi_state", "international", "organized_crime"],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        userId: "user-1"
      }
    ];

    casesStore.push(...sampleCases);
    console.log("📊 Initialized with sample case data");
  }
}

function generateCaseNumber(): string {
  const year = new Date().getFullYear();
  const sequence = casesStore.length + 1;
  return `CR-${year}-${sequence.toString().padStart(3, '0')}`;
}

function getStatusBreakdown() {
  const statuses = ["open", "closed", "pending", "archived", "under_review"];
  return statuses.reduce((acc, status) => {
    acc[status] = casesStore.filter(c => c.status === status).length;
    return acc;
  }, {} as Record<string, number>);
}

function getPriorityBreakdown() {
  const priorities = ["low", "medium", "high", "urgent"];
  return priorities.reduce((acc, priority) => {
    acc[priority] = casesStore.filter(c => c.priority === priority).length;
    return acc;
  }, {} as Record<string, number>);
}

function getCategoryBreakdown() {
  const categories = ["violent_crime", "white_collar", "drug_crime", "property_crime", "cyber_crime"];
  return categories.reduce((acc, category) => {
    acc[category] = casesStore.filter(c => c.category === category).length;
    return acc;
  }, {} as Record<string, number>);
}