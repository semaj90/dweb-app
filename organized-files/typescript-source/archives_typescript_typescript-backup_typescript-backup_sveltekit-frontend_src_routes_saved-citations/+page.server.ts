
import { caseService } from "$lib/services/caseService";
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ locals }) => {
  if (!locals.user) {
    return {
      savedCitations: [],
      user: null,
    };
  }
  try {
    // In a real app, this would fetch from the database
    // For now, we'll return the citations from the service
    let citations: any[] = [];

    // This is a mock - in reality you'd query the database
    const mockCitations = [
      {
        id: "1",
        title: "Fourth Amendment Search and Seizure",
        content:
          "The right of the people to be secure in their persons, houses, papers, and effects, against unreasonable searches and seizures, shall not be violated...",
        source: "U.S. Constitution, Amendment IV",
        tags: ["constitutional-law", "search-seizure", "evidence"],
        category: "constitutional",
        isFavorite: true,
        notes: "Key precedent for evidence admissibility",
        savedAt: new Date("2024-01-15"),
        contextData: { reportId: "report-123", caseId: "case-456" },
      },
      {
        id: "2",
        title: "Miranda Rights Requirement",
        content:
          "You have the right to remain silent. Anything you say can and will be used against you in a court of law...",
        source: "Miranda v. Arizona, 384 U.S. 436 (1966)",
        tags: ["miranda", "interrogation", "rights"],
        category: "case-law",
        isFavorite: false,
        notes: "Must be read before custodial interrogation",
        savedAt: new Date("2024-01-10"),
        contextData: { reportId: "report-124", caseId: "case-456" },
      },
    ];

    return {
      savedCitations: mockCitations,
      user: locals.user,
    };
  } catch (error: any) {
    console.error("Error loading saved citations:", error);
    return {
      savedCitations: [],
      user: locals.user,
      error: "Failed to load saved citations",
    };
  }
};
