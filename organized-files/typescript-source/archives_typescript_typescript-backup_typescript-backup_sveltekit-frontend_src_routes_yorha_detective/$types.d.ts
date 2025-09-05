import type { PageServerLoad } from './$types';

export interface PageData {
  systemData: {
    activeCases: number;
    evidenceItems: number;
    personsOfInterest: number;
    aiQueries: number;
    systemLoad: number;
    gpuUtilization: number;
    memoryUsage: number;
    networkLatency: number;
  };
  recentCases: Array<{
    id: string;
    title: string;
    description: string | null;
    status: string | null;
    priority: string | null;
    caseNumber: string | null;
    createdAt: Date;
    updatedAt: Date;
    createdBy: string | null;
    createdByLastName: string | null;
  }>;
  recentEvidence: Array<{
    id: string;
    title: string;
    evidenceType: string | null;
    createdAt: Date;
    caseTitle: string | null;
  }>;
  user: {
    id: string;
    firstName: string | null;
    lastName: string | null;
    role: string;
  };
}

export type { PageServerLoad };