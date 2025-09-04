/**
 * User-Scoped CRUD Operations for Legal AI Platform
 * Ensures all operations are restricted to user-owned entities
 * Integrates with advisory locking and transaction management
 */

import { db } from '$lib/server/db/pg';
import { users, reports, personsOfInterest } from '$lib/server/db/schema-postgres';
import { 
  cases, 
  evidence
} from '$lib/server/db/schema-postgres-enhanced';
// TODO: Implement concurrency modules
// import { transactionManager } from '../concurrency/transaction-manager';
// import { advisoryLocks, LOCK_MODES, LOCK_TYPES } from '../concurrency/advisory-locks';
// import { queueManager } from '../concurrency/queue-manager';
import { eq, and, desc, sql, count } from 'drizzle-orm';
import { z } from 'zod';
import type { User } from 'lucia';

// Zod schemas for validation with Superforms compatibility
export const CreateCaseSchema = z.object({
  title: z.string().min(1, 'Title is required').max(255),
  description: z.string().optional(),
  caseNumber: z.string().optional(),
  status: z.enum(['open', 'closed', 'pending', 'archived']).default('open'),
  priority: z.enum(['low', 'medium', 'high', 'urgent']).default('medium'),
  assignedTo: z.string().uuid().optional(),
  metadata: z.record(z.any()).default({})
});

export const UpdateCaseSchema = CreateCaseSchema.partial().extend({
  id: z.string().uuid()
});

export const CreateEvidenceSchema = z.object({
  caseId: z.string().uuid(),
  title: z.string().min(1, 'Title is required').max(255),
  description: z.string().optional(),
  evidenceType: z.string().min(1, 'Evidence type is required'),
  type: z.string().optional(),
  fileName: z.string().optional(),
  originalFileName: z.string().optional(),
  fileSize: z.string().optional(),
  fileType: z.string().optional(),
  filePath: z.string().optional(),
  tags: z.array(z.string()).default([]),
  metadata: z.record(z.any()).default({}),
  isPublic: z.boolean().default(false)
});

export const UpdateEvidenceSchema = CreateEvidenceSchema.partial().extend({
  id: z.string().uuid()
});

export const CreateReportSchema = z.object({
  title: z.string().min(1, 'Title is required').max(255),
  caseId: z.string().uuid().optional(),
  content: z.string().optional(),
  reportType: z.enum(['analysis', 'summary', 'investigation', 'final']).default('analysis'),
  status: z.enum(['draft', 'review', 'approved', 'published']).default('draft'),
  metadata: z.record(z.any()).default({})
});

export const UpdateReportSchema = CreateReportSchema.partial().extend({
  id: z.string().uuid()
});

export const CreatePersonOfInterestSchema = z.object({
  name: z.string().min(1, 'Name is required').max(255),
  aliases: z.array(z.string()).default([]),
  description: z.string().optional(),
  caseIds: z.array(z.string().uuid()).default([]),
  riskLevel: z.enum(['low', 'medium', 'high', 'critical']).default('low'),
  status: z.enum(['active', 'inactive', 'archived']).default('active'),
  contactInfo: z.record(z.any()).default({}),
  metadata: z.record(z.any()).default({})
});

export const UpdatePersonOfInterestSchema = CreatePersonOfInterestSchema.partial().extend({
  id: z.string().uuid()
});

// Type definitions
export type CreateCaseData = z.infer<typeof CreateCaseSchema>
export type UpdateCaseData = z.infer<typeof UpdateCaseSchema>
export type CreateEvidenceData = z.infer<typeof CreateEvidenceSchema>
export type UpdateEvidenceData = z.infer<typeof UpdateEvidenceSchema>
export type CreateReportData = z.infer<typeof CreateReportSchema>
export type UpdateReportData = z.infer<typeof UpdateReportSchema>
export type CreatePersonOfInterestData = z.infer<typeof CreatePersonOfInterestSchema>
export type UpdatePersonOfInterestData = z.infer<typeof UpdatePersonOfInterestSchema>

export interface PaginationOptions {
  page?: number;
  limit?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface PaginationResult<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}

/**
 * User-Scoped CRUD Service Base Class
 * Provides common functionality for all entity types
 */
class UserScopedCRUDService {
  protected userId: string;

  constructor(userId: string) {
    this.userId = userId;
  }

  /**
   * Verify user exists and is active
   */
  protected async verifyUser(): Promise<void> {
    const user = await db.select().from(users).where(
      and(eq(users.id, this.userId), eq(users.isActive, true))
    ).limit(1);

    if (!user.length) {
      throw new Error('User not found or inactive');
    }
  }

  /**
   * Create a paginated result
   */
  protected createPaginatedResult<T>(
    data: T[],
    total: number,
    page: number,
    limit: number
  ): PaginationResult<T> {
    return {
      data,
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit)
    };
  }
}

/**
 * Cases CRUD Service
 */
export class CasesCRUDService extends UserScopedCRUDService {
  
  async create(data: CreateCaseData): Promise<string> {
    await this.verifyUser();
    
    const validatedData = CreateCaseSchema.parse(data);
    
    return transactionManager.withCaseTransaction('new', async (ctx) => {
      const [newCase] = await db.insert(cases).values({
        ...validatedData,
        createdBy: this.userId,
        assignedTo: validatedData.assignedTo || this.userId
      }).returning({ id: cases.id });

      // Queue background processing
      await queueManager.enqueue({
        type: 'case_synthesis',
        entityType: 'case',
        entityId: newCase.id,
        userId: this.userId,
        payload: { caseId: newCase.id }
      });

      console.log(`📋 Created case ${newCase.id} for user ${this.userId}`);
      return newCase.id;
    });
  }

  async getById(caseId: string): Promise<any> {
    await this.verifyUser();
    
    return advisoryLocks.withLock('case', caseId, async () => {
      const [caseData] = await db.select().from(cases).where(
        and(
          eq(cases.id, caseId),
          eq(cases.createdBy, this.userId)
        )
      ).limit(1);

      if (!caseData) {
        throw new Error('Case not found or access denied');
      }

      return caseData;
    }, LOCK_MODES.SHARED);
  }

  async update(data: UpdateCaseData): Promise<void> {
    await this.verifyUser();
    
    const validatedData = UpdateCaseSchema.parse(data);
    const { id, ...updateData } = validatedData;

    return transactionManager.withCaseTransaction(id, async (ctx) => {
      // Verify ownership
      const [existing] = await db.select().from(cases).where(
        and(eq(cases.id, id), eq(cases.createdBy, this.userId))
      ).limit(1);

      if (!existing) {
        throw new Error('Case not found or access denied');
      }

      await db.update(cases)
        .set({ ...updateData, updatedAt: new Date() })
        .where(eq(cases.id, id));

      console.log(`📋 Updated case ${id} for user ${this.userId}`);
    });
  }

  async delete(caseId: string): Promise<void> {
    await this.verifyUser();
    
    return transactionManager.withCaseTransaction(caseId, async (ctx) => {
      // Verify ownership
      const [existing] = await db.select().from(cases).where(
        and(eq(cases.id, caseId), eq(cases.createdBy, this.userId))
      ).limit(1);

      if (!existing) {
        throw new Error('Case not found or access denied');
      }

      await db.delete(cases).where(eq(cases.id, caseId));
      console.log(`📋 Deleted case ${caseId} for user ${this.userId}`);
    });
  }

  async list(options: PaginationOptions = {}): Promise<PaginationResult<any>> {
    await this.verifyUser();
    
    const { page = 1, limit = 20, sortBy = 'createdAt', sortOrder = 'desc' } = options;
    const offset = (page - 1) * limit;

    const [data, totalResult] = await Promise.all([
      db.select().from(cases)
        .where(eq(cases.createdBy, this.userId))
        .orderBy(sortOrder === 'desc' ? desc(cases.createdAt) : cases.createdAt)
        .limit(limit)
        .offset(offset),
      
      db.select({ count: count() }).from(cases)
        .where(eq(cases.createdBy, this.userId))
    ]);

    return this.createPaginatedResult(data, totalResult[0].count, page, limit);
  }
}

/**
 * Evidence CRUD Service
 */
export class EvidenceCRUDService extends UserScopedCRUDService {
  
  async create(data: CreateEvidenceData): Promise<string> {
    await this.verifyUser();
    
    const validatedData = CreateEvidenceSchema.parse(data);
    
    // Verify user owns the case
    const [caseData] = await db.select().from(cases).where(
      and(eq(cases.id, validatedData.caseId), eq(cases.createdBy, this.userId))
    ).limit(1);

    if (!caseData) {
      throw new Error('Case not found or access denied');
    }

    return transactionManager.withTransactionAndLock(
      'evidence',
      'new',
      async (ctx) => {
        const [newEvidence] = await db.insert(evidence).values({
          ...validatedData,
          caseId: validatedData.caseId,
          evidenceType: validatedData.evidenceType,
          createdBy: this.userId
        }).returning({ id: evidence.id });

        // Queue background processing
        await queueManager.enqueue({
          type: 'evidence_analysis',
          entityType: 'evidence',
          entityId: newEvidence.id,
          userId: this.userId,
          payload: { evidenceId: newEvidence.id, caseId: validatedData.caseId }
        });

        console.log(`🔍 Created evidence ${newEvidence.id} for user ${this.userId}`);
        return newEvidence.id;
      },
      LOCK_MODES.EXCLUSIVE
    );
  }

  async getById(evidenceId: string): Promise<any> {
    await this.verifyUser();
    
    return advisoryLocks.withLock('evidence', evidenceId, async () => {
      const [evidenceData] = await db.select({
        ...evidence,
        case: {
          id: cases.id,
          title: cases.title
        }
      })
      .from(evidence)
      .leftJoin(cases, eq(evidence.caseId, cases.id))
      .where(
        and(
          eq(evidence.id, evidenceId),
          eq(evidence.createdBy, this.userId)
        )
      ).limit(1);

      if (!evidenceData) {
        throw new Error('Evidence not found or access denied');
      }

      return evidenceData;
    }, LOCK_MODES.SHARED);
  }

  async update(data: UpdateEvidenceData): Promise<void> {
    await this.verifyUser();
    
    const validatedData = UpdateEvidenceSchema.parse(data);
    const { id, caseId, ...updateData } = validatedData;

    return transactionManager.withTransactionAndLock(
      'evidence',
      id,
      async (ctx) => {
        // Verify ownership
        const [existing] = await db.select().from(evidence)
          .leftJoin(cases, eq(evidence.caseId, cases.id))
          .where(
            and(
              eq(evidence.id, id),
              eq(evidence.createdBy, this.userId)
            )
          ).limit(1);

        if (!existing) {
          throw new Error('Evidence not found or access denied');
        }

        await db.update(evidence)
          .set({ 
            ...updateData, 
            evidenceType: updateData.evidenceType || existing.evidence.evidenceType,
            updatedAt: new Date() 
          })
          .where(eq(evidence.id, id));

        console.log(`🔍 Updated evidence ${id} for user ${this.userId}`);
      },
      LOCK_MODES.EXCLUSIVE
    );
  }

  async delete(evidenceId: string): Promise<void> {
    await this.verifyUser();
    
    return transactionManager.withTransactionAndLock(
      'evidence',
      evidenceId,
      async (ctx) => {
        // Verify ownership
        const [existing] = await db.select().from(evidence).where(
          and(eq(evidence.id, evidenceId), eq(evidence.createdBy, this.userId))
        ).limit(1);

        if (!existing) {
          throw new Error('Evidence not found or access denied');
        }

        await db.delete(evidence).where(eq(evidence.id, evidenceId));
        console.log(`🔍 Deleted evidence ${evidenceId} for user ${this.userId}`);
      },
      LOCK_MODES.EXCLUSIVE
    );
  }

  async listByCase(caseId: string, options: PaginationOptions = {}): Promise<PaginationResult<any>> {
    await this.verifyUser();
    
    // Verify user owns the case
    const [caseData] = await db.select().from(cases).where(
      and(eq(cases.id, caseId), eq(cases.createdBy, this.userId))
    ).limit(1);

    if (!caseData) {
      throw new Error('Case not found or access denied');
    }

    const { page = 1, limit = 20 } = options;
    const offset = (page - 1) * limit;

    const [data, totalResult] = await Promise.all([
      db.select().from(evidence)
        .where(
          and(
            eq(evidence.caseId, caseId),
            eq(evidence.createdBy, this.userId)
          )
        )
        .orderBy(desc(evidence.createdAt))
        .limit(limit)
        .offset(offset),
      
      db.select({ count: count() }).from(evidence)
        .where(
          and(
            eq(evidence.caseId, caseId),
            eq(evidence.createdBy, this.userId)
          )
        )
    ]);

    return this.createPaginatedResult(data, totalResult[0].count, page, limit);
  }

  async list(options: PaginationOptions = {}): Promise<PaginationResult<any>> {
    await this.verifyUser();
    
    const { page = 1, limit = 20 } = options;
    const offset = (page - 1) * limit;

    const [data, totalResult] = await Promise.all([
      db.select({
        ...evidence,
        case: {
          id: cases.id,
          title: cases.title
        }
      })
      .from(evidence)
      .leftJoin(cases, eq(evidence.caseId, cases.id))
      .where(eq(evidence.createdBy, this.userId))
      .orderBy(desc(evidence.createdAt))
      .limit(limit)
      .offset(offset),
      
      db.select({ count: count() }).from(evidence)
        .where(eq(evidence.createdBy, this.userId))
    ]);

    return this.createPaginatedResult(data, totalResult[0].count, page, limit);
  }
}

/**
 * Reports CRUD Service
 */
export class ReportsCRUDService extends UserScopedCRUDService {
  
  async create(data: CreateReportData): Promise<string> {
    await this.verifyUser();
    
    const validatedData = CreateReportSchema.parse(data);
    
    // Verify user owns the case if caseId is provided
    if (validatedData.caseId) {
      const [caseData] = await db.select().from(cases).where(
        and(eq(cases.id, validatedData.caseId), eq(cases.createdBy, this.userId))
      ).limit(1);

      if (!caseData) {
        throw new Error('Case not found or access denied');
      }
    }

    return transactionManager.withTransaction(async (ctx) => {
      const [newReport] = await db.insert(reports).values({
        title: validatedData.title,
        content: validatedData.content,
        reportType: validatedData.reportType,
        status: validatedData.status,
        caseId: validatedData.caseId,
        createdBy: this.userId,
        metadata: validatedData.metadata
      }).returning({ id: reports.id });

      // Queue background processing
      await queueManager.enqueue({
        type: 'report_generation',
        entityType: 'document',
        entityId: newReport.id,
        userId: this.userId,
        payload: { reportId: newReport.id, caseId: validatedData.caseId }
      });

      console.log(`📄 Created report ${newReport.id} for user ${this.userId}`);
      return newReport.id;
    });
  }

  async getById(reportId: string): Promise<any> {
    await this.verifyUser();
    
    const [reportData] = await db.select({
      ...reports,
      case: {
        id: cases.id,
        title: cases.title
      }
    })
    .from(reports)
    .leftJoin(cases, eq(reports.caseId, cases.id))
    .where(
      and(
        eq(reports.id, reportId),
        eq(reports.createdBy, this.userId)
      )
    ).limit(1);

    if (!reportData) {
      throw new Error('Report not found or access denied');
    }

    return reportData;
  }

  async update(data: UpdateReportData): Promise<void> {
    await this.verifyUser();
    
    const validatedData = UpdateReportSchema.parse(data);
    const { id, ...updateData } = validatedData;

    return transactionManager.withTransaction(async (ctx) => {
      // Verify ownership
      const [existing] = await db.select().from(reports).where(
        and(eq(reports.id, id), eq(reports.createdBy, this.userId))
      ).limit(1);

      if (!existing) {
        throw new Error('Report not found or access denied');
      }

      await db.update(reports)
        .set({ 
          title: updateData.title || existing.title,
          content: updateData.content || existing.content,
          reportType: updateData.reportType || existing.reportType,
          status: updateData.status || existing.status,
          caseId: updateData.caseId || existing.caseId,
          metadata: updateData.metadata || existing.metadata,
          updatedAt: new Date() 
        })
        .where(eq(reports.id, id));

      console.log(`📄 Updated report ${id} for user ${this.userId}`);
    });
  }

  async delete(reportId: string): Promise<void> {
    await this.verifyUser();
    
    return transactionManager.withTransaction(async (ctx) => {
      // Verify ownership
      const [existing] = await db.select().from(reports).where(
        and(eq(reports.id, reportId), eq(reports.createdBy, this.userId))
      ).limit(1);

      if (!existing) {
        throw new Error('Report not found or access denied');
      }

      await db.delete(reports).where(eq(reports.id, reportId));
      console.log(`📄 Deleted report ${reportId} for user ${this.userId}`);
    });
  }

  async list(options: PaginationOptions = {}): Promise<PaginationResult<any>> {
    await this.verifyUser();
    
    const { page = 1, limit = 20 } = options;
    const offset = (page - 1) * limit;

    const [data, totalResult] = await Promise.all([
      db.select({
        ...reports,
        case: {
          id: cases.id,
          title: cases.title
        }
      })
      .from(reports)
      .leftJoin(cases, eq(reports.caseId, cases.id))
      .where(eq(reports.createdBy, this.userId))
      .orderBy(desc(reports.createdAt))
      .limit(limit)
      .offset(offset),
      
      db.select({ count: count() }).from(reports)
        .where(eq(reports.createdBy, this.userId))
    ]);

    return this.createPaginatedResult(data, totalResult[0].count, page, limit);
  }

  async listByCase(caseId: string, options: PaginationOptions = {}): Promise<PaginationResult<any>> {
    await this.verifyUser();
    
    // Verify user owns the case
    const [caseData] = await db.select().from(cases).where(
      and(eq(cases.id, caseId), eq(cases.createdBy, this.userId))
    ).limit(1);

    if (!caseData) {
      throw new Error('Case not found or access denied');
    }

    const { page = 1, limit = 20 } = options;
    const offset = (page - 1) * limit;

    const [data, totalResult] = await Promise.all([
      db.select().from(reports)
        .where(
          and(
            eq(reports.caseId, caseId),
            eq(reports.createdBy, this.userId)
          )
        )
        .orderBy(desc(reports.createdAt))
        .limit(limit)
        .offset(offset),
      
      db.select({ count: count() }).from(reports)
        .where(
          and(
            eq(reports.caseId, caseId),
            eq(reports.createdBy, this.userId)
          )
        )
    ]);

    return this.createPaginatedResult(data, totalResult[0].count, page, limit);
  }
}

/**
 * Persons of Interest CRUD Service
 */
export class PersonsOfInterestCRUDService extends UserScopedCRUDService {
  
  async create(data: CreatePersonOfInterestData): Promise<string> {
    await this.verifyUser();
    
    const validatedData = CreatePersonOfInterestSchema.parse(data);
    
    // Verify user owns all referenced cases
    if (validatedData.caseIds.length > 0) {
      const ownedCases = await db.select({ id: cases.id }).from(cases)
        .where(
          and(
            eq(cases.createdBy, this.userId),
            sql`${cases.id} = ANY(${validatedData.caseIds})`
          )
        );
      
      if (ownedCases.length !== validatedData.caseIds.length) {
        throw new Error('One or more cases not found or access denied');
      }
    }

    return transactionManager.withTransaction(async (ctx) => {
      const [newPerson] = await db.insert(personsOfInterest).values({
        name: validatedData.name,
        aliases: validatedData.aliases,
        description: validatedData.description,
        caseId: validatedData.caseIds,
        risk_level: validatedData.riskLevel,
        status: validatedData.status,
        contact_info: validatedData.contactInfo,
        createdBy: this.userId,
        metadata: validatedData.metadata
      }).returning({ id: personsOfInterest.id });

      console.log(`👤 Created person of interest ${newPerson.id} for user ${this.userId}`);
      return newPerson.id;
    });
  }

  async getById(personId: string): Promise<any> {
    await this.verifyUser();
    
    const [personData] = await db.select().from(personsOfInterest).where(
      and(
        eq(personsOfInterest.id, personId),
        eq(personsOfInterest.createdBy, this.userId)
      )
    ).limit(1);

    if (!personData) {
      throw new Error('Person of interest not found or access denied');
    }

    return personData;
  }

  async update(data: UpdatePersonOfInterestData): Promise<void> {
    await this.verifyUser();
    
    const validatedData = UpdatePersonOfInterestSchema.parse(data);
    const { id, caseIds, ...updateData } = validatedData;

    return transactionManager.withTransaction(async (ctx) => {
      // Verify ownership
      const [existing] = await db.select().from(personsOfInterest).where(
        and(eq(personsOfInterest.id, id), eq(personsOfInterest.createdBy, this.userId))
      ).limit(1);

      if (!existing) {
        throw new Error('Person of interest not found or access denied');
      }

      // Verify user owns all referenced cases if caseIds provided
      if (caseIds && caseIds.length > 0) {
        const ownedCases = await db.select({ id: cases.id }).from(cases)
          .where(
            and(
              eq(cases.createdBy, this.userId),
              sql`${cases.id} = ANY(${caseIds})`
            )
          );
        
        if (ownedCases.length !== caseIds.length) {
          throw new Error('One or more cases not found or access denied');
        }
      }

      await db.update(personsOfInterest)
        .set({ 
          name: updateData.name || existing.name,
          aliases: updateData.aliases || existing.aliases,
          description: updateData.description || existing.description,
          caseId: caseIds || existing.caseId,
          risk_level: updateData.riskLevel || existing.risk_level,
          status: updateData.status || existing.status,
          contact_info: updateData.contactInfo || existing.contact_info,
          metadata: updateData.metadata || existing.metadata,
          updatedAt: new Date() 
        })
        .where(eq(personsOfInterest.id, id));

      console.log(`👤 Updated person of interest ${id} for user ${this.userId}`);
    });
  }

  async delete(personId: string): Promise<void> {
    await this.verifyUser();
    
    return transactionManager.withTransaction(async (ctx) => {
      // Verify ownership
      const [existing] = await db.select().from(personsOfInterest).where(
        and(eq(personsOfInterest.id, personId), eq(personsOfInterest.createdBy, this.userId))
      ).limit(1);

      if (!existing) {
        throw new Error('Person of interest not found or access denied');
      }

      await db.delete(personsOfInterest).where(eq(personsOfInterest.id, personId));
      console.log(`👤 Deleted person of interest ${personId} for user ${this.userId}`);
    });
  }

  async list(options: PaginationOptions = {}): Promise<PaginationResult<any>> {
    await this.verifyUser();
    
    const { page = 1, limit = 20 } = options;
    const offset = (page - 1) * limit;

    const [data, totalResult] = await Promise.all([
      db.select().from(personsOfInterest)
        .where(eq(personsOfInterest.createdBy, this.userId))
        .orderBy(desc(personsOfInterest.createdAt))
        .limit(limit)
        .offset(offset),
      
      db.select({ count: count() }).from(personsOfInterest)
        .where(eq(personsOfInterest.createdBy, this.userId))
    ]);

    return this.createPaginatedResult(data, totalResult[0].count, page, limit);
  }

  async listByRiskLevel(riskLevel: string, options: PaginationOptions = {}): Promise<PaginationResult<any>> {
    await this.verifyUser();
    
    const { page = 1, limit = 20 } = options;
    const offset = (page - 1) * limit;

    const [data, totalResult] = await Promise.all([
      db.select().from(personsOfInterest)
        .where(
          and(
            eq(personsOfInterest.createdBy, this.userId),
            eq(personsOfInterest.risk_level, riskLevel)
          )
        )
        .orderBy(desc(personsOfInterest.createdAt))
        .limit(limit)
        .offset(offset),
      
      db.select({ count: count() }).from(personsOfInterest)
        .where(
          and(
            eq(personsOfInterest.createdBy, this.userId),
            eq(personsOfInterest.risk_level, riskLevel)
          )
        )
    ]);

    return this.createPaginatedResult(data, totalResult[0].count, page, limit);
  }
}

/**
 * Factory function to create user-scoped CRUD services
 */
export function createUserCRUDServices(userId: string) {
  return {
    cases: new CasesCRUDService(userId),
    evidence: new EvidenceCRUDService(userId),
    reports: new ReportsCRUDService(userId),
    personsOfInterest: new PersonsOfInterestCRUDService(userId)
  };
}

// Export individual service classes for direct use
export { 
  CasesCRUDService, 
  EvidenceCRUDService, 
  ReportsCRUDService, 
  PersonsOfInterestCRUDService 
};