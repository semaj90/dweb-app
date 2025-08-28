/**
 * Enhanced Profile Page Server Load
 * Integrates PostgreSQL + pgvector + Drizzle + Qdrant + Neo4j + Cognitive Cache
 */

import type { PageServerLoad } from './$types';
import { redirect } from '@sveltejs/kit';
import { db } from '$lib/server/db';
import { users, cases, evidence, legal_documents } from '$lib/server/db/schema-postgres';
import { eq, desc, sql, count, and, or } from 'drizzle-orm';
import { getDatabaseHealth } from '../../../lib/database';
import { cognitiveCacheManager } from '$lib/services/cognitive-cache-integration';

export const load: PageServerLoad = async ({ locals }) => {
  // Authentication check
  if (!locals.user) {
    throw redirect(302, '/login');
  }

  const userId = locals.user.id;
  
  // Check cognitive cache for user profile data
  const cacheKey = `user_profile_${userId}`;
  const cacheRequest = {
    key: cacheKey,
    type: 'legal-data' as const,
    context: {
      userId,
      workflowStep: 'profile-load',
      documentType: 'user-profile',
      priority: 'medium' as const,
      semanticTags: ['user-profile', 'legal-professional']
    }
  };

  // Try cognitive cache first
  const cachedProfile = await cognitiveCacheManager.get(cacheRequest);
  if (cachedProfile && cachedProfile.confidence > 0.7) {
    return {
      user: locals.user,
      ...cachedProfile.data,
      loadSource: 'cache',
      loadTime: cachedProfile.processingTime
    };
  }

  // Get database health status
  const healthStatus = await getDatabaseHealth();

  // Parallel database queries for user profile data
  const [userProfile, userCases, userEvidence, userDocuments, activityStats] = await Promise.all([
    // Enhanced user profile with vector embeddings
    db.select({
      id: users.id,
      email: users.email,
      username: users.username,
      firstName: users.first_name,
      lastName: users.last_name,
      role: users.role,
      department: users.department,
      jurisdiction: users.jurisdiction,
      practiceAreas: users.practice_areas,
      barNumber: users.bar_number,
      firmName: users.firm_name,
      avatarUrl: users.avatar_url,
      lastLoginAt: users.last_login_at,
      profileEmbedding: users.profile_embedding,
      metadata: users.metadata,
      createdAt: users.created_at,
      updatedAt: users.updated_at
    }).from(users).where(eq(users.id, userId)).limit(1),

    // Recent cases handled by user
    db.select({
      id: cases.id,
      title: cases.title,
      caseNumber: cases.case_number,
      status: cases.status,
      priority: cases.priority,
      createdAt: cases.created_at,
      updatedAt: cases.updated_at
    }).from(cases)
      .where(eq(cases.created_by, userId))
      .orderBy(desc(cases.updated_at))
      .limit(10),

    // Evidence items associated with user
    db.select({
      id: evidence.id,
      title: evidence.title,
      evidenceType: evidence.evidence_type,
      caseId: evidence.case_id,
      status: evidence.status,
      createdAt: evidence.created_at
    }).from(evidence)
      .where(eq(evidence.created_by, userId))
      .orderBy(desc(evidence.created_at))
      .limit(10),

    // Documents processed by user
    db.select({
      id: legal_documents.id,
      title: legal_documents.title,
      documentType: legal_documents.document_type,
      jurisdiction: legal_documents.jurisdiction,
      practiceArea: legal_documents.practice_area,
      processingStatus: legal_documents.processing_status,
      createdAt: legal_documents.created_at
    }).from(legal_documents)
      .where(eq(legal_documents.created_by, userId))
      .orderBy(desc(legal_documents.created_at))
      .limit(10),

    // Activity statistics
    db.select({
      totalCases: count(cases.id),
      activeCases: sql<number>`COUNT(CASE WHEN ${cases.status} IN ('open', 'active') THEN 1 END)`,
      closedCases: sql<number>`COUNT(CASE WHEN ${cases.status} = 'closed' THEN 1 END)`,
      totalEvidence: sql<number>`(
        SELECT COUNT(*) FROM ${evidence} WHERE ${evidence.created_by} = ${userId}
      )`,
      totalDocuments: sql<number>`(
        SELECT COUNT(*) FROM ${legal_documents} WHERE ${legal_documents.created_by} = ${userId}
      )`
    }).from(cases)
      .where(eq(cases.created_by, userId))
  ]);

  // Vector similarity search for similar legal professionals (if profile has embedding)
  let similarProfessionals = [];
  if (userProfile[0]?.profileEmbedding) {
    try {
      similarProfessionals = await db.select({
        id: users.id,
        username: users.username,
        firstName: users.first_name,
        lastName: users.last_name,
        role: users.role,
        department: users.department,
        practiceAreas: users.practice_areas,
        similarity: sql<number>`1 - (${users.profile_embedding} <=> ${userProfile[0].profileEmbedding})`
      }).from(users)
        .where(
          and(
            sql`${users.profile_embedding} IS NOT NULL`,
            sql`${users.id} != ${userId}`,
            sql`1 - (${users.profile_embedding} <=> ${userProfile[0].profileEmbedding}) > 0.7`
          )
        )
        .orderBy(sql`${users.profile_embedding} <=> ${userProfile[0].profileEmbedding}`)
        .limit(5);
    } catch (error) {
      console.warn('Vector similarity search failed:', error);
      similarProfessionals = [];
    }
  }

  const profileData = {
    user: locals.user,
    userProfile: userProfile[0] || null,
    recentCases: userCases,
    recentEvidence: userEvidence,
    recentDocuments: userDocuments,
    activityStats: activityStats[0],
    similarProfessionals,
    databaseHealth: healthStatus,
    loadSource: 'database',
    loadTime: Date.now()
  };

  // Cache the profile data for future requests
  await cognitiveCacheManager.set(cacheRequest, profileData, {
    distributeAcrossCaches: true,
    cognitiveValue: 0.8
  });

  return profileData;
};