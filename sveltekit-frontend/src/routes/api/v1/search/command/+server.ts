import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db, sql } from '$lib/server/db';
import { cases, evidence, legalDocuments, users } from '$lib/server/db';
import { eq, and, or, ilike, desc, count } from 'drizzle-orm';
import { vectorOps } from '$lib/server/db/enhanced-vector-operations';
import type { CommandSearchRequest, CommandSearchResponse } from '$lib/types/api';

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const user = locals.user;
    if (!user) {
      return json({
        success: false,
        error: 'Unauthorized'
      }, { status: 401 });
    }

    const body: CommandSearchRequest = await request.json();
    const { 
      query, 
      types = ['cases', 'evidence', 'documents', 'people'], 
      limit = 10,
      userId 
    } = body;

    if (!query || query.trim().length < 2) {
      return json({
        success: false,
        error: 'Query must be at least 2 characters long'
      }, { status: 400 });
    }

    const searchQuery = query.trim();
    const results: CommandSearchResponse['results'] = {
      cases: [],
      evidence: [],
      documents: [],
      people: []
    };

    let totalResults = 0;

    // Search Cases
    if (types.includes('cases')) {
      try {
        const caseResults = await db
          .select()
          .from(cases)
          .where(
            and(
              eq(cases.userId, userId || user.id),
              or(
                ilike(cases.title, `%${searchQuery}%`),
                ilike(cases.description, `%${searchQuery}%`),
                ilike(cases.location, `%${searchQuery}%`),
                ilike(cases.jurisdiction, `%${searchQuery}%`)
              )
            )
          )
          .orderBy(desc(cases.updatedAt))
          .limit(limit);

        results.cases = caseResults.map(case_ => ({
          ...case_,
          similarity: calculateSimilarity(searchQuery, case_.title + ' ' + (case_.description || ''))
        }));

        totalResults += caseResults.length;
      } catch (error) {
        console.error('Error searching cases:', error);
      }
    }

    // Search Evidence
    if (types.includes('evidence')) {
      try {
        const evidenceResults = await db
          .select({
            ...evidence,
            caseTitle: cases.title
          })
          .from(evidence)
          .leftJoin(cases, eq(evidence.caseId, cases.id))
          .where(
            and(
              eq(cases.userId, userId || user.id),
              or(
                ilike(evidence.title, `%${searchQuery}%`),
                ilike(evidence.description, `%${searchQuery}%`),
                sql`${evidence.tags}::text ILIKE ${`%${searchQuery}%`}`
              )
            )
          )
          .orderBy(desc(evidence.updatedAt))
          .limit(limit);

        results.evidence = evidenceResults.map(item => ({
          ...item,
          similarity: calculateSimilarity(searchQuery, item.title + ' ' + (item.description || ''))
        }));

        totalResults += evidenceResults.length;
      } catch (error) {
        console.error('Error searching evidence:', error);
      }
    }

    // Search Documents
    if (types.includes('documents')) {
      try {
        const documentResults = await db
          .select()
          .from(legalDocuments)
          .where(
            or(
              ilike(legalDocuments.title, `%${searchQuery}%`),
              ilike(legalDocuments.content, `%${searchQuery}%`),
              ilike(legalDocuments.documentType, `%${searchQuery}%`)
            )
          )
          .orderBy(desc(legalDocuments.updatedAt))
          .limit(limit);

        results.documents = documentResults.map(doc => ({
          ...doc,
          similarity: calculateSimilarity(searchQuery, doc.title + ' ' + (doc.content || '').substring(0, 500))
        }));

        totalResults += documentResults.length;
      } catch (error) {
        console.error('Error searching documents:', error);
      }
    }

    // Search People
    if (types.includes('people')) {
      try {
        const userResults = await db
          .select()
          .from(users)
          .where(
            or(
              ilike(users.name, `%${searchQuery}%`),
              ilike(users.email, `%${searchQuery}%`),
              ilike(users.firstName, `%${searchQuery}%`),
              ilike(users.lastName, `%${searchQuery}%`),
              ilike(users.department, `%${searchQuery}%`)
            )
          )
          .orderBy(desc(users.updatedAt))
          .limit(limit);

        results.people = userResults.map(person => ({
          ...person,
          similarity: calculateSimilarity(searchQuery, person.name + ' ' + person.email + ' ' + (person.department || ''))
        }));

        totalResults += userResults.length;
      } catch (error) {
        console.error('Error searching people:', error);
      }
    }

    // Enhanced vector search (if available and requested)
    if (query.length >= 5) {
      try {
        const vectorResults = await vectorOps.performRAGSearch({
          query: searchQuery,
          userId: userId || user.id,
          limit: 5
        });

        // Merge vector results with existing results
        for (const result of vectorResults) {
          const type = result.metadata?.type;
          if (type && results[type as keyof typeof results]) {
            const existing = results[type as keyof typeof results] as any[];
            const existingIds = existing.map(item => item.id);
            
            if (!existingIds.includes(result.id)) {
              // Add vector result with high similarity score
              existing.push({
                id: result.id,
                ...result.metadata,
                content: result.content,
                similarity: result.similarity
              });
            }
          }
        }
      } catch (error) {
        console.warn('Vector search failed:', error);
      }
    }

    // Sort all results by similarity
    Object.keys(results).forEach(key => {
      const resultArray = results[key as keyof typeof results] as any[];
      resultArray.sort((a, b) => (b.similarity || 0) - (a.similarity || 0));
    });

    const response: CommandSearchResponse = {
      success: true,
      data: {
        results,
        totalResults
      },
      timestamp: new Date().toISOString()
    };

    return json(response);

  } catch (error) {
    console.error('Command search error:', error);
    return json({
      success: false,
      error: 'Internal server error',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

// Simple similarity calculation (can be enhanced with more sophisticated algorithms)
function calculateSimilarity(query: string, text: string): number {
  const queryLower = query.toLowerCase();
  const textLower = text.toLowerCase();
  
  // Exact match gets highest score
  if (textLower.includes(queryLower)) {
    const position = textLower.indexOf(queryLower);
    // Earlier matches get higher scores
    return Math.max(0.8, 1 - (position / text.length * 0.2));
  }
  
  // Word-based matching
  const queryWords = queryLower.split(/\s+/);
  const textWords = textLower.split(/\s+/);
  
  let matchCount = 0;
  for (const queryWord of queryWords) {
    for (const textWord of textWords) {
      if (textWord.includes(queryWord) || queryWord.includes(textWord)) {
        matchCount++;
        break;
      }
    }
  }
  
  return Math.min(0.7, matchCount / queryWords.length * 0.7);
}