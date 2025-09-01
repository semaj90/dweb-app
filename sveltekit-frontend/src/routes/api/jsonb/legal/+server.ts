import type { RequestHandler } from './$types.js';

/**
 * JSONB Legal API Endpoints
 * 
 * RESTful API for legal metadata operations using optimized JSONB schema.
 * Provides endpoints for documents, cases, evidence, and analytics.
 * 
 * Integrates with:
 * - Structured logging system
 * - Vector embeddings
 * - Graph relationships
 * - Chain of custody verification
 */

import { json, error } from '@sveltejs/kit';
import { jsonbLegalService } from '$lib/services/jsonb-legal-service.js';
import { logger } from '$lib/logging/structured-logger.js';
import { z } from 'zod';
import path from "path";
import crypto from "crypto";
import { URL } from "url";

// ============================================================================
// REQUEST VALIDATION SCHEMAS
// ============================================================================

const CreateDocumentSchema = z.object({
  title: z.string().min(1).max(500),
  content: z.string().min(1),
  metadata: z.object({
    documentType: z.enum(['contract', 'brief', 'motion', 'pleading', 'evidence', 'citation', 'precedent', 'statute']),
    jurisdiction: z.string().optional(),
    practiceArea: z.enum(['corporate', 'litigation', 'criminal', 'intellectual_property', 'real_estate', 'family', 'tax', 'employment']).optional(),
    confidentialityLevel: z.enum(['public', 'confidential', 'privileged', 'classified']).default('public'),
    urgency: z.enum(['routine', 'priority', 'urgent', 'emergency']).default('routine'),
    parties: z.array(z.object({
      name: z.string(),
      role: z.enum(['plaintiff', 'defendant', 'witness', 'counsel', 'judge', 'expert', 'third_party']),
      entityType: z.enum(['individual', 'corporation', 'government', 'organization']).optional()
    })).optional(),
    semantics: z.object({
      keyTerms: z.array(z.string()).optional(),
      legalConcepts: z.array(z.string()).optional(),
      precedentStrength: z.number().min(0).max(1).optional()
    }).optional(),
    aiMetadata: z.object({
      modelVersion: z.string().optional(),
      processingTimestamp: z.string().datetime().optional(),
      confidence: z.number().min(0).max(1).optional(),
      reviewStatus: z.enum(['pending', 'reviewed', 'approved', 'rejected']).default('pending'),
      humanVerified: z.boolean().default(false)
    }).optional()
  }),
  embeddings: z.object({
    title: z.array(z.number()).optional(),
    content: z.array(z.number()).optional()
  }).optional()
});

const CreateCaseSchema = z.object({
  title: z.string().min(1).max(500),
  description: z.string().optional(),
  metadata: z.object({
    caseNumber: z.string(),
    status: z.enum(['active', 'pending', 'closed', 'on_hold', 'appealed']),
    filingDate: z.string().datetime().optional(),
    timeline: z.array(z.object({
      date: z.string().datetime(),
      event: z.string(),
      significance: z.enum(['low', 'medium', 'high', 'critical']).default('medium')
    })).optional(),
    strategy: z.object({
      approach: z.string().optional(),
      objectives: z.array(z.string()).optional(),
      risks: z.array(z.object({
        description: z.string(),
        probability: z.number().min(0).max(1),
        impact: z.enum(['low', 'medium', 'high', 'critical'])
      })).optional()
    }).optional()
  })
});

const CreateEvidenceSchema = z.object({
  caseId: z.string().uuid(),
  title: z.string().min(1).max(500),
  description: z.string().optional(),
  filePath: z.string().optional(),
  fileSize: z.number().int().positive().optional(),
  mimeType: z.string().optional(),
  metadata: z.object({
    evidenceType: z.enum(['document', 'physical', 'digital', 'testimony', 'expert_opinion', 'demonstrative']),
    authenticity: z.object({
      verified: z.boolean().default(false),
      method: z.string().optional(),
      verifier: z.string().optional(),
      verificationDate: z.string().datetime().optional()
    }).optional(),
    chainOfCustody: z.array(z.object({
      timestamp: z.string().datetime(),
      custodian: z.string(),
      action: z.enum(['collected', 'transferred', 'analyzed', 'stored', 'retrieved']),
      location: z.string().optional(),
      condition: z.string().optional()
    })).optional(),
    relevance: z.object({
      score: z.number().min(0).max(1),
      reasoning: z.string().optional(),
      relatedEvidence: z.array(z.string()).optional()
    }).optional(),
    admissibility: z.object({
      status: z.enum(['admissible', 'inadmissible', 'conditional', 'pending']),
      basis: z.string().optional(),
      objections: z.array(z.string()).optional()
    }).optional()
  }),
  embedding: z.array(z.number()).optional()
});

const DocumentSearchSchema = z.object({
  documentTypes: z.array(z.string()).optional(),
  practiceAreas: z.array(z.string()).optional(),
  jurisdictions: z.array(z.string()).optional(),
  confidentialityLevels: z.array(z.string()).optional(),
  minConfidence: z.number().min(0).max(1).optional(),
  hasHumanVerification: z.boolean().optional(),
  dateRange: z.object({
    start: z.string().datetime(),
    end: z.string().datetime()
  }).optional(),
  keyTerms: z.array(z.string()).optional(),
  parties: z.array(z.object({
    name: z.string(),
    role: z.string().optional()
  })).optional(),
  limit: z.number().int().positive().max(1000).default(50),
  offset: z.number().int().min(0).default(0)
});

// ============================================================================
// API HANDLERS
// ============================================================================

export const GET: RequestHandler = async ({ url, request }) => {
  const requestId = crypto.randomUUID();
  const startTime = performance.now();

  try {
    const path = url.pathname.split('/').pop();
    
    await logger.logAPIRequest({
      requestId,
      method: 'GET',
      endpoint: `/api/jsonb/legal/${path}`,
      userAgent: request.headers.get('user-agent') || 'unknown',
      ipAddress: request.headers.get('x-forwarded-for') || 'unknown',
      headers: Object.fromEntries(request.headers.entries())
    });

    switch (path) {
      case 'analytics':
        const analytics = await jsonbLegalService.getLegalAnalytics();
        const duration = performance.now() - startTime;
        
        await logger.logAPIResponse({
          requestId,
          statusCode: 200,
          responseSize: JSON.stringify(analytics).length,
          processingTime: duration,
          success: true
        });

        return json({
          success: true,
          data: analytics,
          metadata: {
            requestId,
            processingTime: duration,
            timestamp: new Date().toISOString()
          }
        });

      case 'performance':
        const performance_metrics = await jsonbLegalService.getPerformanceMetrics();
        const perfDuration = performance.now() - startTime;
        
        await logger.logAPIResponse({
          requestId,
          statusCode: 200,
          responseSize: JSON.stringify(performance_metrics).length,
          processingTime: perfDuration,
          success: true
        });

        return json({
          success: true,
          data: performance_metrics,
          metadata: {
            requestId,
            processingTime: perfDuration,
            timestamp: new Date().toISOString()
          }
        });

      default:
        throw error(404, `Endpoint not found: ${path}`);
    }
  } catch (err: any) {
    const duration = performance.now() - startTime;
    const errorMessage = err instanceof Error ? err.message : 'Unknown error';
    
    await logger.logAPIResponse({
      requestId,
      statusCode: 500,
      responseSize: 0,
      processingTime: duration,
      success: false,
      error: errorMessage
    });

    await logger.logError({
      error: errorMessage,
      context: 'jsonb_legal_api_get',
      requestId,
      severity: 'high',
      category: 'api'
    });

    throw error(500, errorMessage);
  }
};

export const POST: RequestHandler = async ({ request, url }) => {
  const requestId = crypto.randomUUID();
  const startTime = performance.now();
  let requestBody: any;

  try {
    requestBody = await request.json();
    const path = url.pathname.split('/').slice(-2); // Get last two parts: ['legal', 'documents']
    const operation = path[1] || path[0];

    await logger.logAPIRequest({
      requestId,
      method: 'POST',
      endpoint: `/api/jsonb/legal/${operation}`,
      userAgent: request.headers.get('user-agent') || 'unknown',
      ipAddress: request.headers.get('x-forwarded-for') || 'unknown',
      requestBody,
      headers: Object.fromEntries(request.headers.entries())
    });

    switch (operation) {
      case 'documents':
        // Create legal document
        const documentData = CreateDocumentSchema.parse(requestBody);
        const document = await jsonbLegalService.createLegalDocument(
          {
            title: documentData.title,
            content: documentData.content,
            metadata: documentData.metadata
          },
          documentData.embeddings ? {
            title: documentData.embeddings.title || [],
            content: documentData.embeddings.content || []
          } : undefined
        );
        
        const docDuration = performance.now() - startTime;
        
        await logger.logAPIResponse({
          requestId,
          statusCode: 201,
          responseSize: JSON.stringify(document).length,
          processingTime: docDuration,
          success: true
        });

        return json({
          success: true,
          data: document,
          metadata: {
            requestId,
            processingTime: docDuration,
            timestamp: new Date().toISOString()
          }
        }, { status: 201 });

      case 'cases':
        // Create case
        const caseData = CreateCaseSchema.parse(requestBody);
        const caseRecord = await jsonbLegalService.createCase({
          title: caseData.title,
          description: caseData.description,
          metadata: caseData.metadata
        });
        
        const caseDuration = performance.now() - startTime;
        
        await logger.logAPIResponse({
          requestId,
          statusCode: 201,
          responseSize: JSON.stringify(caseRecord).length,
          processingTime: caseDuration,
          success: true
        });

        return json({
          success: true,
          data: caseRecord,
          metadata: {
            requestId,
            processingTime: caseDuration,
            timestamp: new Date().toISOString()
          }
        }, { status: 201 });

      case 'evidence':
        // Create evidence
        const evidenceData = CreateEvidenceSchema.parse(requestBody);
        const evidence = await jsonbLegalService.createEvidence(
          {
            caseId: evidenceData.caseId,
            title: evidenceData.title,
            description: evidenceData.description,
            filePath: evidenceData.filePath,
            fileSize: evidenceData.fileSize,
            mimeType: evidenceData.mimeType,
            metadata: evidenceData.metadata
          },
          evidenceData.embedding
        );
        
        const evidenceDuration = performance.now() - startTime;
        
        await logger.logAPIResponse({
          requestId,
          statusCode: 201,
          responseSize: JSON.stringify(evidence).length,
          processingTime: evidenceDuration,
          success: true
        });

        return json({
          success: true,
          data: evidence,
          metadata: {
            requestId,
            processingTime: evidenceDuration,
            timestamp: new Date().toISOString()
          }
        }, { status: 201 });

      case 'search':
        // Search documents by criteria
        const searchCriteria = DocumentSearchSchema.parse(requestBody);
        const searchCriteriaWithDates = {
          ...searchCriteria,
          dateRange: searchCriteria.dateRange ? {
            start: new Date(searchCriteria.dateRange.start),
            end: new Date(searchCriteria.dateRange.end)
          } : undefined
        };
        
        const searchResults = await jsonbLegalService.findDocumentsByLegalCriteria(searchCriteriaWithDates);
        const searchDuration = performance.now() - startTime;
        
        await logger.logAPIResponse({
          requestId,
          statusCode: 200,
          responseSize: JSON.stringify(searchResults).length,
          processingTime: searchDuration,
          success: true
        });

        return json({
          success: true,
          data: searchResults,
          metadata: {
            requestId,
            processingTime: searchDuration,
            timestamp: new Date().toISOString()
          }
        });

      case 'concepts':
        // Analyze legal concepts
        const { documentIds } = z.object({
          documentIds: z.array(z.string().uuid()).min(1).max(100)
        }).parse(requestBody);
        
        const conceptAnalysis = await jsonbLegalService.analyzeLegalConcepts(documentIds);
        const conceptDuration = performance.now() - startTime;
        
        await logger.logAPIResponse({
          requestId,
          statusCode: 200,
          responseSize: JSON.stringify(conceptAnalysis).length,
          processingTime: conceptDuration,
          success: true
        });

        return json({
          success: true,
          data: conceptAnalysis,
          metadata: {
            requestId,
            processingTime: conceptDuration,
            timestamp: new Date().toISOString()
          }
        });

      case 'similar-cases':
        // Find similar cases
        const { caseId, threshold } = z.object({
          caseId: z.string().uuid(),
          threshold: z.number().min(0).max(1).default(0.8)
        }).parse(requestBody);
        
        const similarCases = await jsonbLegalService.findSimilarCases(caseId, threshold);
        const similarDuration = performance.now() - startTime;
        
        await logger.logAPIResponse({
          requestId,
          statusCode: 200,
          responseSize: JSON.stringify(similarCases).length,
          processingTime: similarDuration,
          success: true
        });

        return json({
          success: true,
          data: similarCases,
          metadata: {
            requestId,
            processingTime: similarDuration,
            timestamp: new Date().toISOString()
          }
        });

      case 'citation-network':
        // Build citation network
        const { documentId, depth } = z.object({
          documentId: z.string().uuid(),
          depth: z.number().int().min(1).max(5).default(2)
        }).parse(requestBody);
        
        const citationNetwork = await jsonbLegalService.buildCitationNetwork(documentId, depth);
        const networkDuration = performance.now() - startTime;
        
        await logger.logAPIResponse({
          requestId,
          statusCode: 200,
          responseSize: JSON.stringify(citationNetwork).length,
          processingTime: networkDuration,
          success: true
        });

        return json({
          success: true,
          data: citationNetwork,
          metadata: {
            requestId,
            processingTime: networkDuration,
            timestamp: new Date().toISOString()
          }
        });

      default:
        throw error(404, `Operation not found: ${operation}`);
    }
  } catch (err: any) {
    const duration = performance.now() - startTime;
    let errorMessage = 'Unknown error';
    let statusCode = 500;

    if (err instanceof z.ZodError) {
      errorMessage = `Validation error: ${err.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', ')}`;
      statusCode = 400;
    } else if (err instanceof Error) {
      errorMessage = err.message;
      if (err.message.includes('not found')) {
        statusCode = 404;
      }
    }
    
    await logger.logAPIResponse({
      requestId,
      statusCode,
      responseSize: 0,
      processingTime: duration,
      success: false,
      error: errorMessage
    });

    await logger.logError({
      error: errorMessage,
      context: 'jsonb_legal_api_post',
      requestId,
      requestBody,
      severity: statusCode >= 500 ? 'high' : 'medium',
      category: 'api'
    });

    return json({
      success: false,
      error: errorMessage,
      metadata: {
        requestId,
        processingTime: duration,
        timestamp: new Date().toISOString()
      }
    }, { status: statusCode });
  }
};

export const PUT: RequestHandler = async ({ request, url }) => {
  const requestId = crypto.randomUUID();
  const startTime = performance.now();
  let requestBody: any;

  try {
    requestBody = await request.json();
    const pathParts = url.pathname.split('/');
    const operation = pathParts[pathParts.length - 2]; // e.g., 'cases' from '/api/jsonb/legal/cases/timeline'
    const action = pathParts[pathParts.length - 1]; // e.g., 'timeline' or 'custody'

    await logger.logAPIRequest({
      requestId,
      method: 'PUT',
      endpoint: `/api/jsonb/legal/${operation}/${action}`,
      userAgent: request.headers.get('user-agent') || 'unknown',
      ipAddress: request.headers.get('x-forwarded-for') || 'unknown',
      requestBody,
      headers: Object.fromEntries(request.headers.entries())
    });

    if (operation === 'cases' && action === 'timeline') {
      // Add timeline event to case
      const { caseId, event } = z.object({
        caseId: z.string().uuid(),
        event: z.object({
          date: z.string().datetime(),
          event: z.string(),
          significance: z.enum(['low', 'medium', 'high', 'critical'])
        })
      }).parse(requestBody);
      
      const updatedCase = await jsonbLegalService.addCaseTimelineEvent(caseId, event);
      const duration = performance.now() - startTime;
      
      await logger.logAPIResponse({
        requestId,
        statusCode: 200,
        responseSize: JSON.stringify(updatedCase).length,
        processingTime: duration,
        success: true
      });

      return json({
        success: true,
        data: updatedCase,
        metadata: {
          requestId,
          processingTime: duration,
          timestamp: new Date().toISOString()
        }
      });
    } else if (operation === 'evidence' && action === 'custody') {
      // Add custody transfer
      const { evidenceId, transfer } = z.object({
        evidenceId: z.string().uuid(),
        transfer: z.object({
          timestamp: z.string().datetime(),
          custodian: z.string(),
          action: z.enum(['collected', 'transferred', 'analyzed', 'stored', 'retrieved']),
          location: z.string().optional(),
          condition: z.string().optional()
        })
      }).parse(requestBody);
      
      const updatedEvidence = await jsonbLegalService.addCustodyTransfer(evidenceId, transfer);
      const duration = performance.now() - startTime;
      
      await logger.logAPIResponse({
        requestId,
        statusCode: 200,
        responseSize: JSON.stringify(updatedEvidence).length,
        processingTime: duration,
        success: true
      });

      return json({
        success: true,
        data: updatedEvidence,
        metadata: {
          requestId,
          processingTime: duration,
          timestamp: new Date().toISOString()
        }
      });
    } else if (operation === 'evidence' && action === 'verify') {
      // Verify evidence chain
      const { evidenceId } = z.object({
        evidenceId: z.string().uuid()
      }).parse(requestBody);
      
      const verification = await jsonbLegalService.verifyEvidenceChain(evidenceId);
      const duration = performance.now() - startTime;
      
      await logger.logAPIResponse({
        requestId,
        statusCode: 200,
        responseSize: JSON.stringify(verification).length,
        processingTime: duration,
        success: true
      });

      return json({
        success: true,
        data: verification,
        metadata: {
          requestId,
          processingTime: duration,
          timestamp: new Date().toISOString()
        }
      });
    } else {
      throw error(404, `Operation not found: ${operation}/${action}`);
    }
  } catch (err: any) {
    const duration = performance.now() - startTime;
    let errorMessage = 'Unknown error';
    let statusCode = 500;

    if (err instanceof z.ZodError) {
      errorMessage = `Validation error: ${err.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', ')}`;
      statusCode = 400;
    } else if (err instanceof Error) {
      errorMessage = err.message;
      if (err.message.includes('not found')) {
        statusCode = 404;
      }
    }
    
    await logger.logAPIResponse({
      requestId,
      statusCode,
      responseSize: 0,
      processingTime: duration,
      success: false,
      error: errorMessage
    });

    await logger.logError({
      error: errorMessage,
      context: 'jsonb_legal_api_put',
      requestId,
      requestBody,
      severity: statusCode >= 500 ? 'high' : 'medium',
      category: 'api'
    });

    return json({
      success: false,
      error: errorMessage,
      metadata: {
        requestId,
        processingTime: duration,
        timestamp: new Date().toISOString()
      }
    }, { status: statusCode });
  }
};

// Maintenance endpoint
export const PATCH: RequestHandler = async ({ request }) => {
  const requestId = crypto.randomUUID();
  const startTime = performance.now();

  try {
    await logger.logAPIRequest({
      requestId,
      method: 'PATCH',
      endpoint: '/api/jsonb/legal',
      userAgent: request.headers.get('user-agent') || 'unknown',
      ipAddress: request.headers.get('x-forwarded-for') || 'unknown',
      headers: Object.fromEntries(request.headers.entries())
    });

    // Update case counters
    await jsonbLegalService.updateCaseCounters();
    const duration = performance.now() - startTime;
    
    await logger.logAPIResponse({
      requestId,
      statusCode: 200,
      responseSize: 0,
      processingTime: duration,
      success: true
    });

    return json({
      success: true,
      message: 'Case counters updated successfully',
      metadata: {
        requestId,
        processingTime: duration,
        timestamp: new Date().toISOString()
      }
    });
  } catch (err: any) {
    const duration = performance.now() - startTime;
    const errorMessage = err instanceof Error ? err.message : 'Unknown error';
    
    await logger.logAPIResponse({
      requestId,
      statusCode: 500,
      responseSize: 0,
      processingTime: duration,
      success: false,
      error: errorMessage
    });

    await logger.logError({
      error: errorMessage,
      context: 'jsonb_legal_api_patch',
      requestId,
      severity: 'high',
      category: 'api'
    });

    return json({
      success: false,
      error: errorMessage,
      metadata: {
        requestId,
        processingTime: duration,
        timestamp: new Date().toISOString()
      }
    }, { status: 500 });
  }
};