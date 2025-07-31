/**
 * Agents API Routes
 * Multi-agent workflow orchestration endpoints
 */

import express from 'express';
import { z } from 'zod';

const router = express.Router();

// Validation schemas
const workflowSchema = z.object({
  workflowType: z.enum([
    'document_analysis',
    'legal_research',
    'case_preparation',
    'contract_review',
    'evidence_analysis'
  ]),
  input: z.any(),
  options: z.object({
    skipCache: z.boolean().optional().default(false),
    priority: z.enum(['low', 'medium', 'high', 'critical']).optional().default('medium'),
    timeout: z.number().int().min(1000).max(300000).optional().default(60000),
    agents: z.array(z.string()).optional(),
    context: z.record(z.any()).optional().default({})
  }).optional().default({})
});

const analysisSchema = z.object({
  text: z.string().min(1, 'Text is required').max(50000, 'Text too long'),
  analysisType: z.enum(['general', 'contract', 'evidence', 'compliance', 'litigation']).optional().default('general'),
  options: z.object({
    model: z.string().optional(),
    temperature: z.number().min(0).max(1).optional(),
    maxTokens: z.number().int().min(100).max(4096).optional()
  }).optional().default({})
});

const chatSchema = z.object({
  messages: z.array(z.object({
    role: z.enum(['user', 'assistant', 'system']),
    content: z.string().min(1)
  })).min(1, 'At least one message is required'),
  options: z.object({
    model: z.string().optional(),
    temperature: z.number().min(0).max(1).optional(),
    maxTokens: z.number().int().min(100).max(4096).optional(),
    stream: z.boolean().optional().default(false)
  }).optional().default({})
});

export function createAgentRoutes(services, io) {
  const { agentOrchestrator, ollama, cache } = services;

  /**
   * POST /workflow - Execute multi-agent workflow
   */
  router.post('/workflow', async (req, res) => {
    try {
      const validatedData = workflowSchema.parse(req.body);
      const { workflowType, input, options } = validatedData;

      console.log(`🎭 Starting workflow: ${workflowType}`);

      // Execute workflow
      const result = await agentOrchestrator.orchestrateWorkflow(workflowType, input, options);

      // Emit real-time update if this is for a specific case
      if (io && input.caseId) {
        io.to(`case-${input.caseId}`).emit('workflow-completed', {
          workflowType,
          documentId: input.id || input.documentId,
          processingTime: result.metadata.processingTime,
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        result
      });

    } catch (error) {
      console.error('Workflow execution failed:', error);
      
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          success: false,
          error: 'Validation error',
          details: error.errors
        });
      }

      res.status(500).json({
        success: false,
        error: 'Workflow execution failed',
        message: error.message
      });
    }
  });

  /**
   * POST /analyze - Single agent document analysis
   */
  router.post('/analyze', async (req, res) => {
    try {
      const validatedData = analysisSchema.parse(req.body);
      const { text, analysisType, options } = validatedData;

      // Check cache
      const cacheKey = `agent_analysis:${analysisType}:${Buffer.from(text.substring(0, 1000)).toString('base64')}`;
      const cached = await cache.get(cacheKey);
      
      if (cached) {
        return res.json({
          success: true,
          analysis: cached,
          cached: true
        });
      }

      // Perform analysis
      const result = await ollama.analyzeLegalDocument(text, analysisType, options);

      // Cache result
      await cache.set(cacheKey, result, 3600); // 1 hour

      res.json({
        success: true,
        analysis: result,
        cached: false
      });

    } catch (error) {
      console.error('Agent analysis failed:', error);
      
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          success: false,
          error: 'Validation error',
          details: error.errors
        });
      }

      res.status(500).json({
        success: false,
        error: 'Agent analysis failed',
        message: error.message
      });
    }
  });

  /**
   * POST /chat - Chat with AI agent
   */
  router.post('/chat', async (req, res) => {
    try {
      const validatedData = chatSchema.parse(req.body);
      const { messages, options } = validatedData;

      if (options.stream) {
        // Streaming response
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Cache-Control'
        });

        try {
          const stream = ollama.streamCompletion(
            messages[messages.length - 1].content,
            {
              model: options.model,
              temperature: options.temperature,
              maxTokens: options.maxTokens,
              context: messages.slice(0, -1).map(m => m.content)
            }
          );

          for await (const chunk of stream) {
            const data = JSON.stringify({
              content: chunk.content,
              done: chunk.done,
              model: chunk.model
            });
            
            res.write(`data: ${data}\n\n`);
            
            if (chunk.done) {
              break;
            }
          }

          res.write('data: [DONE]\n\n');
          res.end();

        } catch (streamError) {
          console.error('Streaming chat failed:', streamError);
          res.write(`data: ${JSON.stringify({ error: streamError.message })}\n\n`);
          res.end();
        }

      } else {
        // Regular response
        const result = await ollama.chatCompletion(messages, options);

        res.json({
          success: true,
          response: {
            content: result.content,
            model: result.model,
            created_at: result.created_at,
            usage: {
              prompt_tokens: result.prompt_eval_count,
              completion_tokens: result.eval_count,
              total_tokens: (result.prompt_eval_count || 0) + (result.eval_count || 0)
            }
          }
        });
      }

    } catch (error) {
      console.error('Chat failed:', error);
      
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          success: false,
          error: 'Validation error',
          details: error.errors
        });
      }

      if (!res.headersSent) {
        res.status(500).json({
          success: false,
          error: 'Chat failed',
          message: error.message
        });
      }
    }
  });

  /**
   * GET /workflows - List available workflows
   */
  router.get('/workflows', async (req, res) => {
    try {
      const stats = await agentOrchestrator.getWorkflowStats();

      res.json({
        success: true,
        workflows: stats.availableWorkflows.map(workflow => ({
          id: workflow,
          name: workflow.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
          description: getWorkflowDescription(workflow),
          estimatedTime: getEstimatedTime(workflow),
          agents: getWorkflowAgents(workflow)
        })),
        agents: stats.agentDetails,
        systemStatus: {
          ollamaHealthy: stats.ollamaHealthy,
          cacheEnabled: stats.cacheEnabled
        }
      });

    } catch (error) {
      console.error('Workflows listing failed:', error);
      res.status(500).json({
        success: false,
        error: 'Workflows listing failed',
        message: error.message
      });
    }
  });

  /**
   * GET /agents - List available agents
   */
  router.get('/agents', async (req, res) => {
    try {
      const stats = await agentOrchestrator.getWorkflowStats();

      res.json({
        success: true,
        agents: Object.entries(stats.agentDetails).map(([id, agent]) => ({
          id,
          name: agent.name,
          description: getAgentDescription(id),
          model: agent.model,
          temperature: agent.temperature,
          maxTokens: agent.maxTokens,
          capabilities: getAgentCapabilities(id)
        })),
        systemStatus: {
          ollamaHealthy: stats.ollamaHealthy,
          totalAgents: Object.keys(stats.agentDetails).length
        }
      });

    } catch (error) {
      console.error('Agents listing failed:', error);
      res.status(500).json({
        success: false,
        error: 'Agents listing failed',
        message: error.message
      });
    }
  });

  /**
   * POST /workflow/document-analysis - Specialized document analysis endpoint
   */
  router.post('/workflow/document-analysis', async (req, res) => {
    try {
      const { documentId, documentContent, title, documentType, caseId, options = {} } = req.body;

      if (!documentContent && !documentId) {
        return res.status(400).json({
          success: false,
          error: 'Either documentId or documentContent is required'
        });
      }

      const input = {
        id: documentId,
        content: documentContent,
        title: title || 'Untitled Document',
        documentType: documentType || 'general',
        caseId
      };

      const result = await agentOrchestrator.orchestrateWorkflow('document_analysis', input, options);

      // Emit real-time update
      if (io && caseId) {
        io.to(`case-${caseId}`).emit('document-analysis-completed', {
          documentId,
          processingTime: result.metadata.processingTime,
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        analysis: result.synthesis,
        phases: result.phases,
        metadata: result.metadata
      });

    } catch (error) {
      console.error('Document analysis failed:', error);
      res.status(500).json({
        success: false,
        error: 'Document analysis failed',
        message: error.message
      });
    }
  });

  /**
   * POST /workflow/legal-research - Specialized legal research endpoint
   */
  router.post('/workflow/legal-research', async (req, res) => {
    try {
      const { query, context, jurisdiction, caseId, options = {} } = req.body;

      if (!query) {
        return res.status(400).json({
          success: false,
          error: 'Query is required'
        });
      }

      const input = {
        text: query,
        context: context || 'General legal research',
        jurisdiction: jurisdiction || 'federal',
        caseId
      };

      const result = await agentOrchestrator.orchestrateWorkflow('legal_research', input, options);

      // Emit real-time update
      if (io && caseId) {
        io.to(`case-${caseId}`).emit('legal-research-completed', {
          query,
          processingTime: result.metadata.processingTime,
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        research: result.synthesis,
        phases: result.phases,
        metadata: result.metadata
      });

    } catch (error) {
      console.error('Legal research failed:', error);
      res.status(500).json({
        success: false,
        error: 'Legal research failed',
        message: error.message
      });
    }
  });

  /**
   * GET /workflow/:workflowId/status - Get workflow status (placeholder for async workflows)
   */
  router.get('/workflow/:workflowId/status', async (req, res) => {
    try {
      const { workflowId } = req.params;

      // This is a placeholder for future async workflow implementation
      // For now, all workflows are synchronous
      
      res.json({
        success: true,
        status: 'completed',
        message: 'All workflows are currently executed synchronously'
      });

    } catch (error) {
      console.error('Workflow status check failed:', error);
      res.status(500).json({
        success: false,
        error: 'Workflow status check failed',
        message: error.message
      });
    }
  });

  return router;
}

// Helper functions for workflow and agent descriptions
function getWorkflowDescription(workflow) {
  const descriptions = {
    document_analysis: 'Comprehensive analysis of legal documents with AI-powered insights',
    legal_research: 'Research legal precedents, case law, and statutory requirements',
    case_preparation: 'Strategic case preparation with evidence analysis and timeline development',
    contract_review: 'Detailed contract analysis with risk assessment and recommendations',
    evidence_analysis: 'Forensic analysis of evidence with authenticity and relevance assessment'
  };
  return descriptions[workflow] || 'Multi-agent workflow execution';
}

function getEstimatedTime(workflow) {
  const times = {
    document_analysis: '2-5 minutes',
    legal_research: '3-7 minutes',
    case_preparation: '5-10 minutes',
    contract_review: '3-6 minutes',
    evidence_analysis: '2-4 minutes'
  };
  return times[workflow] || '2-5 minutes';
}

function getWorkflowAgents(workflow) {
  const agents = {
    document_analysis: ['analyzer', 'summarizer', 'reviewer'],
    legal_research: ['researcher', 'strategist'],
    case_preparation: ['analyzer', 'strategist'],
    contract_review: ['analyzer', 'reviewer', 'strategist'],
    evidence_analysis: ['analyzer', 'reviewer', 'strategist']
  };
  return agents[workflow] || ['analyzer'];
}

function getAgentDescription(agentId) {
  const descriptions = {
    analyzer: 'Analyzes legal documents and identifies key clauses, risks, and important information',
    summarizer: 'Creates concise, accurate summaries that capture the most important information',
    researcher: 'Finds relevant case law, statutes, and legal precedents',
    reviewer: 'Evaluates accuracy, completeness, and relevance of legal content',
    strategist: 'Provides strategic recommendations and identifies potential legal approaches'
  };
  return descriptions[agentId] || 'AI agent for legal assistance';
}

function getAgentCapabilities(agentId) {
  const capabilities = {
    analyzer: ['Document Analysis', 'Risk Assessment', 'Key Clause Identification'],
    summarizer: ['Content Summarization', 'Key Point Extraction', 'Length Optimization'],
    researcher: ['Case Law Research', 'Statute Analysis', 'Precedent Finding'],
    reviewer: ['Quality Assurance', 'Accuracy Verification', 'Completeness Check'],
    strategist: ['Strategic Planning', 'Recommendation Generation', 'Risk Mitigation']
  };
  return capabilities[agentId] || ['General AI Assistance'];
}