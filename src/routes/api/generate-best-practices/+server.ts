/**
 * Best Practices Generation API Endpoint
 * Generates actionable best practices for the current codebase
 */

import { json } from '@sveltejs/kit';
import { bestPracticesService } from '$lib/services/best-practices-service.js';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const body = await request.json().catch(() => ({}));
    
    // Get project path from request or use current directory
    const projectPath = body.projectPath || process.cwd();
    const options = body.options || {};

    console.log(`🔍 Generating best practices for: ${projectPath}`);

    // Generate comprehensive best practices report
    const report = await bestPracticesService.generateBestPractices(projectPath);

    return json({
      success: true,
      data: {
        report,
        summary: {
          total_practices: report.best_practices.length,
          high_priority: report.summary.high_priority_count,
          quick_wins: report.summary.quick_wins.length,
          estimated_effort: report.summary.estimated_total_effort,
          analysis: {
            project_type: report.codebase_analysis.project_type,
            technologies: report.codebase_analysis.technologies,
            file_count: report.codebase_analysis.file_count,
            lines_of_code: report.codebase_analysis.lines_of_code
          }
        }
      },
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Best practices generation failed:', error);
    
    return json({
      success: false,
      error: {
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        code: 'BEST_PRACTICES_GENERATION_FAILED',
        timestamp: new Date().toISOString()
      }
    }, { status: 500 });
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const limit = parseInt(url.searchParams.get('limit') || '10');
    
    // Get stored best practices reports
    const reports = await bestPracticesService.getStoredReports(limit);

    return json({
      success: true,
      data: {
        reports,
        count: reports.length
      },
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Failed to retrieve stored reports:', error);
    
    return json({
      success: false,
      error: {
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        code: 'REPORTS_RETRIEVAL_FAILED',
        timestamp: new Date().toISOString()
      }
    }, { status: 500 });
  }
};

export const OPTIONS: RequestHandler = async () => {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400'
    }
  });
};