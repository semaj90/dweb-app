/**
 * SvelteKit API route for generating a comprehensive summary.
 * This file demonstrates the correct pattern for a POST handler, including
 * type-safe imports, request validation, and error handling.
 */
import { json } from '@sveltejs/kit';
import type { RequestHandler } from '@sveltejs/kit';
import type { APIResponse } from '$lib/types/index';

// Safely import the server-only summarizer logic
import {
  comprehensiveOllamaSummarizer,
  type ComprehensiveSummaryRequest
} from '$lib/services/comprehensive-ollama-summarizer';

/**
 * Handles POST requests to generate a comprehensive summary.
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    // 1. Get the request body.
    const body = await request.json() as ComprehensiveSummaryRequest;

    // 2. Validate the incoming payload.
    if (!body.content || typeof body.content !== 'string' || body.content.length < 10) {
      return json(
        { error: 'Invalid request. "content" must be a string with at least 10 characters.' },
        { status: 400 }
      );
    }

    // 3. Call the core server-side business logic.
    const started = Date.now();
    const summaryResponse = await comprehensiveOllamaSummarizer.summarize(body);

    // 4. Return the successful response (standard envelope).
    return json({
      success: true,
      data: summaryResponse,
      metadata: {
        timestamp: new Date().toISOString(),
        processingTimeMs: Date.now() - started
      }
    } satisfies APIResponse<typeof summaryResponse>, { status: 200 });

  } catch (error) {
    // 5. Provide robust error handling for unexpected issues.
    console.error('Critical error in comprehensive-summary endpoint:', error);

    let errorMessage = 'An internal server error occurred.';
    if (error instanceof Error) {
      // Avoid leaking sensitive stack traces in production
      errorMessage = 'Failed to process request. Please check server logs.';
    }

    return json({
      success: false,
      error: { code: 'INTERNAL_ERROR', message: errorMessage },
      metadata: {
        timestamp: new Date().toISOString(),
        processingTimeMs: 0
      }
    } satisfies APIResponse<never>, { status: 500 });
  }
};