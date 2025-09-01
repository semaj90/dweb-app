/**
 * Test Context7 MCP Helper Functions
 * Tests the utility functions from mcp-context72-get-library-docs
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { 
  getSvelte5Docs, 
  getBitsUIv2Docs, 
  getMeltUIDocs, 
  getXStateDocs 
} from '$lib/mcp-context72-get-library-docs';

export const GET: RequestHandler = async ({ fetch }) => {
  try {
    console.log('🧪 Testing Context7 MCP helper functions...');
    
    // Test all helper functions with event.fetch
    const [svelteRunes, bitsDialog, meltBuilder, xstateMachine] = await Promise.all([
      getSvelte5Docs('runes', fetch).catch(e => ({ error: e.message })),
      getBitsUIv2Docs('dialog', fetch).catch(e => ({ error: e.message })),
      getMeltUIDocs('button', fetch).catch(e => ({ error: e.message })),
      getXStateDocs('machine', fetch).catch(e => ({ error: e.message }))
    ]);
    
    return json({
      success: true,
      message: 'Context7 MCP integration test completed',
      results: {
        svelteRunes: {
          status: svelteRunes.error ? 'error' : 'success',
          tokenCount: svelteRunes.error ? 0 : svelteRunes.metadata?.tokenCount,
          snippets: svelteRunes.error ? 0 : svelteRunes.snippets?.length || 0,
          error: svelteRunes.error
        },
        bitsDialog: {
          status: bitsDialog.error ? 'error' : 'success',
          tokenCount: bitsDialog.error ? 0 : bitsDialog.metadata?.tokenCount,
          snippets: bitsDialog.error ? 0 : bitsDialog.snippets?.length || 0,
          error: bitsDialog.error
        },
        meltBuilder: {
          status: meltBuilder.error ? 'error' : 'success', 
          tokenCount: meltBuilder.error ? 0 : meltBuilder.metadata?.tokenCount,
          snippets: meltBuilder.error ? 0 : meltBuilder.snippets?.length || 0,
          error: meltBuilder.error
        },
        xstateMachine: {
          status: xstateMachine.error ? 'error' : 'success',
          tokenCount: xstateMachine.error ? 0 : xstateMachine.metadata?.tokenCount,
          snippets: xstateMachine.error ? 0 : xstateMachine.snippets?.length || 0,
          error: xstateMachine.error
        }
      },
      summary: {
        totalTests: 4,
        successful: [svelteRunes, bitsDialog, meltBuilder, xstateMachine].filter(r => !r.error).length,
        failed: [svelteRunes, bitsDialog, meltBuilder, xstateMachine].filter(r => r.error).length
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error: any) {
    console.error('❌ Context7 test error:', error);
    
    return json({
      success: false,
      error: error.message || 'Context7 test failed',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};