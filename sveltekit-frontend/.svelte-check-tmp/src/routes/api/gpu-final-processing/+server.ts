import { completeErrorPipeline } from '$lib/services/complete-gpu-error-pipeline';
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async () => {
  try {
    console.log('🚀 Starting final GPU error processing with gemma3-legal GGUF...');

    const result = await completeErrorPipeline.runCompleteErrorProcessing();

    const statusReport = await completeErrorPipeline.generateStatusReport();

    return json({
      success: true,
      pipeline: result,
      statusReport,
      timestamp: new Date().toISOString(),
      message: 'Complete GPU error processing pipeline executed successfully'
    });
  } catch (error: any) {
    console.error('❌ GPU error processing failed:', error);

    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      pipeline: completeErrorPipeline.getPipelineStatus(),
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action } = await request.json();

    switch (action) {
      case 'status':
        const status = completeErrorPipeline.getPipelineStatus();
        return json({ success: true, status });

      case 'report':
        const report = await completeErrorPipeline.generateStatusReport();
        return json({ success: true, report });

      case 'run':
        const result = await completeErrorPipeline.runCompleteErrorProcessing();
        return json({ success: true, result });

      default:
        return json({ success: false, error: 'Invalid action' }, { status: 400 });
    }
  } catch (error: any) {
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};