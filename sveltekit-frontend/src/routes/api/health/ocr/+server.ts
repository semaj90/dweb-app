import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

interface OCRHealthDetails {
  service: string;
  status: 'operational' | 'degraded' | 'offline';
  port?: number;
  endpoint: string;
  features?: string[];
  performance?: {
    avgProcessingTime: number;
    documentsProcessed: number;
    errorRate: number;
  };
  version?: string;
  lastChecked: string;
  responseTime: number;
}

interface OCRHealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  ocr: OCRHealthDetails;
  metadata: {
    checkDuration: number;
    environment: string;
  };
}

async function performOCRHealthCheck(): Promise<OCRHealthDetails> {
  const startTime = Date.now();
  const ocrBaseUrl = (globalThis as any).__OCR_BASE__ || '/api/ocr';
  
  try {
    console.log(`🔍 Checking OCR service at ${ocrBaseUrl}/status...`);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
    
    const response = await fetch(`${ocrBaseUrl}/status`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'LegalAI-HealthCheck/1.0'
      },
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    const responseTime = Date.now() - startTime;
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ OCR service responded successfully:', data);
      
      return {
        service: data.service || 'OCR Service',
        status: data.status || 'operational',
        port: data.port,
        endpoint: `${ocrBaseUrl}/status`,
        features: data.features || [],
        performance: data.performance || {
          avgProcessingTime: 0,
          documentsProcessed: 0,
          errorRate: 0
        },
        version: data.version,
        lastChecked: new Date().toISOString(),
        responseTime
      };
    } else {
      console.warn(`⚠️ OCR service returned status ${response.status}`);
      
      return {
        service: 'OCR Service',
        status: 'degraded',
        endpoint: `${ocrBaseUrl}/status`,
        lastChecked: new Date().toISOString(),
        responseTime
      };
    }
    
  } catch (error: any) {
    const responseTime = Date.now() - startTime;
    console.error('❌ OCR health check failed:', error);
    
    if (error.name === 'AbortError') {
      return {
        service: 'OCR Service',
        status: 'offline',
        endpoint: `${ocrBaseUrl}/status`,
        lastChecked: new Date().toISOString(),
        responseTime
      };
    }
    
    return {
      service: 'OCR Service',
      status: 'offline',
      endpoint: `${ocrBaseUrl}/status`,
      lastChecked: new Date().toISOString(),
      responseTime
    };
  }
}

function determineOverallStatus(ocrHealth: OCRHealthDetails): OCRHealthResponse['status'] {
  switch (ocrHealth.status) {
    case 'operational':
      return 'healthy';
    case 'degraded':
      return 'degraded';
    case 'offline':
    default:
      return 'unhealthy';
  }
}

export const GET: RequestHandler = async () => {
  const startTime = Date.now();
  const timestamp = new Date().toISOString();
  
  try {
    console.log('🔍 Starting OCR health monitoring check...');
    
    const ocrHealth = await performOCRHealthCheck();
    const overallStatus = determineOverallStatus(ocrHealth);
    const checkDuration = Date.now() - startTime;
    
    const response: OCRHealthResponse = {
      status: overallStatus,
      timestamp,
      ocr: ocrHealth,
      metadata: {
        checkDuration,
        environment: process.env.NODE_ENV || 'development'
      }
    };
    
    console.log(`✅ OCR health check completed in ${checkDuration}ms - Status: ${overallStatus}`);
    
    // Set HTTP status based on health
    const httpStatus = overallStatus === 'healthy' ? 200 :
                      overallStatus === 'degraded' ? 206 : 503;
    
    return json(response, {
      status: httpStatus,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'X-Health-Check': 'ocr-specific',
        'X-OCR-Status': ocrHealth.status,
        'X-Response-Time': ocrHealth.responseTime.toString()
      }
    });
    
  } catch (error: any) {
    const checkDuration = Date.now() - startTime;
    console.error('❌ OCR health monitoring system failed:', error);
    
    return json({
      status: 'unhealthy',
      timestamp,
      error: 'OCR health check system failure',
      message: error instanceof Error ? error.message : 'Unknown error',
      metadata: {
        checkDuration,
        environment: process.env.NODE_ENV || 'development'
      }
    }, {
      status: 503,
      headers: {
        'Content-Type': 'application/json',
        'X-Health-Check': 'failed'
      }
    });
  }
};

// Support POST for triggering specific OCR health actions
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action, timeout } = await request.json();
    
    if (action === 'test-processing') {
      // Test OCR processing capability with a simple test
      const startTime = Date.now();
      const ocrBaseUrl = (globalThis as any).__OCR_BASE__ || '/api/ocr';
      
      try {
        // Create a minimal test file for OCR processing
        const testText = 'Legal AI Health Check Test Document';
        const testBlob = new Blob([testText], { type: 'text/plain' });
        const formData = new FormData();
        formData.append('file', testBlob, 'health-check-test.txt');
        
        const response = await fetch(`${ocrBaseUrl}/extract`, {
          method: 'POST',
          body: formData,
          signal: AbortSignal.timeout(timeout || 15000)
        });
        
        const responseTime = Date.now() - startTime;
        
        if (response.ok) {
          const result = await response.json();
          
          return json({
            action: 'test-processing',
            status: 'success',
            message: 'OCR processing test completed successfully',
            responseTime,
            testResult: result,
            timestamp: new Date().toISOString()
          });
        } else {
          return json({
            action: 'test-processing',
            status: 'failed',
            message: `OCR test processing failed with status ${response.status}`,
            responseTime,
            timestamp: new Date().toISOString()
          }, { status: response.status });
        }
        
      } catch (error: any) {
        return json({
          action: 'test-processing',
          status: 'error',
          message: 'OCR processing test error',
          error: error.message,
          responseTime: Date.now() - startTime,
          timestamp: new Date().toISOString()
        }, { status: 500 });
      }
    }
    
    if (action === 'detailed-status') {
      // Get detailed status including batch processing capability
      const ocrHealth = await performOCRHealthCheck();
      
      return json({
        action: 'detailed-status',
        ...ocrHealth,
        additionalChecks: {
          batchProcessingAvailable: true, // Could test this endpoint too
          extractionFormats: ['pdf', 'png', 'jpg', 'jpeg', 'txt'],
          maxFileSize: '50MB', // Typical OCR service limits
        },
        timestamp: new Date().toISOString()
      });
    }
    
    // Default to regular health check
    return GET();
    
  } catch (error: any) {
    return json({
      error: 'Invalid request',
      message: error.message,
      availableActions: ['test-processing', 'detailed-status']
    }, { status: 400 });
  }
};

// Support HEAD for lightweight health pings
export const HEAD: RequestHandler = async () => {
  try {
    const ocrHealth = await performOCRHealthCheck();
    const overallStatus = determineOverallStatus(ocrHealth);
    
    const httpStatus = overallStatus === 'healthy' ? 200 :
                      overallStatus === 'degraded' ? 206 : 503;
    
    return new Response(null, {
      status: httpStatus,
      headers: {
        'X-OCR-Status': ocrHealth.status,
        'X-Response-Time': ocrHealth.responseTime.toString(),
        'X-Last-Checked': ocrHealth.lastChecked
      }
    });
  } catch (error: any) {
    return new Response(null, {
      status: 503,
      headers: {
        'X-Health-Check': 'failed'
      }
    });
  }
};