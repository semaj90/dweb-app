// Simple client wrapper for the security orchestrator service
// Assumes service reachable at SECURITY_ORCH_URL (default http://localhost:8600)

// Use the SvelteKit API route we created for security validation
const BASE_URL = (typeof process !== 'undefined' && process.env?.SECURITY_ORCH_URL) || '';

export interface SecurityValidationRequestClient {
  task: 'security_validation';
  fingerprint: Record<string, any>;
  user: { email: string; username: string; requestedRole?: string; referralCode?: string } & Record<string, any>;
  context?: Record<string, any>;
}

export interface SecurityValidationResponseClient {
  requestId: string;
  riskScore: number;
  securityScore: number;
  verification: Record<string, any>;
  signals: Array<{ name: string; weight: number; value: any; contribution: number }>;
  status: 'allow' | 'review' | 'deny';
  modelVersion: string;
  durationMs: number;
  timestamp: string;
}

export async function validateSecurity(payload: SecurityValidationRequestClient): Promise<SecurityValidationResponseClient> {
  // Use our SvelteKit API route for security validation
  const res = await fetch(`${BASE_URL}/api/security/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: payload.user.email,
      firstName: payload.user.firstName || payload.user.username.split('.')[0],
      lastName: payload.user.lastName || payload.user.username.split('.')[1],
      role: payload.user.requestedRole,
      department: payload.user.department,
      jurisdiction: payload.user.jurisdiction,
      badgeNumber: payload.user.badgeNumber,
      deviceInfo: payload.fingerprint
    })
  });
  
  if (!res.ok) {
    let detail: any;
    try { detail = await res.json(); } catch {}
    throw new Error(`Security validation failed (${res.status}): ${detail?.error || res.statusText}`);
  }
  
  const apiResponse = await res.json();
  
  // Transform our API response to match the expected SecurityValidationResponseClient interface
  return {
    requestId: apiResponse.validationId || 'unknown',
    riskScore: apiResponse.riskScore || 0,
    securityScore: apiResponse.securityScore || 0,
    verification: apiResponse.professionalVerification || {},
    signals: apiResponse.threatIntelligence?.signals || [],
    status: apiResponse.riskLevel === 'critical' ? 'deny' : 
            apiResponse.riskLevel === 'high' ? 'review' : 'allow',
    modelVersion: 'enhanced-rag-v1',
    durationMs: apiResponse.processingTime || 0,
    timestamp: new Date().toISOString()
  };
}

export function connectProgress(onMessage: (msg: any) => void): WebSocket {
  // Use our SvelteKit WebSocket progress endpoint
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/api/security/validate/progress`;
  
  const ws = new WebSocket(wsUrl);
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)); } catch { /* ignore */ }
  };
  return ws;
}
