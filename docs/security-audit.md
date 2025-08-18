# Security Audit Report - Legal AI Application
## SvelteKit 2 + Ollama + LangChain.js + PostgreSQL + pgvector Stack

### Executive Summary
This security audit covers the complete Legal AI application stack focusing on:
- **Frontend**: SvelteKit 2 with TypeScript
- **Backend**: SvelteKit API routes
- **AI/LLM**: Ollama with llama.cpp
- **AI Framework**: LangChain.js
- **Database**: PostgreSQL with pgvector extension
- **Authentication**: Lucia Auth with Drizzle ORM

### 🔒 Critical Security Findings

#### HIGH RISK

**1. API Security**
- **Issue**: No API key authentication implemented in production
- **Impact**: Unrestricted access to AI endpoints could lead to abuse
- **Recommendation**: 
  ```typescript
  // Implement API key middleware
  export async function handle({ event, resolve }) {
    if (event.url.pathname.startsWith('/api/')) {
      const apiKey = event.request.headers.get('X-API-Key');
      if (!validateApiKey(apiKey)) {
        return new Response('Unauthorized', { status: 401 });
      }
    }
    return resolve(event);
  }
  ```

**2. Database Connection Security**
- **Issue**: Database credentials in plain text environment variables
- **Impact**: Credentials exposure in logs/memory dumps
- **Recommendation**: Use encrypted credential store or vault
  ```env
  # Use encrypted database URLs
  DATABASE_URL=postgresql://$(vault read -field=username db/legal-ai):$(vault read -field=password db/legal-ai)@localhost:5432/legal_ai
  ```

**3. Ollama API Exposure**
- **Issue**: Ollama API accessible without authentication
- **Impact**: Direct model access could lead to resource abuse
- **Recommendation**: Implement reverse proxy with authentication
  ```nginx
  location /ollama/ {
    auth_request /auth;
    proxy_pass http://localhost:11434/;
  }
  ```

#### MEDIUM RISK

**4. Input Validation**
- **Issue**: Insufficient input sanitization for LLM prompts
- **Impact**: Prompt injection attacks
- **Recommendation**: Implement input sanitization
  ```typescript
  function sanitizePrompt(input: string): string {
    return input
      .replace(/[<>\"']/g, '') // Remove potential HTML/script chars
      .slice(0, 10000) // Limit input length
      .replace(/\b(ignore|forget|disregard)\s+(previous|above|all)\b/gi, '[REDACTED]');
  }
  ```

**5. Vector Database Security**
- **Issue**: pgvector embeddings not encrypted at rest
- **Impact**: Sensitive document content exposure
- **Recommendation**: Enable PostgreSQL encryption
  ```sql
  -- Enable TDE (Transparent Data Encryption)
  ALTER SYSTEM SET ssl = on;
  ALTER SYSTEM SET ssl_cert_file = 'server.crt';
  ALTER SYSTEM SET ssl_key_file = 'server.key';
  ```

#### LOW RISK

**6. Session Management**
- **Issue**: Default session configuration
- **Impact**: Session hijacking in certain scenarios
- **Recommendation**: Harden session configuration
  ```typescript
  // Lucia configuration
  export const lucia = new Lucia(adapter, {
    sessionCookie: {
      attributes: {
        secure: true,
        httpOnly: true,
        sameSite: 'strict',
        domain: 'yourdomain.com'
      }
    }
  });
  ```

### 🔐 Authentication & Authorization

#### Current Implementation: Lucia Auth
**Strengths:**
- Type-safe session management
- CSRF protection built-in
- Secure cookie handling

**Vulnerabilities:**
- No rate limiting on login attempts
- No multi-factor authentication
- No role-based access control

**Recommendations:**
```typescript
// Rate limiting middleware
import { rateLimit } from 'express-rate-limit';

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // Limit each IP to 5 requests per windowMs
  message: 'Too many login attempts, please try again later'
});

// RBAC implementation
export interface UserRole {
  id: string;
  name: 'admin' | 'lawyer' | 'paralegal' | 'viewer';
  permissions: Permission[];
}

export function requirePermission(permission: string) {
  return async (event: RequestEvent) => {
    const user = await getUser(event);
    if (!user?.roles.some(role => 
      role.permissions.includes(permission)
    )) {
      throw error(403, 'Insufficient permissions');
    }
  };
}
```

### 🛡️ Data Protection & Privacy

#### Personal Data Handling
**GDPR/Privacy Compliance:**
- Legal documents may contain PII
- Client-attorney privilege considerations
- Data retention requirements

**Current Gaps:**
- No data classification system
- No automatic PII detection
- No data retention policies

**Recommendations:**
```typescript
// PII Detection for legal documents
export class PIIDetector {
  private patterns = {
    ssn: /\b\d{3}-\d{2}-\d{4}\b/g,
    phone: /\b\d{3}-\d{3}-\d{4}\b/g,
    email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
    address: /\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b/gi
  };

  detectAndRedact(text: string): { text: string; detectedPII: string[] } {
    const detected: string[] = [];
    let redactedText = text;

    Object.entries(this.patterns).forEach(([type, pattern]) => {
      const matches = text.match(pattern);
      if (matches) {
        detected.push(...matches.map(match => `${type}: ${match}`));
        redactedText = redactedText.replace(pattern, `[${type.toUpperCase()}_REDACTED]`);
      }
    });

    return { text: redactedText, detectedPII: detected };
  }
}
```

#### Data Encryption
**At Rest:**
- Database: Enable PostgreSQL TDE
- File uploads: Encrypt using AES-256
- Vector embeddings: Consider encryption before storage

**In Transit:**
- HTTPS/TLS 1.3 for all communications
- mTLS for Ollama API communication
- Database connections over SSL

```typescript
// File encryption for uploads
import { createCipheriv, createDecipheriv, randomBytes } from 'crypto';

export class FileEncryption {
  private algorithm = 'aes-256-gcm';
  private key = Buffer.from(process.env.ENCRYPTION_KEY!, 'hex');

  encrypt(buffer: Buffer): { encrypted: Buffer; iv: Buffer; tag: Buffer } {
    const iv = randomBytes(16);
    const cipher = createCipheriv(this.algorithm, this.key, iv);
    
    const encrypted = Buffer.concat([
      cipher.update(buffer),
      cipher.final()
    ]);
    
    const tag = cipher.getAuthTag();
    return { encrypted, iv, tag };
  }

  decrypt(encrypted: Buffer, iv: Buffer, tag: Buffer): Buffer {
    const decipher = createDecipheriv(this.algorithm, this.key, iv);
    decipher.setAuthTag(tag);
    
    return Buffer.concat([
      decipher.update(encrypted),
      decipher.final()
    ]);
  }
}
```

### 🚨 Ollama & LLM Security

#### Model Security
**Current Risks:**
- Models loaded in memory are accessible
- No model integrity verification
- Potential model poisoning

**Recommendations:**
```bash
#!/bin/bash
# Model integrity verification
verify_model() {
  local model_name=$1
  local expected_hash=$2
  
  actual_hash=$(ollama show $model_name --modelfile | sha256sum | cut -d' ' -f1)
  
  if [ "$actual_hash" != "$expected_hash" ]; then
    echo "WARNING: Model hash mismatch for $model_name"
    exit 1
  fi
}

# Verify critical models
verify_model "gemma3-legal" "a1b2c3d4e5f6..."
verify_model "nomic-embed-text" "f6e5d4c3b2a1..."
```

#### Prompt Injection Prevention
```typescript
export class PromptSecurityFilter {
  private dangerousPatterns = [
    /ignore\s+(previous|all|above)\s+(instructions|prompts)/gi,
    /forget\s+(everything|all)\s+(before|above)/gi,
    /you\s+are\s+now\s+a\s+different/gi,
    /disregard\s+(safety|security)\s+(guidelines|measures)/gi,
    /execute\s+(code|script|command)/gi
  ];

  private systemPatterns = [
    /\[SYSTEM\]/gi,
    /\[INST\]/gi,
    /\[\/INST\]/gi,
    /<\|system\|>/gi,
    /<\|assistant\|>/gi
  ];

  validatePrompt(prompt: string): { isValid: boolean; issues: string[] } {
    const issues: string[] = [];

    // Check for prompt injection attempts
    this.dangerousPatterns.forEach((pattern, index) => {
      if (pattern.test(prompt)) {
        issues.push(`Potential prompt injection detected (pattern ${index + 1})`);
      }
    });

    // Check for system token abuse
    this.systemPatterns.forEach((pattern, index) => {
      if (pattern.test(prompt)) {
        issues.push(`System token detected (pattern ${index + 1})`);
      }
    });

    // Length validation
    if (prompt.length > 10000) {
      issues.push('Prompt exceeds maximum length');
    }

    return {
      isValid: issues.length === 0,
      issues
    };
  }

  sanitizePrompt(prompt: string): string {
    let sanitized = prompt;

    // Remove dangerous patterns
    this.dangerousPatterns.forEach(pattern => {
      sanitized = sanitized.replace(pattern, '[FILTERED]');
    });

    // Remove system tokens
    this.systemPatterns.forEach(pattern => {
      sanitized = sanitized.replace(pattern, '');
    });

    return sanitized.trim();
  }
}
```

### 🔍 Database Security (PostgreSQL + pgvector)

#### Connection Security
```typescript
// Secure database configuration
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';

const secureConnectionConfig = {
  host: process.env.DB_HOST,
  port: parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME,
  username: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  ssl: {
    rejectUnauthorized: true,
    ca: process.env.DB_SSL_CA,
    cert: process.env.DB_SSL_CERT,
    key: process.env.DB_SSL_KEY
  },
  // Connection security
  connect_timeout: 10,
  idle_timeout: 30,
  max: 20,
  // Query security
  prepare: false, // Prevent statement caching attacks
  types: {
    // Prevent type confusion attacks
    bigint: postgres.BigInt
  }
};
```

#### Vector Security
```sql
-- Row-level security for vector data
CREATE POLICY legal_documents_policy ON legal_documents
  FOR ALL TO authenticated_users
  USING (user_id = current_user_id() OR is_public = true);

-- Prevent vector injection attacks
CREATE OR REPLACE FUNCTION validate_embedding(embedding vector)
RETURNS boolean AS $$
BEGIN
  -- Check vector dimensions
  IF array_length(embedding::float[], 1) != 384 THEN
    RETURN false;
  END IF;
  
  -- Check for NaN or infinite values
  IF EXISTS (
    SELECT 1 FROM unnest(embedding::float[]) AS val 
    WHERE val = 'NaN'::float OR val = 'Infinity'::float OR val = '-Infinity'::float
  ) THEN
    RETURN false;
  END IF;
  
  RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Add constraint
ALTER TABLE legal_documents 
ADD CONSTRAINT valid_embedding_check 
CHECK (validate_embedding(content_embedding));
```

### 🌐 Network Security

#### API Security Headers
```typescript
// SvelteKit security headers
export async function handle({ event, resolve }) {
  const response = await resolve(event);

  // Security headers
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  
  // HSTS
  if (process.env.NODE_ENV === 'production') {
    response.headers.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
  }

  // CSP for AI/LLM security
  const cspPolicy = [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'", // Required for Svelte
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "connect-src 'self' http://localhost:11434", // Ollama API
    "worker-src 'self' blob:",
    "child-src 'none'",
    "object-src 'none'",
    "frame-ancestors 'none'"
  ].join('; ');
  
  response.headers.set('Content-Security-Policy', cspPolicy);

  return response;
}
```

#### Rate Limiting
```typescript
// API rate limiting for AI endpoints
import { rateLimit } from '@/lib/rate-limit';

const aiEndpointLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 10, // 10 requests per minute
  keyGenerator: (event) => {
    return event.getClientAddress() + ':' + event.url.pathname;
  },
  onLimitReached: (event) => {
    console.warn(`Rate limit exceeded for ${event.getClientAddress()}`);
  }
});

// In API routes
export async function POST({ request, getClientAddress }) {
  await aiEndpointLimiter.check(getClientAddress());
  // ... rest of handler
}
```

### 📋 Security Checklist

#### ✅ Implemented
- [x] TypeScript for type safety
- [x] HTTPS in production
- [x] Environment variable configuration
- [x] Error handling and logging
- [x] Input validation (basic)

#### ⚠️ Needs Implementation
- [ ] API key authentication
- [ ] Rate limiting
- [ ] Database encryption at rest
- [ ] PII detection and redaction
- [ ] Prompt injection prevention
- [ ] Model integrity verification
- [ ] Comprehensive audit logging
- [ ] Security headers
- [ ] RBAC implementation
- [ ] Data retention policies

#### 🔴 Critical Immediate Actions
1. **Implement API authentication** - Prevent unauthorized access
2. **Enable database SSL** - Protect data in transit
3. **Add rate limiting** - Prevent abuse of AI endpoints
4. **Implement prompt filtering** - Prevent injection attacks
5. **Setup monitoring** - Detect security incidents

### 🛠️ Implementation Timeline

**Week 1 (Critical):**
- API key authentication
- Database SSL configuration
- Basic rate limiting
- Security headers

**Week 2 (High Priority):**
- Prompt injection prevention
- PII detection system
- Comprehensive logging
- Model integrity checks

**Week 3 (Medium Priority):**
- RBAC implementation
- Data encryption at rest
- Audit trail system
- Security monitoring

**Week 4 (Enhancement):**
- Advanced threat detection
- Automated security testing
- Penetration testing
- Security documentation

### 📊 Security Metrics to Monitor

```typescript
// Security monitoring dashboard
export interface SecurityMetrics {
  authenticationFailures: number;
  rateLimitViolations: number;
  promptInjectionAttempts: number;
  databaseConnectionFailures: number;
  unauthorizedApiAccess: number;
  suspiciousFileUploads: number;
  modelIntegrityFailures: number;
}

// Security KPIs
const securityKPIs = {
  maxAuthFailuresPerHour: 100,
  maxRateLimitViolationsPerHour: 50,
  maxPromptInjectionAttemptsPerDay: 10,
  targetSecurityScore: 95, // percentage
  maxSecurityIncidentResponseTime: 300 // seconds
};
```

### 🔧 Security Tools Integration

**Recommended Tools:**
- **SAST**: ESLint with security plugins
- **DAST**: OWASP ZAP for API testing
- **Dependency Scanning**: npm audit, Snyk
- **Secrets Scanning**: GitLeaks, TruffleHog
- **Container Security**: Trivy, Clair (if containerized)
- **Runtime Security**: Falco, OSSEC

**Automated Security Pipeline:**
```yaml
# GitHub Actions security workflow
name: Security Scan
on: [push, pull_request]
jobs:
  security:
    runs-on: ubuntu-latest, need to make this work on windows native local.
    steps:
      - uses: actions/checkout@v4
      - name: Run npm audit
        run: npm audit --audit-level=moderate
      - name: Run SAST scan
        run: npx eslint . --ext .ts,.js --config .eslintrc.security.js
      - name: Check for secrets
        run: npx secretlint "**/*"
      - name: Run OWASP ZAP baseline scan
        run: docker run -t owasp/zap2docker-stable zap-baseline.py -t http://localhost:3000
```

---

**Report Generated:** 2024-12-XX  
**Classification:** Internal Use  
**Next Review Date:** 2024-XX-XX

earch session insert: stringify queryEmbedding.
OCR: force processedBuffer = Buffer.from(await sharp(buffer)...toBuffer()).
Analyze route:
Remove extra insert block if duplicated.
Cast insertedDoc.id to string where required.
Use Drizzle where with (tbl, {eq}) => eq(tbl.id, insertedDoc.id) only if builder returns correct signature; otherwise skip returning updated row (not used).
Replace direct dbDoc.tags with (dbDoc as any).tags ?? [] when reading.
Need your direction
Which approach should I take? Reply with: A = Patch root src only now B = Consolidate trees (describe desired canonical path) C = Stop here D = Something else (describe)

Let me know (A/B/C/D) and I’ll proceed.
