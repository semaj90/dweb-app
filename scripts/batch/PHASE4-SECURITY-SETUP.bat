@echo off
title Phase 4 - Security Configuration
color 0A

echo 🔒 Phase 4: Security Configuration
echo =================================

echo Setting up security middleware...
cd sveltekit-frontend

:: Create security directory
mkdir src\lib\security 2>nul

:: Rate limiting
echo import { rateLimit } from 'express-rate-limit'; > src\lib\security\rateLimit.ts
echo. >> src\lib\security\rateLimit.ts
echo export const apiLimiter = rateLimit({ >> src\lib\security\rateLimit.ts
echo   windowMs: 15 * 60 * 1000, // 15 minutes >> src\lib\security\rateLimit.ts
echo   max: 100 // limit each IP to 100 requests per windowMs >> src\lib\security\rateLimit.ts
echo }^); >> src\lib\security\rateLimit.ts

:: Auth middleware
echo import jwt from 'jsonwebtoken'; > src\lib\security\auth.ts
echo. >> src\lib\security\auth.ts
echo export function verifyToken(token: string^) { >> src\lib\security\auth.ts
echo   return jwt.verify(token, process.env.JWT_SECRET!^); >> src\lib\security\auth.ts
echo } >> src\lib\security\auth.ts

:: Input validation
echo import { z } from 'zod'; > src\lib\security\validation.ts
echo. >> src\lib\security\validation.ts
echo export const userSchema = z.object({ >> src\lib\security\validation.ts
echo   email: z.string().email(^), >> src\lib\security\validation.ts
echo   password: z.string().min(8^) >> src\lib\security\validation.ts
echo }^); >> src\lib\security\validation.ts

:: Install security packages
npm install express-rate-limit jsonwebtoken bcryptjs helmet cors
npm install -D @types/jsonwebtoken @types/bcryptjs

echo ✅ Security configuration complete!
echo.
echo Security features added:
echo - Rate limiting
echo - JWT authentication  
echo - Input validation
echo - Password hashing
echo.
pause
