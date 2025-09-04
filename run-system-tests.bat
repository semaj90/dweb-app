@echo off
REM Windows test runner - executes architecture verification and CRUD smoke test
echo Running system verification scripts...
node verify-system-architecture.cjs
if ERRORLEVEL 1 (
  echo Architecture checks failed. See output above.
) else (
  echo Architecture checks passed.
)

echo Running CRUD smoke tests (Node)...
node test-complete-crud-system.js
echo All tests finished.
pause
