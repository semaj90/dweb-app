:: Rest of LEGAL-AI-CONTROL-PANEL.bat handlers

:HEALTH_CHECK
echo Running comprehensive health check...
echo Docker Services:
docker ps --format "table {{.Names}}\t{{.Status}}"
echo TypeScript Check:
cd sveltekit-frontend && npm run check
cd ..
goto MENU

:SETUP_AI
echo Setting up AI stack...
powershell -ExecutionPolicy Bypass -File setup-local-ai-stack.ps1
goto MENU

:LOAD_MODELS
echo Loading local models...
call LOAD-LOCAL-MODELS.bat
goto MENU

:FIX_TS
echo Fixing TypeScript issues...
call FIX-TYPESCRIPT-ISSUES.bat
goto MENU

:START_DEV
echo Starting development server...
cd sveltekit-frontend
npm run dev
goto MENU

:VIEW_TODO
if exist "%TODO_FILE%" (
    type "%TODO_FILE%"
) else (
    echo No TODO file generated yet
)
pause
goto MENU

:EXIT
exit
