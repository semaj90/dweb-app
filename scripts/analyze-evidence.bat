@echo off
REM Multi-Agent Evidence Analysis Pipeline for Windows
REM Usage: scripts\analyze-evidence.bat <case_id> <evidence_file_path>

setlocal enabledelayedexpansion

set CASE_ID=%1
set EVIDENCE_FILE=%2
set TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set TEMP_DIR=temp\analysis_%CASE_ID%_%TIMESTAMP%

REM Validate inputs
if "%CASE_ID%"=="" (
    echo Usage: %0 ^<case_id^> ^<evidence_file_path^>
    exit /b 1
)

if "%EVIDENCE_FILE%"=="" (
    echo Usage: %0 ^<case_id^> ^<evidence_file_path^>
    exit /b 1
)

if not exist "%EVIDENCE_FILE%" (
    echo Error: Evidence file not found: %EVIDENCE_FILE%
    exit /b 1
)

REM Create temp directory
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

echo Starting multi-agent evidence analysis...
echo Case ID: %CASE_ID%
echo Evidence: %EVIDENCE_FILE%
echo Working directory: %TEMP_DIR%

REM Step 1: Evidence Analyzer Agent
echo === Step 1: Analyzing Evidence ===
echo Processing evidence document with Evidence Analyzer Agent...

claude -f .claude\agents\evidence-analyzer.md --input-file "%EVIDENCE_FILE%" > "%TEMP_DIR%\evidence_analysis.json"

if errorlevel 1 (
    echo X Evidence analysis failed
    exit /b 1
) else (
    echo ✓ Evidence analysis completed
)

REM Step 2: Person Extractor Agent
echo === Step 2: Extracting Persons of Interest ===
echo Extracting persons and relationships...

claude -f .claude\agents\person-extractor.md --input-file "%TEMP_DIR%\evidence_analysis.json" > "%TEMP_DIR%\persons_extracted.json"

if errorlevel 1 (
    echo X Person extraction failed
    exit /b 1
) else (
    echo ✓ Person extraction completed
)

REM Step 3: Relationship Mapper Agent
echo === Step 3: Mapping Relationships to Neo4j ===
echo Updating Neo4j graph with persons and relationships...

REM Create mapper input file
echo { > "%TEMP_DIR%\mapper_input.json"
echo   "caseId": "%CASE_ID%", >> "%TEMP_DIR%\mapper_input.json"
echo   "evidenceFile": "%EVIDENCE_FILE%", >> "%TEMP_DIR%\mapper_input.json"
type "%TEMP_DIR%\persons_extracted.json" | findstr /v "^{" | findstr /v "^}" >> "%TEMP_DIR%\mapper_input.json"
echo } >> "%TEMP_DIR%\mapper_input.json"

claude -f .claude\agents\relationship-mapper.md --input-file "%TEMP_DIR%\mapper_input.json" > "%TEMP_DIR%\neo4j_updates.json"

if errorlevel 1 (
    echo X Neo4j relationship mapping failed
    exit /b 1
) else (
    echo ✓ Neo4j relationship mapping completed
)

REM Step 4: Case Synthesizer Agent
echo === Step 4: Synthesizing Case Analysis ===
echo Generating prosecutorial analysis and recommendations...

REM Create synthesis input (simplified for batch)
echo { > "%TEMP_DIR%\synthesis_input.json"
echo   "caseId": "%CASE_ID%", >> "%TEMP_DIR%\synthesis_input.json"
echo   "evidenceAnalysis": >> "%TEMP_DIR%\synthesis_input.json"
type "%TEMP_DIR%\evidence_analysis.json" >> "%TEMP_DIR%\synthesis_input.json"
echo   , >> "%TEMP_DIR%\synthesis_input.json"
echo   "personsData": >> "%TEMP_DIR%\synthesis_input.json"
type "%TEMP_DIR%\persons_extracted.json" >> "%TEMP_DIR%\synthesis_input.json"
echo } >> "%TEMP_DIR%\synthesis_input.json"

claude -f .claude\agents\case-synthesizer.md --input-file "%TEMP_DIR%\synthesis_input.json" > "%TEMP_DIR%\case_synthesis.json"

if errorlevel 1 (
    echo X Case synthesis failed
    exit /b 1
) else (
    echo ✓ Case synthesis completed
)

echo.
echo === Analysis Complete ===
echo ✓ Evidence analyzed and structured
echo ✓ Persons of interest identified
echo ✓ Neo4j updates prepared
echo ✓ Case synthesis and recommendations generated
echo.
echo Output files:
echo   - Evidence Analysis: %TEMP_DIR%\evidence_analysis.json
echo   - Persons Extracted: %TEMP_DIR%\persons_extracted.json
echo   - Neo4j Updates: %TEMP_DIR%\neo4j_updates.json
echo   - Case Synthesis: %TEMP_DIR%\case_synthesis.json
echo.
echo Multi-agent analysis pipeline completed successfully!

endlocal