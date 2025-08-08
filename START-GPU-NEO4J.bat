@echo off
echo Starting Neo4j + GPU System...

REM Neo4j Desktop path
set NEO4J_PATH=C:\Users\james\.Neo4jDesktop2\Data\dbmss\dbms-2f714300-bca3-42d3-9362-b8d8984b265a\bin
"%NEO4J_PATH%\neo4j.bat" console

REM Start with GPU
cd go-microservice
set CGO_ENABLED=1
set CGO_CFLAGS=-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include"
set CGO_LDFLAGS=-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64" -lcudart -lcublas
go build -o ai-gpu.exe .
start /B ai-gpu.exe

echo GPU+Neo4j Ready
pause
