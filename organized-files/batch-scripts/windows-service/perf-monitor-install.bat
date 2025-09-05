@echo off
SET SVC_NAME=LegalAIPerfMonitor
SET DISPLAY_NAME=LegalAI Performance Monitor
SET BIN_PATH="%~dp0..\go-microservice\bin\perf-monitor.exe"

IF NOT EXIST %BIN_PATH% (
  echo [BUILD] Compiling perf-monitor...
  pushd "%~dp0..\go-microservice\cmd\perf-monitor"
  go build -o ..\..\bin\perf-monitor.exe .
  popd
)

sc create %SVC_NAME% binPath= %BIN_PATH% start= auto DisplayName= "%DISPLAY_NAME%"
sc description %SVC_NAME% "Collects runtime metrics (goroutines, heap, GC, CPU proxy) for Legal AI stack"
sc failure %SVC_NAME% reset= 86400 actions= restart/5000/restart/10000/restart/60000

echo Service %SVC_NAME% installed.
sc start %SVC_NAME%
