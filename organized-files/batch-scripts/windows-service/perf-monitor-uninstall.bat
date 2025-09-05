@echo off
SET SVC_NAME=LegalAIPerfMonitor
sc stop %SVC_NAME%
sc delete %SVC_NAME%
echo Service %SVC_NAME% removed.
