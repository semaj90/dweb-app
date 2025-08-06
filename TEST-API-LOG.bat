@echo off
set LOG=test_%date:~10,4%%date:~4,2%%date:~7,2%.log
echo TEST %date% %time% > %LOG%

echo Testing API...
curl http://localhost:8080/health 2>>%LOG%
curl http://localhost:11434/api/tags 2>>%LOG%
curl -X POST http://localhost:8080/process-document -H "Content-Type: application/json" -d "{\"document_id\":\"test\",\"content\":\"test doc\"}" 2>>%LOG%

type %LOG%
echo Log: %LOG%
timeout /t 5
