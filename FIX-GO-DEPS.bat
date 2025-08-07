// FIX-GO-DEPS.bat
@echo off
cd go-microservice
go get github.com/gorilla/websocket
go mod tidy
cd ..
echo Fixed!