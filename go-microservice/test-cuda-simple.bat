@echo off
echo Testing CUDA compilation...

set CGO_ENABLED=1
set CC=gcc
set CXX=g++

echo Environment:
echo CGO_ENABLED=%CGO_ENABLED%
echo CC=%CC%
echo CXX=%CXX%

echo.
echo Compiling test_cuda.go...
go run test_cuda.go

pause