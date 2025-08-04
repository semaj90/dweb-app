@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd pgvector
set PGROOT=C:\Program Files\PostgreSQL\17
nmake /F Makefile.win