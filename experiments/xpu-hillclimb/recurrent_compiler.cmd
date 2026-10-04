@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
rem Triton's pinned Windows build dispatch recognizes this uppercase suffix.
set "CC=cl.EXE"
set "CXX=cl.EXE"
%*
exit /b %errorlevel%
