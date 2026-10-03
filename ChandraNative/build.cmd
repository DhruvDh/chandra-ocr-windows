@echo off
setlocal
if "%~1"=="" (set "CHANDRA_BUILD=E:\build\chandra-native") else (set "CHANDRA_BUILD=%~1")
if not exist "%CHANDRA_BUILD%" mkdir "%CHANDRA_BUILD%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%~dp0main.cpp" /Fo"%CHANDRA_BUILD%\main.obj" /Fe"%CHANDRA_BUILD%\chandra-native.exe" /link d3d11.lib dxgi.lib d3dcompiler.lib bcrypt.lib gdi32.lib
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%~dp0oracle_test.cpp" /Fo"%CHANDRA_BUILD%\oracle_test.obj" /Fe"%CHANDRA_BUILD%\oracle-test.exe"
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%~dp0shadercheck.cpp" /Fo"%CHANDRA_BUILD%\shadercheck.obj" /Fe"%CHANDRA_BUILD%\shadercheck.exe" /link d3dcompiler.lib
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%~dp0fixture_check.cpp" /Fo"%CHANDRA_BUILD%\fixture_check.obj" /Fe"%CHANDRA_BUILD%\fixture-check.exe" /link bcrypt.lib
exit /b %errorlevel%
