@echo off
setlocal
if "%~1"=="" (echo Pass an explicit build output directory. & exit /b 2)
set "CHANDRA_RUNTIME_BUILD=%~1"
if not exist "%CHANDRA_RUNTIME_BUILD%" mkdir "%CHANDRA_RUNTIME_BUILD%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
pushd "%CHANDRA_RUNTIME_BUILD%"
if errorlevel 1 exit /b 1
cl /nologo /c /std:c++17 /O2 /fp:strict /EHsc /W4 "%~dp0device.cpp" "%~dp0operators.cpp" "%~dp0model_weights.cpp" "%~dp0text_model.cpp" "%~dp0vision_model.cpp" "%~dp0diagnostics.cpp"
if errorlevel 1 goto failed
lib /nologo /out:chandra-runtime.lib device.obj operators.obj model_weights.obj text_model.obj vision_model.obj diagnostics.obj
if errorlevel 1 goto failed
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%~dp0operator_fixture.cpp" device.obj operators.obj /Fe:operator-fixture.exe /link d3d11.lib dxgi.lib d3dcompiler.lib bcrypt.lib gdi32.lib
if errorlevel 1 goto failed
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%~dp0dispatch_calibration.cpp" device.obj operators.obj /Fe:dispatch-calibration.exe /link d3d11.lib dxgi.lib d3dcompiler.lib bcrypt.lib gdi32.lib
if errorlevel 1 goto failed
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%~dp0inference.cpp" chandra-runtime.lib /Fe:chandra-inference.exe /link d3d11.lib dxgi.lib d3dcompiler.lib bcrypt.lib gdi32.lib
if errorlevel 1 goto failed
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%~dp0shader_compile.cpp" /Fe:shader-compile.exe /link d3dcompiler.lib
if errorlevel 1 goto failed
shader-compile.exe "%~dp0..\shaders\runtime"
if errorlevel 1 goto failed
popd
exit /b 0
:failed
popd
exit /b 1
