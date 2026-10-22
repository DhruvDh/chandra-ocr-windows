// New code, MPL-2.0. Offline shader compilation; never creates a GPU device.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>
int wmain(int argc,wchar_t** argv) {
 if(argc!=2){std::cerr<<"usage: shader-compile.exe runtime-shader-directory\n";return 2;}
 try {
  std::filesystem::path root(argv[1]);
  if(!std::filesystem::is_directory(root))throw std::runtime_error("Shader directory absent");
  std::vector<std::filesystem::path> paths;
  for(const auto& item:std::filesystem::directory_iterator(root))
   if(item.is_regular_file()&&item.path().extension()==L".hlsl"&&item.path().filename()!=L"vision_common.hlsl")paths.push_back(item.path());
  std::sort(paths.begin(),paths.end());if(paths.empty())throw std::runtime_error("No runtime shaders");
  for(const auto& path:paths){
   Microsoft::WRL::ComPtr<ID3DBlob> code,errors;
   HRESULT hr=D3DCompileFromFile(path.c_str(),nullptr,D3D_COMPILE_STANDARD_FILE_INCLUDE,"main","cs_5_0",
    D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS,0,&code,&errors);
   if(errors)std::cerr.write(static_cast<const char*>(errors->GetBufferPointer()),errors->GetBufferSize());
   if(FAILED(hr))throw std::runtime_error("Runtime HLSL compilation failed");
   std::wcout<<path.filename().wstring()<<L" bytecode_bytes="<<code->GetBufferSize()<<std::endl;
  }
  std::cout<<"CPU-only runtime shader compilation passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
