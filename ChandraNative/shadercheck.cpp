// New ChandraNative code, MPL-2.0. Offline DXBC compilation only: no D3D device.
#include <windows.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <iostream>
#include <string>
int wmain(int argc,wchar_t** argv) {
    if(argc!=2){std::cerr<<"usage: shadercheck.exe shader-directory\n";return 2;}
    for(const wchar_t* name:{L"bf16.hlsl",L"rmsnorm.hlsl",L"delta.hlsl"}) {
        std::wstring path=std::wstring(argv[1])+L"/"+name;
        Microsoft::WRL::ComPtr<ID3DBlob> code,errors;
        HRESULT hr=D3DCompileFromFile(path.c_str(),nullptr,D3D_COMPILE_STANDARD_FILE_INCLUDE,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS,0,&code,&errors);
        if(errors)std::cerr.write(static_cast<const char*>(errors->GetBufferPointer()),errors->GetBufferSize());
        if(FAILED(hr)){std::cerr<<"shader compilation failed\n";return 1;}
        std::wcout<<name<<L" bytecode_bytes="<<code->GetBufferSize()<<L"\n";
    }
}
