// New ChandraNative code, MPL-2.0. Adapts the typed-buffer/dispatch/readback
// architecture of Const-me/Whisper; intentionally independent of audio/model APIs.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include "oracle.h"
#include "fixture.h"
#include "adapter_identity.h"
#include <iostream>
#include <string>
#include <chrono>
#include <cstring>
#include <thread>
#include <limits>
#include <memory>
using Microsoft::WRL::ComPtr;
static void check(HRESULT hr,const char* operation) {if(FAILED(hr))throw std::runtime_error(std::string(operation)+" HRESULT="+std::to_string(uint32_t(hr)));}
struct Buffer {ComPtr<ID3D11Buffer> data;ComPtr<ID3D11ShaderResourceView> srv;ComPtr<ID3D11UnorderedAccessView> uav;unsigned bytes=0;};
struct Measurement {double gpuMs,wallMs;};
class Engine {
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11Buffer> constants;
public:
    std::string adapter;uint64_t dedicatedBytes=0;chandra::Json identity;
    explicit Engine(const std::wstring& requested,int requestedIndex,const std::string& pci,const std::string& luid) {
        auto candidates=chandra::intelAdapters(requested);
        for(const auto& candidate:candidates)std::cerr<<candidate.identity.dump()<<"\n";
        auto chosen=chandra::selectAdapter(candidates,requestedIndex,pci,luid);
        auto selected=chosen.handle;identity=chosen.identity;
        adapter=identity.at("description").get<std::string>();dedicatedBytes=identity.at("dedicated_bytes").get<uint64_t>();
        D3D_FEATURE_LEVEL levels[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0},actual;
        HRESULT hr=D3D11CreateDevice(selected.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,levels,2,D3D11_SDK_VERSION,&device,&actual,&context);
        if(hr==E_INVALIDARG)hr=D3D11CreateDevice(selected.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,levels+1,1,D3D11_SDK_VERSION,&device,&actual,&context);
        check(hr,"D3D11CreateDevice");D3D11_BUFFER_DESC cb{};cb.ByteWidth=32;cb.Usage=D3D11_USAGE_DEFAULT;cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;check(device->CreateBuffer(&cb,nullptr,&constants),"constant buffer");
    }
    Buffer buffer(unsigned count,const void* initial=nullptr,bool writable=false,bool integer=false) {
        if(!count || uint64_t(count)*4>INT_MAX)throw std::runtime_error("invalid buffer size");Buffer b;b.bytes=count*4;
        D3D11_BUFFER_DESC desc{};desc.ByteWidth=b.bytes;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|(writable?D3D11_BIND_UNORDERED_ACCESS:0);D3D11_SUBRESOURCE_DATA init{};init.pSysMem=initial;
        check(device->CreateBuffer(&desc,initial?&init:nullptr,&b.data),"CreateBuffer");
        D3D11_SHADER_RESOURCE_VIEW_DESC sd{};sd.Format=integer?DXGI_FORMAT_R32_UINT:DXGI_FORMAT_R32_FLOAT;sd.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;sd.Buffer.NumElements=count;check(device->CreateShaderResourceView(b.data.Get(),&sd,&b.srv),"CreateSRV");
        if(writable) {D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=sd.Format;ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=count;check(device->CreateUnorderedAccessView(b.data.Get(),&ud,&b.uav),"CreateUAV");}return b;
    }
    Buffer floats(const std::vector<float>& v) {return buffer(unsigned(v.size()),v.data());}
    ComPtr<ID3D11ComputeShader> shader(const std::wstring& path) {
        ComPtr<ID3DBlob> code,errors;HRESULT hr=D3DCompileFromFile(path.c_str(),nullptr,D3D_COMPILE_STANDARD_FILE_INCLUDE,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS,0,&code,&errors);
        if(errors)std::cerr.write(static_cast<const char*>(errors->GetBufferPointer()),errors->GetBufferSize());check(hr,"D3DCompileFromFile");ComPtr<ID3D11ComputeShader> s;check(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&s),"CreateComputeShader");return s;
    }
    template<class Params> void dispatch(ID3D11ComputeShader* shader,const std::vector<Buffer*>& inputs,const std::vector<Buffer*>& outputs,const Params& params,unsigned groups) {
        static_assert(sizeof(Params)<=32,"constant buffer overflow");uint32_t cb[8]{};std::memcpy(cb,&params,sizeof(params));context->UpdateSubresource(constants.Get(),0,nullptr,cb,0,0);
        ID3D11Buffer* c=constants.Get();context->CSSetConstantBuffers(0,1,&c);std::vector<ID3D11ShaderResourceView*> srvs;for(auto b:inputs)srvs.push_back(b->srv.Get());std::vector<ID3D11UnorderedAccessView*> uavs;for(auto b:outputs)uavs.push_back(b->uav.Get());
        context->CSSetShader(shader,nullptr,0);context->CSSetShaderResources(0,unsigned(srvs.size()),srvs.data());context->CSSetUnorderedAccessViews(0,unsigned(uavs.size()),uavs.data(),nullptr);context->Dispatch(groups,1,1);
        ID3D11ShaderResourceView* noSrv[8]{};ID3D11UnorderedAccessView* noUav[8]{};context->CSSetShaderResources(0,8,noSrv);context->CSSetUnorderedAccessViews(0,8,noUav,nullptr);
    }
    void zero(Buffer& b) {float values[4]{};context->ClearUnorderedAccessViewFloat(b.uav.Get(),values);}
    std::vector<float> read(const Buffer& b) {
        D3D11_BUFFER_DESC desc{};b.data->GetDesc(&desc);desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ComPtr<ID3D11Buffer> staging;check(device->CreateBuffer(&desc,nullptr,&staging),"CreateStaging");context->CopyResource(staging.Get(),b.data.Get());D3D11_MAPPED_SUBRESOURCE map{};check(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&map),"MapReadback");std::vector<float> result(b.bytes/4);std::memcpy(result.data(),map.pData,b.bytes);context->Unmap(staging.Get(),0);return result;
    }
    template<class F> Measurement time(F operation) {
        D3D11_QUERY_DESC desc{D3D11_QUERY_TIMESTAMP_DISJOINT,0};ComPtr<ID3D11Query> disjoint,start,end;check(device->CreateQuery(&desc,&disjoint),"CreateDisjoint");desc.Query=D3D11_QUERY_TIMESTAMP;check(device->CreateQuery(&desc,&start),"CreateTimestamp");check(device->CreateQuery(&desc,&end),"CreateTimestamp");
        auto begin=std::chrono::steady_clock::now();context->Begin(disjoint.Get());context->End(start.Get());operation();context->End(end.Get());context->End(disjoint.Get());context->Flush();
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT d{};uint64_t a=0,b=0;
        auto wait=[&](ID3D11Query* q,void* p,unsigned n) {for(;;){HRESULT hr=context->GetData(q,p,n,0);if(hr==S_OK)break;check(hr,"GetData");if(std::chrono::steady_clock::now()-begin>std::chrono::seconds(120))throw std::runtime_error("GPU query timeout");std::this_thread::sleep_for(std::chrono::milliseconds(1));}};
        wait(disjoint.Get(),&d,sizeof(d));wait(start.Get(),&a,sizeof(a));wait(end.Get(),&b,sizeof(b));if(d.Disjoint || !d.Frequency)throw std::runtime_error("disjoint GPU timestamps");return {double(b-a)*1000/d.Frequency,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()};
    }
};
static void require(bool value,const char* text) {if(!value)throw std::runtime_error(text);}
static void printError(const char* name,const chandra::Error& e) {std::cout<<",\""<<name<<"\":{\"max_abs\":"<<e.maximum<<",\"rms\":"<<e.rms<<"}";}
static void bf16Test(Engine& e,ID3D11ComputeShader* s) {
    std::vector<uint32_t> packed(32768);for(unsigned i=0;i<32768;++i)packed[i]=(2*i)|((2*i+1)<<16);
    auto input=e.buffer(unsigned(packed.size()),packed.data(),false,true),out=e.buffer(65536,nullptr,true);struct P {uint32_t count;} p{65536};
    auto m=e.time([&]{e.dispatch(s,{&input},{&out},p,256);});auto data=e.read(out);unsigned checked=0;
    for(unsigned i=0;i<65536;++i) {uint32_t bits=i<<16;float expected;std::memcpy(&expected,&bits,4);if(std::isnan(expected))require(std::isnan(data[i]),"BF16 NaN mismatch");else {uint32_t actual;std::memcpy(&actual,&data[i],4);require(actual==bits,"BF16 exact bit mismatch");}++checked;}
    std::cout<<"{\"test\":\"bf16_all_patterns\",\"passed\":true,\"checked\":"<<checked<<",\"gpu_ms\":"<<m.gpuMs<<"}\n";
}
static void rmsTest(Engine& e,ID3D11ComputeShader* s,bool gated,unsigned width=2560) {
    unsigned rows=4;std::vector<float> x(width*rows),z(x.size()),w(width);for(unsigned i=0;i<x.size();++i){x[i]=chandra::sample(i,6)*3;z[i]=chandra::sample(i,7)*2;}for(unsigned i=0;i<width;++i)w[i]=1+0.2f*chandra::sample(i,8);
    auto bx=e.floats(x),bw=e.floats(w),bz=e.floats(z),out=e.buffer(unsigned(x.size()),nullptr,true);struct P{uint32_t width,rows,gated;float eps;}p{width,rows,unsigned(gated),1e-6f};
    auto m=e.time([&]{e.dispatch(s,{&bx,&bw,&bz},{&out},p,rows);});auto data=e.read(out);std::vector<double> expected(x.size());
    for(unsigned r=0;r<rows;++r){double sum=0;for(unsigned i=0;i<width;++i)sum+=double(x[r*width+i])*x[r*width+i];double scale=1/std::sqrt(sum/width+double(p.eps));for(unsigned i=0;i<width;++i){unsigned n=r*width+i;expected[n]=x[n]*scale*w[i];if(gated)expected[n]*=double(z[n])/(1+std::exp(-double(z[n])));}}
    auto err=chandra::error(data,expected);require(err.maximum<2e-5,"RMSNorm tolerance exceeded");std::cout<<"{\"test\":\""<<(gated?"gated_rmsnorm":"rmsnorm")<<"\",\"passed\":true,\"width\":"<<width<<",\"gpu_ms\":"<<m.gpuMs;printError("double_oracle",err);std::cout<<"}\n";
}
static void deltaTest(Engine& e,ID3D11ComputeShader* s,unsigned t,unsigned h,unsigned k,unsigned v) {
    auto f=chandra::fixture(t,h,k,v);auto cpuStart=std::chrono::steady_clock::now();auto fp=chandra::recurrent<float>(f);auto dp=chandra::recurrent<double>(f);double cpuMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cpuStart).count();
    auto bq=e.floats(f.q),bk=e.floats(f.k),bv=e.floats(f.v),bg=e.floats(f.g),bb=e.floats(f.beta),state=e.buffer(h*k*v,nullptr,true),out=e.buffer(t*h*v,nullptr,true);
    struct P{uint32_t heads,keys,values,tokens,start;}p{h,k,v,t,0};auto prefill=[&]{e.dispatch(s,{&bq,&bk,&bv,&bg,&bb},{&state,&out},p,h);};
    e.zero(state);e.zero(out);auto pm=e.time(prefill);auto ps=e.read(state),po=e.read(out);auto se=chandra::error(ps,dp.state),oe=chandra::error(po,dp.output),fe=chandra::error(ps,fp.state);
    require(se.maximum<2e-5 && oe.maximum<2e-5 && fe.maximum<2e-5,"prefill recurrence tolerance exceeded");
    e.zero(state);e.zero(out);auto dm=e.time([&]{for(unsigned i=0;i<t;++i){P step{h,k,v,1,i};e.dispatch(s,{&bq,&bk,&bv,&bg,&bb},{&state,&out},step,h);}});auto ds=e.read(state),dout=e.read(out);auto de=chandra::error(ds,ps),doe=chandra::error(dout,po);require(de.maximum<2e-5 && doe.maximum<2e-5,"decode/prefill mismatch");
    e.zero(state);e.zero(out);e.time(prefill);auto reset=chandra::error(e.read(state),ps),resetOutput=chandra::error(e.read(out),po);require(reset.maximum==0 && resetOutput.maximum==0,"state reset repeat mismatch");
    std::cout<<"{\"test\":\"gated_delta_recurrence\",\"passed\":true,\"tokens\":"<<t<<",\"heads\":"<<h<<",\"keys\":"<<k<<",\"values\":"<<v<<",\"prefill_gpu_ms\":"<<pm.gpuMs<<",\"decode_gpu_ms\":"<<dm.gpuMs<<",\"decode_wall_ms\":"<<dm.wallMs<<",\"cpu_oracles_ms\":"<<cpuMs;printError("state_double",se);printError("output_double",oe);printError("state_fp32",fe);printError("decode_state",de);printError("decode_output",doe);printError("reset_state",reset);printError("reset_output",resetOutput);std::cout<<"}\n";
}
static void externalTest(Engine& e,ID3D11ComputeShader* shader,const chandra::ExternalFixture& fixture) {
    const auto& f=fixture.inputs;auto fp=chandra::recurrent<float>(f);auto dp=chandra::recurrent<double>(f);
    auto bq=e.floats(f.q),bk=e.floats(f.k),bv=e.floats(f.v),bg=e.floats(f.g),bb=e.floats(f.beta),state=e.buffer(f.heads*f.keys*f.values,nullptr,true),out=e.buffer(f.tokens*f.heads*f.values,nullptr,true);
    struct P{uint32_t heads,keys,values,tokens,start;}p{f.heads,f.keys,f.values,f.tokens,0};auto prefill=[&]{e.dispatch(shader,{&bq,&bk,&bv,&bg,&bb},{&state,&out},p,f.heads);};
    e.zero(state);e.zero(out);auto pm=e.time(prefill);auto ps=e.read(state),po=e.read(out);
    e.zero(state);e.zero(out);auto dm=e.time([&]{for(unsigned i=0;i<f.tokens;++i){P step{f.heads,f.keys,f.values,1,i};e.dispatch(shader,{&bq,&bk,&bv,&bg,&bb},{&state,&out},step,f.heads);}});auto ds=e.read(state),dout=e.read(out);
    e.zero(state);e.zero(out);e.time(prefill);auto rs=e.read(state),ro=e.read(out);
    chandra::Json errors=chandra::Json::object();bool passed=true;
    auto comparison=[&](const char* name,const auto& observed,const auto& expected,double tolerance=2e-5) {auto c=chandra::compare(observed,expected,tolerance,0);errors[name]={{"max_abs",c.absolute.maximum},{"rms",c.absolute.rms},{"max_tolerance_ratio",c.maxToleranceRatio},{"passed",c.passed}};passed=passed && c.passed;};
    comparison("prefill_state_reference",ps,fixture.expectedState);comparison("prefill_output_reference",po,fixture.expectedOutput);
    comparison("prefill_state_cpu_fp32",ps,fp.state);comparison("prefill_output_cpu_fp32",po,fp.output);
    comparison("prefill_state_cpu_double",ps,dp.state);comparison("prefill_output_cpu_double",po,dp.output);
    comparison("cpu_fp32_state_reference",fp.state,fixture.expectedState);comparison("cpu_fp32_output_reference",fp.output,fixture.expectedOutput);
    comparison("cpu_double_state_reference",dp.state,fixture.expectedState);comparison("cpu_double_output_reference",dp.output,fixture.expectedOutput);
    comparison("decode_state_prefill",ds,ps,0);comparison("decode_output_prefill",dout,po,0);comparison("reset_state_prefill",rs,ps,0);comparison("reset_output_prefill",ro,po,0);
    chandra::Json result={{"test","trained_activation_recurrence"},{"passed",passed},{"metadata_sha256",fixture.metadataHash},{"payload_sha256",fixture.payloadHash},{"tokens",f.tokens},{"heads",f.heads},{"keys",f.keys},{"values",f.values},{"atol",2e-5},{"rtol",0},{"prefill_gpu_ms",pm.gpuMs},{"decode_gpu_ms",dm.gpuMs},{"decode_wall_ms",dm.wallMs},{"errors",errors}};
    std::cout<<result.dump()<<"\n";require(passed,"external trained activation comparison failed");
}
int wmain(int argc,wchar_t** argv) {
    try {
        std::wstring adapter=L"A770",shaders=L"shaders",fixturePath;
        bool smallOnly=false,listOnly=false,validateOnly=false;int adapterIndex=-1;std::string pci,luid,expectedMetadataHash;
        auto ascii=[](const wchar_t* value) {std::wstring w(value);std::string r;for(auto c:w){if(c>127)throw std::runtime_error("identity/hash must be ASCII");r.push_back(char(c));}return r;};
        for(int i=1;i<argc;++i) {
            std::wstring a=argv[i];
            if(a==L"--adapter" && i+1<argc)adapter=argv[++i];
            else if(a==L"--shader-dir" && i+1<argc)shaders=argv[++i];
            else if(a==L"--adapter-index" && i+1<argc){adapterIndex=std::stoi(argv[++i]);if(adapterIndex<0)throw std::runtime_error("adapter index must be nonnegative");}
            else if(a==L"--pci" && i+1<argc)pci=ascii(argv[++i]);
            else if(a==L"--luid" && i+1<argc)luid=ascii(argv[++i]);
            else if(a==L"--fixture" && i+1<argc)fixturePath=argv[++i];
            else if(a==L"--fixture-sha256" && i+1<argc)expectedMetadataHash=ascii(argv[++i]);
            else if(a==L"--small-only")smallOnly=true;
            else if(a==L"--list-adapters")listOnly=true;
            else if(a==L"--validate-fixture")validateOnly=true;
            else throw std::runtime_error("unsupported/missing argument; see ChandraNative README");
        }
        if(listOnly){for(const auto& c:chandra::intelAdapters(adapter))std::cout<<c.identity.dump()<<"\n";return 0;}
        std::unique_ptr<chandra::ExternalFixture> fixture;
        if(!fixturePath.empty()){
            fixture=std::make_unique<chandra::ExternalFixture>(chandra::loadFixture(std::filesystem::path(fixturePath)));
            require(expectedMetadataHash.empty() || expectedMetadataHash==fixture->metadataHash,"requested fixture metadata SHA256 mismatch");
            if(validateOnly){std::cout<<chandra::Json{{"test","external_fixture_validation"},{"passed",true},{"metadata_sha256",fixture->metadataHash},{"payload_sha256",fixture->payloadHash},{"tokens",fixture->inputs.tokens}}.dump()<<"\n";return 0;}
        }else require(!validateOnly && expectedMetadataHash.empty(),"fixture option without fixture");
        Engine e(adapter,adapterIndex,pci,luid);std::cout.precision(10);
        std::cout<<chandra::Json{{"adapter_identity",e.identity},{"backend","D3D11 cs_5_0 FP32"},{"full_chandra_graph",false}}.dump()<<"\n";
        auto delta=e.shader(shaders+L"/delta.hlsl");
        if(fixture)externalTest(e,delta.Get(),*fixture);
        else {auto bf=e.shader(shaders+L"/bf16.hlsl"),rms=e.shader(shaders+L"/rmsnorm.hlsl");bf16Test(e,bf.Get());rmsTest(e,rms.Get(),false);rmsTest(e,rms.Get(),true);rmsTest(e,rms.Get(),true,128);deltaTest(e,delta.Get(),32,2,8,8);if(!smallOnly)deltaTest(e,delta.Get(),1000,32,128,128);}
        return 0;
    }catch(const std::exception& ex){std::cerr<<"FAILED: "<<ex.what()<<"\n";return 1;}
}
