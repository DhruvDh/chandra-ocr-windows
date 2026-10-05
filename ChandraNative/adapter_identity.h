// New ChandraNative code, MPL-2.0. Bind DXGI logical identity to WDDM PCI address.
#pragma once
#include <windows.h>
#include <bcrypt.h>
#include <d3dkmthk.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <cstdio>
#include "fixture.h"
namespace chandra {
struct AdapterCandidate {
    Microsoft::WRL::ComPtr<IDXGIAdapter1> handle;DXGI_ADAPTER_DESC1 description{};
    unsigned index;Json identity;
};
inline std::vector<AdapterCandidate> intelAdapters(const std::wstring& substring) {
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;fixtureRequire(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))),"CreateDXGIFactory1 failed");std::vector<AdapterCandidate> result;
    for(unsigned i=0;;++i) {
        AdapterCandidate item{};HRESULT hr=factory->EnumAdapters1(i,&item.handle);if(hr==DXGI_ERROR_NOT_FOUND)break;fixtureRequire(SUCCEEDED(hr),"EnumAdapters1 failed");fixtureRequire(SUCCEEDED(item.handle->GetDesc1(&item.description)),"GetDesc1 failed");const auto& d=item.description;
        if(d.VendorId!=0x8086 || (d.Flags&DXGI_ADAPTER_FLAG_SOFTWARE) || std::wstring(d.Description).find(substring)==std::wstring::npos)continue;
        item.index=i;std::wstring wide(d.Description);int count=WideCharToMultiByte(CP_UTF8,0,wide.data(),int(wide.size()),nullptr,0,nullptr,nullptr);fixtureRequire(count>0,"adapter name encoding failure");std::string name(size_t(count),'\0');WideCharToMultiByte(CP_UTF8,0,wide.data(),int(wide.size()),name.data(),count,nullptr,nullptr);
        char luid[32];std::snprintf(luid,sizeof(luid),"%08x:%08x",unsigned(d.AdapterLuid.HighPart),unsigned(d.AdapterLuid.LowPart));
        item.identity={{"dxgi_index",i},{"description",name},{"vendor_id",d.VendorId},{"device_id",d.DeviceId},{"subsystem_id",d.SubSysId},{"revision",d.Revision},{"luid",luid},{"dedicated_bytes",uint64_t(d.DedicatedVideoMemory)}};
        D3DKMT_OPENADAPTERFROMLUID open{};open.AdapterLuid=d.AdapterLuid;NTSTATUS status=D3DKMTOpenAdapterFromLuid(&open);
        if(status>=0) {
            D3DKMT_ADAPTERADDRESS address{};D3DKMT_QUERYADAPTERINFO query{};query.hAdapter=open.hAdapter;query.Type=KMTQAITYPE_ADAPTERADDRESS;query.pPrivateDriverData=&address;query.PrivateDriverDataSize=sizeof(address);status=D3DKMTQueryAdapterInfo(&query);
            D3DKMT_CLOSEADAPTER close{};close.hAdapter=open.hAdapter;D3DKMTCloseAdapter(&close);
            if(status>=0 && address.BusNumber<=255 && address.DeviceNumber<=31 && address.FunctionNumber<=7) {char pci[32];std::snprintf(pci,sizeof(pci),"%02x:%02x.%x",address.BusNumber,address.DeviceNumber,address.FunctionNumber);item.identity["pci_bdf"]=pci;item.identity["pci_bus"]=address.BusNumber;item.identity["pci_device"]=address.DeviceNumber;item.identity["pci_function"]=address.FunctionNumber;}else if(status>=0){item.identity["pci_bdf"]=nullptr;item.identity["pci_address_raw"]={address.BusNumber,address.DeviceNumber,address.FunctionNumber};}
        }
        if(status<0){item.identity["pci_bdf"]=nullptr;item.identity["pci_query_ntstatus"]=uint32_t(status);}result.push_back(std::move(item));
    }
    return result;
}
inline AdapterCandidate selectAdapter(const std::vector<AdapterCandidate>& candidates,int index,const std::string& pci,const std::string& luid) {
    const AdapterCandidate* selected=nullptr;
    for(const auto& item:candidates) {
        if(index>=0 && item.index!=unsigned(index))continue;
        if(!pci.empty() && item.identity.at("pci_bdf")!=pci)continue;
        if(!luid.empty() && item.identity.at("luid")!=luid)continue;
        fixtureRequire(selected==nullptr,"ambiguous Intel adapter; select PCI/LUID or index plus PCI guard");selected=&item;
    }
    fixtureRequire(selected!=nullptr,"requested Intel adapter identity not found");fixtureRequire(selected->identity.at("pci_bdf").is_string(),"selected adapter PCI identity unavailable");return *selected;
}
}
