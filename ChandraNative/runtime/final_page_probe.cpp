// New ChandraNative code, MPL-2.0. Windows host for the model-free final-page probe (final_page_probe.h).
// Inactive by default: without --execute it prints the forecast plan and creates no Device, listener or model.
// --execute requires the explicit adapter PCI and LUID (strict chandra::selectAdapter, no fallback) and an
// externally owned kill-on-close Job with a process memory limit. UNCOMPILED IN THIS ISOLATE: root builds it.
// Source12 repair: every file is admitted from memory with a pinned size and SHA-256 before the Job screen,
// adapter or Device (p::run); shaders compile from those exact bytes with no include handler; each constant
// upload is a full zero-initialised 256-byte p::Constants; every operation is timed by p::timedOperation from
// before submission through drain and Map; every view and copy box is bounds-checked against GetDesc.
//
// Separation of paths. REUSED production code: chandra::selectAdapter/intelAdapters (adapter_identity.h), the
// production buffer flags (DEFAULT, SRV|UAV, ALLOW_RAW_VIEWS) and UpdateSubresource upload, the production
// shader compile flags, and the unchanged, hash-pinned shader files runtime/weight_row_checksum.hlsl and
// runtime/weight_row_words.hlsl. INDEPENDENT code: the D3D11 device, buffer, raw SRV with an explicit logical
// NumElements, and the direct CopySubresourceRegion-to-staging read, which uses no shader and no SRV and runs
// alone with --direct-only (no shader admission or compilation). The
// production Device class is deliberately not used, so its readback/dispatch path cannot establish identity.
#include "final_page_probe.h"
#include "../adapter_identity.h"
#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace p = chandra::final_page_probe;
using Microsoft::WRL::ComPtr;
namespace {
void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
void checked(HRESULT hr, const char* what) { if (FAILED(hr)) { char t[16]; std::snprintf(t, sizeof(t), "0x%08x", unsigned(hr)); throw std::runtime_error(std::string(what) + " failed " + t); } }
std::string narrow(const wchar_t* w) { std::string s; for (; *w; ++w) { require(*w < 128, "Non-ASCII argument"); s.push_back(char(*w)); } return s; }
uint64_t byteWidth(ID3D11Buffer* b) { D3D11_BUFFER_DESC d{}; b->GetDesc(&d); return d.ByteWidth; }

p::Json jobScreen() {
    BOOL inJob = FALSE; require(IsProcessInJob(GetCurrentProcess(), nullptr, &inJob) != 0, "IsProcessInJob failed");
    require(inJob != FALSE, "--execute requires an externally owned Job");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    require(QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &info, sizeof(info), nullptr) != 0, "Job limits unreadable");
    const DWORD flags = info.BasicLimitInformation.LimitFlags;
    require((flags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) != 0, "Job must kill on close");
    require((flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) != 0 && info.ProcessMemoryLimit > 0 && info.ProcessMemoryLimit <= 1024ull * 1024 * 1024,
            "Job must impose a finite process memory limit of at most 1 GiB");
    return {{"in_job", true}, {"limit_flags", flags}, {"process_memory_limit", uint64_t(info.ProcessMemoryLimit)},
            {"wall_timer_and_job_handle_owner", "not observable in process; root retains the external receipt"}};
}

struct Gpu {
    ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> context; ComPtr<ID3D11Buffer> constants;
    ComPtr<ID3D11ComputeShader> checksum, rowCopy; p::Json shaders = p::Json::object();
    p::Timer* timer = nullptr;

    // One timed operation: submit() runs every API submission call, then an event query is ended and flushed, then
    // the drain is polled with DONOTFLUSH, then mapped() polls Map with DO_NOT_WAIT and copies out.
    template <class Submit, class Mapped> void operation(const char* label, uint64_t limitMs, Submit&& submit, Mapped&& mapped) {
        ComPtr<ID3D11Query> q;
        p::timedOperation(*timer, label, limitMs,
            [&] { submit(); D3D11_QUERY_DESC desc{D3D11_QUERY_EVENT, 0}; checked(device->CreateQuery(&desc, &q), "CreateQuery");
                  context->End(q.Get()); context->Flush(); },
            [&] { BOOL done = FALSE; const HRESULT hr = context->GetData(q.Get(), &done, sizeof(done), D3D11_ASYNC_GETDATA_DONOTFLUSH);
                  checked(hr, "Drain query"); return hr == S_OK && done != FALSE; },
            mapped);
    }
    ComPtr<ID3D11ComputeShader> compile(const p::AdmittedFile& file, const char* label) {
        require(file.bytes.size() <= p::shaderSourceLimit, "Shader source bound");
        shaders[label] = {{"bytes", file.bytes.size()}, {"sha256", file.sha256}, {"profile", "cs_5_0"}, {"include_handler", nullptr},
                          {"flags", "D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS"}};
        ComPtr<ID3DBlob> code, errors; // Null include handler: any #include fails compilation.
        const HRESULT hr = D3DCompile(file.bytes.data(), file.bytes.size(), label, nullptr, nullptr, "main", "cs_5_0",
                                      D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_IEEE_STRICTNESS, 0, &code, &errors);
        checked(hr, label); require(code && code->GetBufferSize() > 0 && code->GetBufferSize() <= p::bytecodeLimit, "Bytecode bound");
        ComPtr<ID3D11ComputeShader> s;
        checked(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &s), "CreateComputeShader");
        shaders[label]["bytecode_bytes"] = code->GetBufferSize(); shaders[label]["bytecode_sha256"] = p::sha256(code->GetBufferPointer(), code->GetBufferSize());
        return s;
    }
    // Production buffer flags; ByteWidth is the requested width, the raw SRV covers srvWords and the UAV uavWords only.
    void create(uint64_t width, uint32_t srvWords, ComPtr<ID3D11Buffer>& buffer, ComPtr<ID3D11ShaderResourceView>* srv,
                ComPtr<ID3D11UnorderedAccessView>* uav, uint32_t uavWords) {
        p::requireViewWords(width, srv ? srvWords : 0); p::requireViewWords(width, uav ? uavWords : 0);
        require((!srv || srvWords > 0) && (!uav || uavWords > 0), "Empty view");
        D3D11_BUFFER_DESC d{}; d.ByteWidth = UINT(width); d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS; d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        checked(device->CreateBuffer(&d, nullptr, &buffer), "CreateBuffer");
        if (srv) {
            D3D11_SHADER_RESOURCE_VIEW_DESC s{}; s.Format = DXGI_FORMAT_R32_TYPELESS; s.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
            s.BufferEx.NumElements = srvWords; s.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
            checked(device->CreateShaderResourceView(buffer.Get(), &s, srv->GetAddressOf()), "Create raw SRV");
        }
        if (uav) {
            D3D11_UNORDERED_ACCESS_VIEW_DESC u{}; u.Format = DXGI_FORMAT_R32_TYPELESS; u.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            u.Buffer.NumElements = uavWords; u.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
            checked(device->CreateUnorderedAccessView(buffer.Get(), &u, uav->GetAddressOf()), "Create raw UAV");
        }
    }
    using Regions = std::vector<std::pair<uint64_t, uint32_t>>;
    // Copies regions of source into a fresh 8 KiB staging buffer; every box is checked against both ByteWidths.
    void copyToStaging(ID3D11Buffer* source, ID3D11Buffer* staging, const Regions& regions) {
        uint64_t at = 0; const uint64_t sourceWidth = byteWidth(source), stagingWidth = byteWidth(staging);
        for (const auto& [first, bytes] : regions) {
            p::requireCopyBox(sourceWidth, first, bytes, at, stagingWidth);
            D3D11_BOX box{UINT(first), 0, 0, UINT(first + bytes), 1, 1};
            context->CopySubresourceRegion(staging, 0, UINT(at), 0, 0, source, 0, &box); at += bytes;
        }
    }
    ComPtr<ID3D11Buffer> staging() {
        D3D11_BUFFER_DESC d{}; d.ByteWidth = p::stagingBytes; d.Usage = D3D11_USAGE_STAGING; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Buffer> s; checked(device->CreateBuffer(&d, nullptr, &s), "Create staging"); return s;
    }
    auto mapInto(ID3D11Buffer* staging, std::vector<uint32_t>& out, uint32_t bytes) {
        return [this, staging, &out, bytes] {
            require(bytes % 4 == 0 && bytes <= p::stagingBytes, "Readback bound");
            D3D11_MAPPED_SUBRESOURCE m{}; const HRESULT hr = context->Map(staging, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &m);
            if (hr == DXGI_ERROR_WAS_STILL_DRAWING) return false;
            checked(hr, "Map staging"); out.assign(bytes / 4, 0); std::memcpy(out.data(), m.pData, bytes); context->Unmap(staging, 0); return true;
        };
    }
    // Independent direct read: no shader, no SRV.
    std::vector<uint32_t> direct(ID3D11Buffer* source, const Regions& regions, const char* label) {
        uint32_t bytes = 0; for (const auto& r : regions) bytes += r.second;
        auto s = staging(); std::vector<uint32_t> out;
        operation(label, p::slowOperationMs, [&] { copyToStaging(source, s.Get(), regions); }, mapInto(s.Get(), out, bytes));
        return out;
    }
    void bind(ID3D11ComputeShader* shader, ID3D11ShaderResourceView* srv, ID3D11UnorderedAccessView* uav, const p::Constants& c, UINT groups) {
        require(groups >= 1 && groups <= 8, "Dispatch bound");
        p::requireFullConstantUpload(byteWidth(constants.Get()), sizeof(c));
        context->UpdateSubresource(constants.Get(), 0, nullptr, &c, 0, 0);
        context->CSSetShader(shader, nullptr, 0); context->CSSetConstantBuffers(0, 1, constants.GetAddressOf());
        context->CSSetShaderResources(0, 1, &srv); context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
        context->Dispatch(groups, 1, 1);
        ID3D11ShaderResourceView* noSrv = nullptr; ID3D11UnorderedAccessView* noUav = nullptr;
        context->CSSetShaderResources(0, 1, &noSrv); context->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
    }
    // Dispatch plus copy of output [0, bytes) to staging plus drain plus Map, timed as one operation.
    std::vector<uint32_t> dispatchRead(ID3D11ComputeShader* shader, ID3D11ShaderResourceView* srv, ID3D11Buffer* output,
                                       ID3D11UnorderedAccessView* uav, const p::Constants& c, UINT groups, uint32_t bytes, const char* label) {
        auto s = staging(); std::vector<uint32_t> out;
        operation(label, p::slowOperationMs, [&] { bind(shader, srv, uav, c, groups); copyToStaging(output, s.Get(), {{0, bytes}}); },
                  mapInto(s.Get(), out, bytes));
        return out;
    }
};

p::Json readbackRecord(const std::vector<uint32_t>& w) { return {{"bytes", w.size() * 4}, {"sha256", p::sha256(w.data(), w.size() * 4)}}; }

p::Json observe(Gpu& g, p::Contrast c, ID3D11Buffer* big, ID3D11ShaderResourceView* srv, const std::vector<uint32_t>& cpu, const char* phase, bool directOnly) {
    p::Observation o;
    Gpu::Regions regions{{p::directFirstByte, p::directBytes}};
    if (c == p::Contrast::padded) regions.push_back({p::logicalBytes, p::paddingBytes});
    auto words = g.direct(big, regions, "direct_copy");
    require(words.size() >= p::rowWords, "Direct readback geometry");
    o.direct.assign(words.begin(), words.begin() + p::rowWords); o.padding.assign(words.begin() + p::rowWords, words.end());
    p::Json records = {{"direct_copy", readbackRecord(words)}};
    if (!directOnly) {
        ComPtr<ID3D11Buffer> out; ComPtr<ID3D11UnorderedAccessView> uav; g.create(p::rowBytes, 0, out, nullptr, &uav, p::rowWords);
        o.checksums = g.dispatchRead(g.checksum.Get(), srv, out.Get(), uav.Get(), p::params({p::rowWords, p::selectedRows, p::selectedFirstRow, 0}),
                                     p::groups(p::selectedRows, p::checksumThreads), p::selectedRows * 4, "checksum");
        o.rowCopy = g.dispatchRead(g.rowCopy.Get(), srv, out.Get(), uav.Get(), p::params({p::finalRow * p::rowWords, p::rowWords, 0}),
                                   p::groups(p::rowWords, p::rowCopyThreads), p::rowBytes, "row_copy");
        records["checksums"] = readbackRecord(o.checksums); records["row_copy"] = readbackRecord(o.rowCopy);
    }
    auto j = p::judge(cpu, c, o, directOnly); j["phase"] = phase; j["readbacks"] = records;
    return j;
}

// Bounded churn: allocate one 1 MiB scratch, copy the final row into it (SRV shader, or direct copy in direct-only
// mode), drain, release; eight times, one scratch live at a time.
p::Json churn(Gpu& g, ID3D11Buffer* big, ID3D11ShaderResourceView* srv, bool directOnly) {
    for (uint32_t i = 0; i < p::churnIterations; ++i) {
        ComPtr<ID3D11Buffer> scratch; ComPtr<ID3D11UnorderedAccessView> uav; g.create(uint64_t(p::churnWords) * 4, 0, scratch, nullptr, &uav, p::churnWords);
        g.operation("churn", p::slowOperationMs, [&] {
            if (directOnly) {
                p::requireCopyBox(byteWidth(big), p::directFirstByte, p::rowBytes, 0, byteWidth(scratch.Get()));
                D3D11_BOX box{UINT(p::directFirstByte), 0, 0, UINT(p::directFirstByte + p::rowBytes), 1, 1};
                g.context->CopySubresourceRegion(scratch.Get(), 0, 0, 0, 0, big, 0, &box);
            } else g.bind(g.rowCopy.Get(), srv, uav.Get(), p::params({p::finalRow * p::rowWords, p::rowWords, 0}), p::groups(p::rowWords, p::rowCopyThreads));
        }, [] { return true; });
    }
    return {{"iterations", p::churnIterations}, {"scratch_bytes", uint64_t(p::churnWords) * 4}, {"mode", directOnly ? "direct_copy" : "srv_row_copy"}};
}

p::Json driver(const std::string& luid) {
    ComPtr<IDXGIFactory1> f; checked(CreateDXGIFactory1(IID_PPV_ARGS(&f)), "CreateDXGIFactory1");
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> a; const HRESULT hr = f->EnumAdapters1(i, &a); if (hr == DXGI_ERROR_NOT_FOUND) break; checked(hr, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 d{}; checked(a->GetDesc1(&d), "GetDesc1"); char t[32];
        std::snprintf(t, sizeof(t), "%08x:%08x", unsigned(d.AdapterLuid.HighPart), unsigned(d.AdapterLuid.LowPart));
        if (luid != t) continue; LARGE_INTEGER v{}; checked(a->CheckInterfaceSupport(__uuidof(IDXGIDevice), &v), "Driver version");
        const uint64_t q = uint64_t(v.QuadPart); char u[64];
        std::snprintf(u, sizeof(u), "%u.%u.%u.%u", unsigned(q >> 48), unsigned((q >> 32) & 0xffff), unsigned((q >> 16) & 0xffff), unsigned(q & 0xffff));
        return {{"luid", t}, {"umd_version", u}};
    }
    throw std::runtime_error("Pinned LUID absent");
}

struct WindowsHost final : p::Host {
    p::Json execute(const p::Arguments& a, const p::Admitted& m) override {
        p::SteadyClock clock;
        p::Timer timer(clock, 2 * (a.directOnly ? p::directOperationsPerContrast : p::shaderOperationsPerContrast));
        p::Json r = {{"schema", "chandra.directcompute.final-page-probe.v2"}, {"plan", p::plan(a.directOnly)}, {"job", jobScreen()}};
        const std::vector<uint8_t>* splice = m.sourceRow ? &m.source.bytes : nullptr;
        r["source_row"] = m.sourceRow ? p::Json{{"bytes", m.source.bytes.size()}, {"sha256", m.source.sha256}, {"spliced_at_row", p::finalRow}} : p::Json(nullptr);
        auto chosen = chandra::selectAdapter(chandra::intelAdapters(L"A770"), -1, a.pci, a.luid); // Strict; no fallback.
        r["adapter"] = chosen.identity; r["driver"] = driver(a.luid);
        Gpu g; g.timer = &timer; D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0}, actual{};
        HRESULT hr = D3D11CreateDevice(chosen.handle.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, levels, 2, D3D11_SDK_VERSION, &g.device, &actual, &g.context);
        if (hr == E_INVALIDARG) hr = D3D11CreateDevice(chosen.handle.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, levels + 1, 1, D3D11_SDK_VERSION, &g.device, &actual, &g.context);
        checked(hr, "D3D11CreateDevice"); r["feature_level"] = unsigned(actual);
        if (!a.directOnly) {
            D3D11_BUFFER_DESC cb{}; cb.ByteWidth = p::constantBufferBytes; cb.Usage = D3D11_USAGE_DEFAULT; cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            checked(g.device->CreateBuffer(&cb, nullptr, &g.constants), "Create constants");
            g.checksum = g.compile(m.checksumShader, p::checksumShaderPolicy.label); g.rowCopy = g.compile(m.rowCopyShader, p::rowCopyShaderPolicy.label);
        }
        p::Json contrasts = p::Json::array(); bool pass = true;
        for (p::Contrast c : {p::Contrast::original, p::Contrast::padded}) {
            timer.requireRun("fill");
            std::vector<uint32_t> cpu; p::fill(cpu, c, splice);
            p::Json record = {{"contrast", p::name(c)}, {"requested_byte_width", p::physicalBytes(c)}, {"srv_words", p::srvWords(c)},
                              {"logical_input_sha256", p::sha256(cpu.data(), p::logicalBytes)}, {"physical_input_sha256", p::sha256(cpu.data(), cpu.size() * 4)},
                              {"final_row_sha256", p::sha256(cpu.data() + uint64_t(p::finalRow) * p::rowWords, p::rowBytes)}};
            { // Exactly one large buffer is live inside this scope.
                ComPtr<ID3D11Buffer> big; ComPtr<ID3D11ShaderResourceView> srv; g.create(p::physicalBytes(c), p::srvWords(c), big, &srv, nullptr, 0);
                require(byteWidth(big.Get()) == cpu.size() * 4, "Upload source must equal the large buffer ByteWidth");
                g.operation("upload", p::longOperationMs, [&] { g.context->UpdateSubresource(big.Get(), 0, nullptr, cpu.data(), 0, 0); }, [] { return true; });
                record["before_churn"] = observe(g, c, big.Get(), srv.Get(), cpu, "before_churn", a.directOnly);
                record["churn"] = churn(g, big.Get(), srv.Get(), a.directOnly);
                record["after_churn"] = observe(g, c, big.Get(), srv.Get(), cpu, "after_churn", a.directOnly);
                pass = pass && record["before_churn"]["pass"] == true && record["after_churn"]["pass"] == true;
            }
            g.operation("release_large_buffer", p::longOperationMs, [&] { g.context->ClearState(); }, [] { return true; });
            contrasts.push_back(std::move(record));
        }
        timer.requireRun("report");
        r["contrasts"] = std::move(contrasts); r["shaders"] = g.shaders; r["operation_timings"] = timer.records();
        r["operations"] = timer.operations(); r["synthetic_pass"] = pass;
        r["claim_limit"] = "Copied-byte observations only; a synthetic pass cannot clear the trained native-0 failure and no observation proves physical storage";
        return r;
    }
};
} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        std::vector<std::string> args; for (int i = 1; i < argc; ++i) args.push_back(narrow(argv[i]));
        const auto a = p::parse(args);
        WindowsHost host; const auto result = p::run(a, host).dump(2); // Inactive mode returns the plan with no host call.
        if (!a.execute) { std::cout << result << "\n"; return 0; }
        require(result.size() <= p::reportLimitBytes, "Report exceeds the declared bound");
        std::ofstream out(std::filesystem::path(a.output), std::ios::binary); out << result; out.close(); require(bool(out), "Write output");
        std::cout << result << "\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "final_page_probe: " << e.what() << "\n"; return 2; }
}
