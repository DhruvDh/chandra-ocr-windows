// New ChandraNative code, MPL-2.0. Windows host and wmain of the public, model-free text attention softmax
// multiwave regression. All checks, admission and receipt logic live in text_softmax_regression.h, which
// the CPU suite exercises; this file only supplies Win32 services: a CREATE_NEW receipt rewritten and
// flushed in place, read-only queries of the externally owned Job, executable, DXGI driver version and
// loaded HLSL compiler, and console-control cancellation. It never creates, joins or modifies a Job, loads
// no model and touches no service. Root owns the Job, CPU placement, build, device admission and execution;
// see docs/directcompute-exact-input-correctness.md.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "text_softmax_regression.h"
#include <windows.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <atomic>
#include <cstdio>
#include <iostream>

using namespace chandra::text_softmax_regression;
static_assert(jobLimitProcessTime == JOB_OBJECT_LIMIT_PROCESS_TIME && jobLimitJobTime == JOB_OBJECT_LIMIT_JOB_TIME &&
              jobLimitProcessMemory == JOB_OBJECT_LIMIT_PROCESS_MEMORY && jobLimitJobMemory == JOB_OBJECT_LIMIT_JOB_MEMORY &&
              jobLimitKillOnJobClose == JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, "Portable Job limit constants");

namespace {
std::atomic<bool> cancellationRequested{false};
std::atomic<DWORD> cancellationSignal{0xffffffffu};
HANDLE finishedEvent = nullptr;
// Ctrl+C and Ctrl+Break request cancellation at the next admission boundary. Close, logoff and shutdown
// end the process when this handler returns, so it waits (bounded) for the main thread's failure receipt
// and drain. Root's kill-on-close Job remains the hard retirement bound in every case.
BOOL WINAPI onConsoleControl(DWORD type) {
    cancellationSignal.store(type); cancellationRequested.store(true);
    if ((type == CTRL_CLOSE_EVENT || type == CTRL_LOGOFF_EVENT || type == CTRL_SHUTDOWN_EVENT) && finishedEvent)
        WaitForSingleObject(finishedEvent, 4000);
    return TRUE;
}
std::filesystem::path modulePath(HMODULE module) {
    std::vector<wchar_t> buffer(32768);
    const DWORD length = GetModuleFileNameW(module, buffer.data(), DWORD(buffer.size()));
    require(length > 0 && length < buffer.size(), "Module path unavailable");
    return std::filesystem::path(std::wstring(buffer.data(), length));
}
struct ReceiptFile {
    HANDLE handle = INVALID_HANDLE_VALUE;
    bool create(const std::filesystem::path& path) {
        handle = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        return handle != INVALID_HANDLE_VALUE;
    }
    void rewrite(const std::string& text) {
        require(handle != INVALID_HANDLE_VALUE && text.size() <= receiptCap, "Receipt not open or text above its cap");
        LARGE_INTEGER first{};
        require(SetFilePointerEx(handle, first, nullptr, FILE_BEGIN) != 0, "Receipt seek failed");
        DWORD written = 0;
        require(WriteFile(handle, text.data(), DWORD(text.size()), &written, nullptr) != 0 && written == text.size(), "Receipt write failed");
        require(SetEndOfFile(handle) != 0 && FlushFileBuffers(handle) != 0, "Receipt truncate/flush failed");
    }
    void close() {
        if (handle != INVALID_HANDLE_VALUE) { CloseHandle(handle); handle = INVALID_HANDLE_VALUE; }
    }
};
JobLimits queryJob() {
    JobLimits result; BOOL inJob = FALSE;
    require(IsProcessInJob(GetCurrentProcess(), nullptr, &inJob) != 0, "IsProcessInJob failed");
    result.inJob = inJob != FALSE;
    if (!result.inJob) return result;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    require(QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &info, sizeof(info), nullptr) != 0,
            "External Job limits cannot be queried");
    result.flags = info.BasicLimitInformation.LimitFlags;
    result.processMemory = info.ProcessMemoryLimit; result.jobMemory = info.JobMemoryLimit;
    return result;
}
// User-mode driver version of the adapter with the pinned LUID, read before Device creation.
Json driverIdentity(const std::string& luid) {
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    require(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))), "CreateDXGIFactory1 failed");
    for (UINT i = 0;; ++i) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        const HRESULT hr = factory->EnumAdapters1(i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        require(SUCCEEDED(hr), "EnumAdapters1 failed");
        DXGI_ADAPTER_DESC1 d{};
        require(SUCCEEDED(adapter->GetDesc1(&d)), "GetDesc1 failed");
        char text[32];
        std::snprintf(text, sizeof(text), "%08x:%08x", unsigned(d.AdapterLuid.HighPart), unsigned(d.AdapterLuid.LowPart));
        if (luid != text) continue;
        LARGE_INTEGER version{};
        require(SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &version)), "Adapter driver version query failed");
        const uint64_t v = uint64_t(version.QuadPart);
        char umd[64];
        std::snprintf(umd, sizeof(umd), "%u.%u.%u.%u", unsigned(v >> 48), unsigned((v >> 32) & 0xffffu), unsigned((v >> 16) & 0xffffu),
                      unsigned(v & 0xffffu));
        return {{"luid", text}, {"umd_driver_version", umd}, {"vendor_id", d.VendorId}, {"device_id", d.DeviceId},
                {"subsystem_id", d.SubSysId}, {"revision", d.Revision},
                {"source", "IDXGIAdapter1::CheckInterfaceSupport(IDXGIDevice) before Device creation"}};
    }
    throw std::runtime_error("No DXGI adapter has the pinned LUID");
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    finishedEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!finishedEvent || !SetConsoleCtrlHandler(onConsoleControl, TRUE)) {
        std::cerr << Json{{"state", "REFUSED"}, {"schema", schema}, {"device_created", false},
                          {"error", "Console cancellation handler unavailable"}}.dump() << "\n";
        return exitRefused;
    }
    ReceiptFile receipt;
    Host host;
    host.createFresh = [&receipt](const std::filesystem::path& path) { return receipt.create(path); };
    host.rewrite = [&receipt](const std::string& text) { receipt.rewrite(text); };
    host.close = [&receipt] { receipt.close(); };
    host.queryJob = queryJob;
    host.producer = [] {
        return Json{{"executable", fileCommitment(modulePath(nullptr), 64ull << 20)},
#ifdef _MSC_FULL_VER
                    {"msvc_full_version", _MSC_FULL_VER},
#endif
                    {"source_schema", schema}};
    };
    host.driver = driverIdentity;
    host.shaderCompiler = [] {
        const HMODULE module = GetModuleHandleW(L"d3dcompiler_47.dll");
        return module ? fileCommitment(modulePath(module), 64ull << 20) : Json::object();
    };
    host.cancelled = [] { return cancellationRequested.load(); };
    host.now = [] { return Clock::now(); };
    const int code = entry(std::vector<std::wstring>(argv + 1, argv + argc), host, std::cout, std::cerr);
    receipt.close();
    if (code == exitCancelled) std::cerr << "Console control signal " << cancellationSignal.load() << "\n";
    SetEvent(finishedEvent);
    return code;
}
