// New ChandraNative code, MPL-2.0. Public, model-free, finite GEMM candidate calibration (Windows
// entry). run() in gemm_candidate_fixture.h dispatches runtime/linear.hlsl and the two inactive
// candidates directly through the unchanged Device with the same 32-byte cbuffer, bindings and
// groups, one dispatch per profile window, and compares every output word with an exact CPU
// oracle, the predecessor and per-dispatch sentinels. This file adds argument refusal, external
// Job admission, SHA-256 (BCrypt), producer commitments and the fresh bounded receipt. Root owns
// the Job, CPU placement, build, device admission and execution; docs/directcompute-gemm-candidates.md.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "api.h"
#include "gemm_candidate_fixture.h"
#include <windows.h>
#include <bcrypt.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

using namespace chandra::gemm_candidates;
using chandra::dc::Device;
static_assert(jobLimitProcessTime == JOB_OBJECT_LIMIT_PROCESS_TIME && jobLimitJobTime == JOB_OBJECT_LIMIT_JOB_TIME &&
              jobLimitProcessMemory == JOB_OBJECT_LIMIT_PROCESS_MEMORY && jobLimitJobMemory == JOB_OBJECT_LIMIT_JOB_MEMORY &&
              jobLimitKillOnJobClose == JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, "Portable Job limit constants");
static_assert(sizeof(LinearParams) == 32 && sizeof(uint32_t) == 4, "linear.hlsl cbuffer ABI");

namespace {
// SHA-256 of raw bytes; word vectors are hashed as their little-endian x64 storage.
std::string sha256(const void* data, size_t bytes) {
    BCRYPT_ALG_HANDLE algorithm = nullptr; BCRYPT_HASH_HANDLE hash = nullptr;
    require(BCRYPT_SUCCESS(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0)), "SHA-256 provider unavailable");
    std::unique_ptr<void, void (*)(void*)> closeAlgorithm(algorithm, [](void* h) { BCryptCloseAlgorithmProvider(h, 0); });
    require(BCRYPT_SUCCESS(BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0)), "SHA-256 hash creation failed");
    std::unique_ptr<void, void (*)(void*)> closeHash(hash, [](void* h) { BCryptDestroyHash(h); });
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t done = 0; done < bytes;) {
        const ULONG chunk = ULONG(std::min<size_t>(bytes - done, 1u << 30));
        require(BCRYPT_SUCCESS(BCryptHashData(hash, const_cast<PUCHAR>(p + done), chunk, 0)), "SHA-256 update failed");
        done += chunk;
    }
    unsigned char digest[32];
    require(BCRYPT_SUCCESS(BCryptFinishHash(hash, digest, sizeof(digest), 0)), "SHA-256 finish failed");
    static const char* hex = "0123456789abcdef"; std::string text;
    for (unsigned char b : digest) { text.push_back(hex[b >> 4]); text.push_back(hex[b & 15]); }
    return text;
}
Json fileCommitment(const std::filesystem::path& path, uint64_t cap) {
    const uint64_t size = std::filesystem::file_size(path);
    require(size > 0 && size <= cap, "Producer file empty or above its size bound: " + path.u8string());
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "Producer file unreadable: " + path.u8string());
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    require(bytes.size() == size, "Producer file changed while it was read: " + path.u8string());
    return {{"path", path.u8string()}, {"bytes", bytes.size()}, {"sha256", sha256(bytes.data(), bytes.size())}};
}
std::filesystem::path executablePath() {
    std::vector<wchar_t> buffer(32768);
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), DWORD(buffer.size()));
    require(length > 0 && length < buffer.size(), "Executable path unavailable");
    return std::filesystem::path(std::wstring(buffer.data(), length));
}
std::string ascii(const std::wstring& text) {
    std::string result;
    for (wchar_t c : text) { require(c > 0 && c < 128, "ASCII PCI and LUID required"); result.push_back(char(c)); }
    return result;
}
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
void writeFresh(HANDLE file, const Json& report) {
    const std::string text = report.dump(2) + "\n";
    require(text.size() <= receiptCapBytes, "Complete calibration receipt exceeded 8 MiB");
    LARGE_INTEGER first{};
    require(SetFilePointerEx(file, first, nullptr, FILE_BEGIN) != 0, "Receipt seek failed");
    DWORD written = 0;
    require(WriteFile(file, text.data(), DWORD(text.size()), &written, nullptr) != 0 && written == text.size(), "Receipt write failed");
    require(SetEndOfFile(file) != 0 && FlushFileBuffers(file) != 0, "Receipt truncate/flush failed");
}

} // namespace

int wmain(int argc, wchar_t** argv) {
    Arguments args;
    Json report;
    const auto cases = plan();
    try {
        args = parseArguments(std::vector<std::wstring>(argv + 1, argv + argc));
        report = initialReport(cases);
        if (!args.execute) {
            Json inactive = {{"state", "INACTIVE"}, {"device_created", false}, {"model_loaded", false},
                             {"correctness_dispatches", report.at("correctness_dispatches")},
                             {"timing_dispatches_if_selected", report.at("timing_dispatches_if_selected")}, {"cases", Json::array()}};
            for (const auto& c : report.at("plan")) inactive["cases"].push_back(c.at("name"));
            std::cout << inactive.dump() << "\n";
            return 0;
        }
        report["external_job"] = admitJob(queryJob());
        const std::filesystem::path root(args.shaderRoot), output(args.output);
        require(root.is_absolute() && std::filesystem::is_directory(root), "Absolute existing shader root required");
        for (const auto& kernel : kernels)
            require(std::filesystem::is_regular_file(root / L"runtime" / kernel.file), std::string("Shader absent: runtime/") + kernel.file);
        require(output.is_absolute() && output.has_filename() && std::filesystem::is_directory(output.parent_path()),
                "Absolute fresh receipt path in an existing directory required");
    } catch (const std::exception& error) {
        std::cerr << Json{{"state", "REFUSED"}, {"device_created", false}, {"error", error.what()}}.dump() << "\n";
        return 2;
    }
    HANDLE receipt = INVALID_HANDLE_VALUE;
    try {
        receipt = CreateFileW(std::filesystem::path(args.output).c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
        require(receipt != INVALID_HANDLE_VALUE, "Fresh task-owned calibration receipt required (CREATE_NEW)");
        const Hooks hooks{[](const void* data, size_t bytes) { return sha256(data, bytes); },
                          [receipt](const Json& current) { writeFresh(receipt, current); }};
        const std::filesystem::path root(args.shaderRoot);
        const std::string pci = ascii(args.pci), luid = ascii(args.luid);
        report["plan_selector"] = args.plan == PlanSelector::correctness ? "correctness" : "correctness-then-timing";
        report["shader_root"] = root.u8string(); report["requested_pci"] = pci; report["requested_luid"] = luid;
        report["producer"] = {{"executable", fileCommitment(executablePath(), 64ull << 20)},
#ifdef _MSC_FULL_VER
                              {"msvc", _MSC_FULL_VER},
#endif
                              {"shaders_before", Json::array()}};
        for (const auto& kernel : kernels)
            report["producer"]["shaders_before"].push_back(fileCommitment(root / L"runtime" / kernel.file, 1u << 20));
        writeFresh(receipt, report);
        Deadline deadline;
        {
            Device device(root.wstring(), pci, luid);
            report["native_executed"] = true;
            report["device_identity"] = Json::parse(device.identityJson());
            authenticateIdentity(report.at("device_identity"), pci, luid);
            report["memory_before"] = Json::parse(device.memoryJson());
            writeFresh(receipt, report);
            run(device, report, hooks, args.plan, deadline, cases);
            device.drain(); deadline.check();
            require(device.trackedBufferBytes() == 0, "Completed calibration left owned buffers live");
            report["memory_after"] = Json::parse(device.memoryJson());
        }
        report["producer"]["shaders_after"] = Json::array();
        for (const auto& kernel : kernels)
            report["producer"]["shaders_after"].push_back(fileCommitment(root / L"runtime" / kernel.file, 1u << 20));
        require(report["producer"]["shaders_after"] == report["producer"]["shaders_before"], "Shader files changed during the run");
        report["elapsed_seconds"] = deadline.seconds();
        report["passed"] = true;
        report["state"] = args.plan == PlanSelector::correctness ? "GEMM_CANDIDATE_CORRECTNESS_PASS" : "GEMM_CANDIDATE_CORRECTNESS_AND_TIMING_PASS";
        writeFresh(receipt, report); CloseHandle(receipt); receipt = INVALID_HANDLE_VALUE;
        std::cout << Json{{"state", report.at("state")}, {"passed", true}, {"full_model_accepted", false},
                          {"page_throughput_accepted", false}}.dump() << "\n";
        return 0;
    } catch (const std::exception& error) {
        report["passed"] = false; report["state"] = "FAIL_OR_INCOMPLETE"; report["error"] = error.what();
        if (receipt != INVALID_HANDLE_VALUE) { try { writeFresh(receipt, report); } catch (...) {} CloseHandle(receipt); }
        std::cerr << "GEMM candidate calibration FAIL_OR_INCOMPLETE: " << error.what() << "\n";
        return 1;
    }
}
