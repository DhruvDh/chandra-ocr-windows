// New ChandraNative code, MPL-2.0. Public, finite B1 GEMV candidate calibration (Windows entry).
// No model/private assets. Root owns the external Job, CPU affinity/priority, process commit
// limit, physical device admission, build and native execution. Plan: gemv_candidate_calibration.h.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "gemv_candidate_calibration.h"
#include <windows.h>
#include <filesystem>
#include <iostream>
#include <map>

using namespace chandra::gemv_calibration;
namespace {
std::string ascii(const std::wstring& text) {
    std::string result;
    for (wchar_t c : text) { require(c > 0 && c < 128, "ASCII PCI and LUID required"); result.push_back(char(c)); }
    return result;
}
void writeFresh(HANDLE file, const Json& report) {
    std::string text = report.dump(2) + "\n";
    require(text.size() <= 8 * 1024 * 1024, "Complete calibration receipt exceeded 8 MiB");
    LARGE_INTEGER first{};
    require(SetFilePointerEx(file, first, nullptr, FILE_BEGIN), "Receipt seek failed");
    DWORD written = 0;
    require(WriteFile(file, text.data(), DWORD(text.size()), &written, nullptr) && written == text.size(), "Receipt write failed");
    require(SetEndOfFile(file) && FlushFileBuffers(file), "Receipt truncate/flush failed");
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    Json report = initialReport();
    HANDLE receipt = INVALID_HANDLE_VALUE;
    try {
        bool execute = false;
        std::map<std::wstring, std::wstring> args;
        for (int i = 1; i < argc; ++i) {
            std::wstring name = argv[i];
            if (name == L"--execute") { require(!execute, "Duplicate execute flag"); execute = true; }
            else {
                require(name == L"--shader-root" || name == L"--pci" || name == L"--luid" || name == L"--output" || name == L"--variant",
                        "Unknown calibration option");
                require(i + 1 < argc, "Option value absent");
                require(args.emplace(name, argv[++i]).second, "Duplicate calibration option");
            }
        }
        // The process-scoped selector was captured before main; an invalid value throws here,
        // before any receipt, device or buffer exists.
        std::string selection = chandra::dc::experimental::gemvB1Selection();
        if (!execute) {
            std::cout << Json{{"state", "INACTIVE"}, {"device_created", false}, {"model_loaded", false},
                              {"gemv_b1_selection", selection}}.dump() << "\n";
            return 0;
        }
        const Shape& shape = orderedRoute(selection);
        // Optional confirmation: a stated --variant must equal the selector captured at process start.
        const bool confirmed = args.count(L"--variant") != 0;
        if (confirmed) require(ascii(args.at(L"--variant")) == selection, "--variant differs from CHANDRA_EXPERIMENTAL_GEMV_B1");
        for (const auto* name : {L"--shader-root", L"--pci", L"--luid", L"--output"})
            require(args.count(name) && !args.at(name).empty(), "Explicit shader root, PCI, LUID and fresh output required");
        std::filesystem::path root(args.at(L"--shader-root"));
        require(root.is_absolute() && std::filesystem::is_directory(root) &&
                std::filesystem::is_regular_file(root / std::filesystem::u8path(shape.shader)),
                "Absolute existing shader root and candidate shader required");
        std::string pci = ascii(args.at(L"--pci")), luid = ascii(args.at(L"--luid"));
        std::filesystem::path output(args.at(L"--output"));
        require(output.is_absolute() && std::filesystem::is_directory(output.parent_path()), "Absolute fresh task-owned result path required");
        receipt = CreateFileW(output.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        require(receipt != INVALID_HANDLE_VALUE, "Fresh task-owned calibration receipt required");
        report["gemv_b1_selection"] = selection; report["variant_confirmed_by_argument"] = confirmed;
        describeRoute(report, shape);
        report["shader_root"] = root.u8string(); report["requested_pci"] = pci; report["requested_luid"] = luid;
        writeFresh(receipt, report);
        Deadline deadline;
        {
            Device device(root.wstring(), pci, luid);
            report["native_executed"] = true;
            report["device_identity"] = Json::parse(device.identityJson());
            report["memory_before"] = Json::parse(device.memoryJson());
            run(device, report, deadline, shape);
            device.drain(); deadline.check();
            require(device.trackedBufferBytes() == 0, "Completed calibration left owned buffers live");
            report["memory_after"] = Json::parse(device.memoryJson());
        }
        report["elapsed_seconds"] = deadline.seconds();
        report["state"] = "GEMV_B1_CANDIDATE_CALIBRATION_PASS";
        writeFresh(receipt, report); CloseHandle(receipt); receipt = INVALID_HANDLE_VALUE;
        std::cout << "{\"state\":\"GEMV_B1_CANDIDATE_CALIBRATION_PASS\",\"passed\":true,\"full_model_accepted\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        report["passed"] = false; report["state"] = "FAIL_OR_INCOMPLETE"; report["error"] = error.what();
        if (receipt != INVALID_HANDLE_VALUE) { try { writeFresh(receipt, report); } catch (...) {} CloseHandle(receipt); }
        std::cerr << "GEMV candidate calibration FAIL_OR_INCOMPLETE: " << error.what() << "\n";
        return 1;
    }
}
