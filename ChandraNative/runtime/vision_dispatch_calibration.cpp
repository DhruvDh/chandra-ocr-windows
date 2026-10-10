// New ChandraNative code, MPL-2.0. Public, finite vision attention dispatch calibration (Windows entry).
// No model/private assets. Root owns the external Job, CPU affinity/priority, process commit limit,
// physical device admission, build and native execution. Plan and checks: vision_dispatch_calibration.h.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "vision_dispatch_calibration.h"
#include <windows.h>
#include <iostream>

using namespace chandra::vision_calibration;
namespace {
// Rewrites the receipt with bounded text; false when only the RECEIPT_OVERSIZE summary fitted.
bool writeFresh(HANDLE file, const Json& report) {
    const BoundedReceipt bounded = boundedReceipt(report); const std::string& text = bounded.text;
    LARGE_INTEGER first{};
    require(SetFilePointerEx(file, first, nullptr, FILE_BEGIN), "Receipt seek failed");
    DWORD written = 0;
    require(WriteFile(file, text.data(), DWORD(text.size()), &written, nullptr) && written == text.size(), "Receipt write failed");
    require(SetEndOfFile(file) && FlushFileBuffers(file), "Receipt truncate/flush failed");
    return bounded.complete;
}
// Every initial, progress and final receipt must be complete; detail lost to the cap stops further work.
void writeComplete(HANDLE file, const Json& report) {
    require(writeFresh(file, report), "Receipt exceeded the 8 MiB cap; detail lost; no further dispatch");
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    Json report = initialReport();
    HANDLE receipt = INVALID_HANDLE_VALUE;
    try {
        const Arguments args = parseArguments(std::vector<std::wstring>(argv + 1, argv + argc));
        if (!args.execute) {
            std::cout << Json{{"state", "INACTIVE"}, {"schema", schema}, {"device_created", false}, {"model_loaded", false},
                              {"native_executed", false},
                              {"execution_requires", "--execute, absolute --shader-root, lowercase --pci bb:dd.f, --luid hhhhhhhh:llllllll, "
                                                     "absolute fresh --output, and the root-owned external Job"},
                              {"plan", planJson()}}.dump() << "\n";
            return 0;
        }
        std::filesystem::path root(args.values.at(L"--shader-root"));
        require(root.is_absolute() && std::filesystem::is_directory(root), "Absolute existing shader root required");
        const std::string pci = pinnedPci(args.values.at(L"--pci")), luid = pinnedLuid(args.values.at(L"--luid"));
        std::filesystem::path output(args.values.at(L"--output"));
        require(output.is_absolute() && std::filesystem::is_directory(output.parent_path()), "Absolute fresh task-owned result path required");
        receipt = CreateFileW(output.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        require(receipt != INVALID_HANDLE_VALUE, "Fresh task-owned calibration receipt required");
        report["shader_root"] = root.u8string(); report["requested_pci"] = pci; report["requested_luid"] = luid;
        report["device_created"] = false;
        writeComplete(receipt, report);
        Deadline deadline;
        report["frozen_shader_verification"] = verifyFrozenShaders(root);
        const Persist persist = [&] { writeComplete(receipt, report); };
        persist();
        {
            Device device(root.wstring(), pci, luid);
            report["native_executed"] = true; report["device_created"] = true;
            report["device_identity"] = Json::parse(device.identityJson());
            report["memory_before"] = Json::parse(device.memoryJson());
            persist();
            try {
                run(device, report, deadline, persist);
            } catch (const std::exception& error) {
                // Persist before any further device call or the Device destructor, either of which can block.
                // Guarded so the bounded failure drain is always attempted; an oversized receipt still leaves
                // its bounded summary on disk before persist throws.
                try {
                    report["passed"] = false; report["state"] = "FAIL_OR_INCOMPLETE"; report["error"] = boundedError(error.what());
                    report["retirement_required"] = true; persist();
                } catch (...) {}
                try {
                    device.drain(); // One bounded completion wait; nothing is resubmitted.
                    report["failure_drain"] = {{"completed", true}, {"tracked_after", device.trackedBufferBytes()}};
                } catch (const std::exception& drainError) {
                    report["failure_drain"] = {{"completed", false}, {"error", boundedError(drainError.what())}};
                }
                try { persist(); } catch (...) {}
                throw;
            }
            device.drain(); deadline.check();
            require(device.trackedBufferBytes() == 0, "Completed calibration left owned buffers live");
            report["memory_after"] = Json::parse(device.memoryJson());
        }
        report["elapsed_seconds"] = deadline.seconds();
        report["state"] = "VISION_DISPATCH_CALIBRATION_PASS";
        writeComplete(receipt, report); CloseHandle(receipt); receipt = INVALID_HANDLE_VALUE;
        std::cout << "{\"state\":\"VISION_DISPATCH_CALIBRATION_PASS\",\"passed\":true,\"full_model_accepted\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        const std::string message = boundedError(error.what());
        report["passed"] = false; report["state"] = "FAIL_OR_INCOMPLETE"; report["error"] = message;
        report["retirement_required"] = report.at("native_executed") == true;
        if (receipt != INVALID_HANDLE_VALUE) { try { writeFresh(receipt, report); } catch (...) {} CloseHandle(receipt); }
        std::cerr << "Vision dispatch calibration FAIL_OR_INCOMPLETE: " << message << "\n";
        return 1;
    }
}
