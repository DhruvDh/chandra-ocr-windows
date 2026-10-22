// New ChandraNative code, MPL-2.0. Test-only main for running the unchanged Windows entry
// (vision_dispatch_calibration.cpp, wmain) on the CPU fake Device. Fault selection comes from the
// environment so each case runs in its own process: VDC_FAKE_FAULT, VDC_FAKE_SLOW_DISPATCH (1-based
// dispatch serial reported at exactly 100 ms), VDC_FAKE_PROFILE_FAULT (including giantError and
// giantMetadata) and VDC_FAKE_PROFILE_FAULT_AT.
#include "vision_dispatch_calibration_fake_device.h"
#include <cstdlib>
#include <iostream>

int wmain(int argc, wchar_t** argv);

namespace {
std::string environment(const char* name) { const char* value = std::getenv(name); return value ? value : ""; }
uint64_t number(const char* name) { const std::string text = environment(name); return text.empty() ? 0 : std::stoull(text); }
}

int main(int argc, char** argv) {
    auto& config = vdc_fake::config();
    const std::map<std::string, vdc_fake::Fault> faults = {
        {"", vdc_fake::Fault::none}, {"scoresIgnoreSequenceStart", vdc_fake::Fault::scoresIgnoreSequenceStart},
        {"softmaxDropLastLane", vdc_fake::Fault::softmaxDropLastLane}, {"valuesSkipLastKey", vdc_fake::Fault::valuesSkipLastKey},
        {"reduceSkipLastTile", vdc_fake::Fault::reduceSkipLastTile}, {"reduceIgnoreRowFirst", vdc_fake::Fault::reduceIgnoreRowFirst}};
    const std::map<std::string, vdc_fake::ProfileFault> profileFaults = {
        {"", vdc_fake::ProfileFault::none}, {"disjointThrows", vdc_fake::ProfileFault::disjointThrows},
        {"disjointReported", vdc_fake::ProfileFault::disjointReported}, {"missing", vdc_fake::ProfileFault::missing},
        {"giantError", vdc_fake::ProfileFault::giantError}, {"giantMetadata", vdc_fake::ProfileFault::giantMetadata}};
    config.fault = faults.at(environment("VDC_FAKE_FAULT"));
    config.profileFault = profileFaults.at(environment("VDC_FAKE_PROFILE_FAULT"));
    config.profileFaultDispatch = number("VDC_FAKE_PROFILE_FAULT_AT");
    const uint64_t slow = number("VDC_FAKE_SLOW_DISPATCH");
    config.gpuMilliseconds = [slow](const std::string&, uint64_t id) { return id == slow ? 100.0 : 0.25; };
    std::vector<std::wstring> wide;
    for (int i = 0; i < argc; ++i) {
        std::wstring text;
        for (const char* p = argv[i]; *p; ++p) text.push_back(wchar_t(static_cast<unsigned char>(*p)));
        wide.push_back(text);
    }
    std::vector<wchar_t*> pointers;
    for (auto& text : wide) pointers.push_back(text.data());
    const int result = wmain(argc, pointers.data());
    std::cerr << "fake dispatches " << vdc_fake::dispatchSerial() << "\n";
    return result;
}
