// New code, MPL-2.0. Test-only: writes the fake-device call trace of the fixed scenario through the
// predecessor public API. tests/native compiles it against predecessor and current graph sources to
// show the disabled diagnostic path keeps identical allocation, dispatch, drain and release order.
#include "diagnostics_fake_device.h"
#include <fstream>
#include <iostream>

int main(int argc,char** argv) {
    using namespace chandra::dc;
    if(argc!=2) { std::cerr<<"usage: diagnostics-trace OUTPUT_TRACE\n";return 2; }
    try {
        Device device(L"","fake","fake");ModelWeights weights(device,L"");FakeScenario scenario;PlainHooks hooks;
        runFakeScenario(device,weights,scenario,hooks);
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
    if(fakeTrace().live!=0) { std::cerr<<"fake buffers leaked\n";return 1; }
    std::ofstream out(argv[1],std::ios::binary);
    for(const auto& line:fakeTrace().lines)out<<line<<'\n';
    return out?0:1;
}
