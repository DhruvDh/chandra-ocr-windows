// New ChandraNative code, MPL-2.0. Portable CPU format/reference checker.
#include "fixture.h"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=2)return 2;
        auto fixture=chandra::loadFixture(std::filesystem::u8path(argv[1]));
        auto fp=chandra::recurrent<float>(fixture.inputs);auto dp=chandra::recurrent<double>(fixture.inputs);
        auto fs=chandra::compare(fp.state,fixture.expectedState,2e-5,0),fo=chandra::compare(fp.output,fixture.expectedOutput,2e-5,0),ds=chandra::compare(dp.state,fixture.expectedState,2e-5,0),dout=chandra::compare(dp.output,fixture.expectedOutput,2e-5,0);
        bool passed=fs.passed && fo.passed && ds.passed && dout.passed;
        std::cout<<chandra::Json{{"test","external_fixture_cpu_reference"},{"passed",passed},{"metadata_sha256",fixture.metadataHash},{"payload_sha256",fixture.payloadHash},{"tokens",fixture.inputs.tokens},{"state_fp32_max_abs",fs.absolute.maximum},{"output_fp32_max_abs",fo.absolute.maximum},{"state_double_max_abs",ds.absolute.maximum},{"output_double_max_abs",dout.absolute.maximum}}.dump()<<"\n";
        return passed?0:1;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
