// New ChandraNative code, MPL-2.0. Portable CPU fixture for external reference checks.
#include "oracle.h"
#include <fstream>
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    auto f=chandra::fixture(1000,2,8,8);auto result=chandra::recurrent<float>(f);
    std::ofstream out(argv[1],std::ios::binary);uint32_t shape[]={f.tokens,f.heads,f.keys,f.values};out.write(reinterpret_cast<char*>(shape),sizeof(shape));
    for(const auto* values:{&f.q,&f.k,&f.v,&f.g,&f.beta,&result.state,&result.output})out.write(reinterpret_cast<const char*>(values->data()),values->size()*sizeof(float));
    return out.good()?0:1;
}
