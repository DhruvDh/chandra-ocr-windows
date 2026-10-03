#include "oracle.h"
#include <iostream>
int main() {
    chandra::Fixture f{2,1,1,1,{1,1},{1,1},{2,4},{0,0},{0.5f,0.5f}};
    auto r=chandra::recurrent<double>(f);
    if(r.output[0]!=1 || r.output[1]!=2.5 || r.state[0]!=2.5) return 1;
    auto p=chandra::fixture(1000,2,8,8);
    auto a=chandra::recurrent<float>(p);auto b=chandra::recurrent<double>(p);
    auto e=chandra::error(a.state,b.state);
    if(e.maximum>1e-5) return 2;
    std::cout<<"scalar_equation_pass=true fp32_double_state_max="<<e.maximum<<"\n";
}
