// New ChandraNative code, MPL-2.0.
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
namespace chandra {
struct Fixture {
    unsigned tokens,heads,keys,values;
    std::vector<float> q,k,v,g,beta;
};
inline float sample(unsigned i, unsigned salt) {
    uint32_t x=i*747796405u+salt*2891336453u+277803737u;
    x=((x>>((x>>28)+4))^x)*277803737u; x=(x>>22)^x;
    return float(int32_t(x&65535)-32768)/32768.0f;
}
inline Fixture fixture(unsigned t,unsigned h,unsigned k,unsigned v) {
    Fixture f{t,h,k,v,{},{},{},{},{}}; f.q.resize(size_t(t)*h*k); f.k.resize(f.q.size());
    f.v.resize(size_t(t)*h*v); f.g.resize(size_t(t)*h); f.beta.resize(f.g.size());
    for(size_t n=0;n<size_t(t)*h;++n) {
        double qsum=0,ksum=0;
        for(unsigned i=0;i<k;++i) {size_t p=n*k+i;f.q[p]=sample(unsigned(p),1);f.k[p]=sample(unsigned(p),2);qsum+=double(f.q[p])*f.q[p];ksum+=double(f.k[p])*f.k[p];}
        for(unsigned i=0;i<k;++i) {size_t p=n*k+i;f.q[p]=float(f.q[p]/std::sqrt(qsum+1e-6)/std::sqrt(double(k)));f.k[p]=float(f.k[p]/std::sqrt(ksum+1e-6));}
        f.g[n]=-0.005f-0.02f*(sample(unsigned(n),3)+1);f.beta[n]=0.1f+0.2f*(sample(unsigned(n),4)+1);
    }
    for(size_t i=0;i<f.v.size();++i) f.v[i]=sample(unsigned(i),5)*0.5f;
    return f;
}
template<class Real> struct Result {std::vector<Real> state,output;};
template<class Real> inline Result<Real> recurrent(const Fixture& f) {
    Result<Real> r; r.state.resize(size_t(f.heads)*f.keys*f.values);r.output.resize(f.v.size());
    for(unsigned t=0;t<f.tokens;++t) for(unsigned h=0;h<f.heads;++h) {
        size_t kb=(size_t(t)*f.heads+h)*f.keys,vb=(size_t(t)*f.heads+h)*f.values,sb=size_t(h)*f.keys*f.values;
        Real decay=std::exp(Real(f.g[size_t(t)*f.heads+h]));
        for(unsigned j=0;j<f.values;++j) {
            Real memory=0;
            for(unsigned i=0;i<f.keys;++i) {Real& s=r.state[sb+size_t(i)*f.values+j];s*=decay;memory+=s*Real(f.k[kb+i]);}
            Real change=(Real(f.v[vb+j])-memory)*Real(f.beta[size_t(t)*f.heads+h]),y=0;
            for(unsigned i=0;i<f.keys;++i) {Real& s=r.state[sb+size_t(i)*f.values+j];s+=Real(f.k[kb+i])*change;y+=s*Real(f.q[kb+i]);}
            r.output[vb+j]=y;
        }
    }
    return r;
}
struct Error {double maximum=0,rms=0;};
template<class A,class B> inline Error error(const std::vector<A>& a,const std::vector<B>& b) {
    if(a.size()!=b.size()) throw std::runtime_error("size mismatch");
    Error e;
    for(size_t i=0;i<a.size();++i) {double d=std::abs(double(a[i])-double(b[i]));if(!std::isfinite(d))throw std::runtime_error("nonfinite result");e.maximum=std::max(e.maximum,d);e.rms+=d*d;}
    e.rms=std::sqrt(e.rms/std::max(size_t(1),a.size()));return e;
}
}
