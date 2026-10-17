// New ChandraNative code, MPL-2.0. Reference: Transformers torch_recurrent_gated_delta_rule.
// Q/K are already L2-normalized; Q additionally scaled by 1/sqrt(K).
// G is log decay, beta is the post-sigmoid update factor. Inputs are [T,H,K/V].
// Each thread owns a value column: no cross-thread state dependency, including prefill.
ByteAddressBuffer q : register(t0);
ByteAddressBuffer k : register(t1);
ByteAddressBuffer v : register(t2);
ByteAddressBuffer g : register(t3);
ByteAddressBuffer beta : register(t4);
RWByteAddressBuffer state : register(u0);
RWByteAddressBuffer output : register(u1);
cbuffer Params : register(b0) { uint heads; uint keyDim; uint valueDim; uint tokens; uint start; uint3 reserved; };
float bf(float x){uint u=asuint(x);if((u&0x7f800000)==0x7f800000&&(u&0x007fffff)!=0)return asfloat(u|0x00400000);return asfloat((u+0x7fff+((u>>16)&1))&0xffff0000);}
float wt(ByteAddressBuffer w,uint i){return asfloat(((w.Load((i>>1)*4)>>((i&1)*16))&65535)<<16);}
[numthreads(128,1,1)]
void main(uint3 group : SV_GroupID, uint col : SV_GroupIndex) {
    uint h=group.x;
    if (h>=heads || col>=valueDim) return;
    for (uint t=start; t<start+tokens; ++t) {
        uint kb=(t*heads+h)*keyDim, vb=(t*heads+h)*valueDim;
        uint sb=h*keyDim*valueDim+col;
        float decay=exp(asfloat(g.Load((t*heads+h)*4))), memory=0;
        for (uint i=0; i<keyDim; ++i) {
            float s=asfloat(state.Load((sb+i*valueDim)*4))*decay;
            state.Store((sb+i*valueDim)*4,asuint(s));
            memory += s*asfloat(k.Load((kb+i)*4));
        }
        float change=(asfloat(v.Load((vb+col)*4))-memory)*asfloat(beta.Load((t*heads+h)*4)), result=0;
        for (uint j=0; j<keyDim; ++j) {
            float s=asfloat(state.Load((sb+j*valueDim)*4))+asfloat(k.Load((kb+j)*4))*change;
            state.Store((sb+j*valueDim)*4,asuint(s));
            result += s*asfloat(q.Load((kb+j)*4));
        }
        output.Store((vb+col)*4,asuint(bf(result)));
    }
}
