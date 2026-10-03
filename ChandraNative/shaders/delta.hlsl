// New ChandraNative code, MPL-2.0. Reference: Transformers torch_recurrent_gated_delta_rule.
// Q/K are already L2-normalized; Q additionally scaled by 1/sqrt(K).
// G is log decay, beta is the post-sigmoid update factor. Inputs are [T,H,K/V].
// Each thread owns a value column: no cross-thread state dependency, including prefill.
Buffer<float> q : register(t0);
Buffer<float> k : register(t1);
Buffer<float> v : register(t2);
Buffer<float> g : register(t3);
Buffer<float> beta : register(t4);
RWBuffer<float> state : register(u0);
RWBuffer<float> output : register(u1);
cbuffer Params : register(b0) { uint heads; uint keyDim; uint valueDim; uint tokens; uint start; uint3 reserved; };
[numthreads(128,1,1)]
void main(uint3 group : SV_GroupID, uint col : SV_GroupIndex) {
    uint h=group.x;
    if (h>=heads || col>=valueDim) return;
    for (uint t=start; t<start+tokens; ++t) {
        uint kb=(t*heads+h)*keyDim, vb=(t*heads+h)*valueDim;
        uint sb=h*keyDim*valueDim+col;
        float decay=exp(g[t*heads+h]), memory=0;
        for (uint i=0; i<keyDim; ++i) {
            float s=state[sb+i*valueDim]*decay;
            state[sb+i*valueDim]=s;
            memory += s*k[kb+i];
        }
        float change=(v[vb+col]-memory)*beta[t*heads+h], result=0;
        for (uint j=0; j<keyDim; ++j) {
            float s=state[sb+j*valueDim]+k[kb+j]*change;
            state[sb+j*valueDim]=s;
            result += s*q[kb+j];
        }
        output[vb+col]=result;
    }
}
