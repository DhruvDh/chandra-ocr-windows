// New code, MPL-2.0. Full-frame stable FP32 softmax, <=256 iterations/lane for admitted frames.
RWByteAddressBuffer scores:register(u0);
cbuffer Params:register(b0) { uint queryHeads,sequenceLength,pad0,pad1; }
groupshared float r[128];
[numthreads(128,1,1)] void main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    float maximum=-3.402823466e38;
    for(uint maxKey=lane;maxKey<sequenceLength;maxKey+=128)maximum=max(maximum,asfloat(scores.Load((group.x*sequenceLength+maxKey)*4)));
    r[lane]=maximum;GroupMemoryBarrierWithGroupSync();
    for(uint maxStep=64;maxStep;maxStep>>=1) { if(lane<maxStep)r[lane]=max(r[lane],r[lane+maxStep]);GroupMemoryBarrierWithGroupSync(); }
    maximum=r[0];GroupMemoryBarrierWithGroupSync();precise float sum=0.0;
    for(uint expKey=lane;expKey<sequenceLength;expKey+=128) {
        uint index=(group.x*sequenceLength+expKey)*4;
        precise float e=exp(asfloat(scores.Load(index))-maximum);scores.Store(index,asuint(e));sum+=e;
    }
    r[lane]=sum;GroupMemoryBarrierWithGroupSync();
    for(uint sumStep=64;sumStep;sumStep>>=1) { if(lane<sumStep)r[lane]+=r[lane+sumStep];GroupMemoryBarrierWithGroupSync(); }
    float denominator=r[0];
    for(uint probabilityKey=lane;probabilityKey<sequenceLength;probabilityKey+=128) {
        uint index=(group.x*sequenceLength+probabilityKey)*4;
        scores.Store(index,asuint(asfloat(scores.Load(index))/denominator));
    }
}
