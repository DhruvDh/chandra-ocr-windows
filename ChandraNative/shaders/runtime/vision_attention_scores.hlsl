// New code, MPL-2.0. One <=32-query, <=128-key tile; every dot has exactly 64 FP32 terms.
ByteAddressBuffer q:register(t0); ByteAddressBuffer k:register(t1);
RWByteAddressBuffer scores:register(u0);
cbuffer Params:register(b0) {
    uint sequenceStart,sequenceLength,queryStart,queryCount;
    uint keyStart,keyCount,tileIndex,tileCount;
    uint width,heads,headDim,pad;
}
groupshared float queryVector[64];
[numthreads(128,1,1)] void main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    uint qh=group.y,qr=qh/16,h=qh%16,key=group.x*128+lane;
    if(lane<64)queryVector[lane]=asfloat(q.Load(((sequenceStart+queryStart+qr)*1024+h*64+lane)*4));
    GroupMemoryBarrierWithGroupSync();
    precise float sum=0.0;
    if(key<sequenceLength) {
        uint kr=sequenceStart+key;
        for(uint c=0;c<64;c++)sum+=queryVector[c]*asfloat(k.Load((kr*1024+h*64+c)*4));
    }
    precise float value=sum*0.125;
    if(key<sequenceLength)scores.Store((qh*sequenceLength+key)*4,asuint(value));
}
