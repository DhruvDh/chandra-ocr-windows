// New code, MPL-2.0. <=128 keys per dispatch; accum remains FP32 between tiles.
ByteAddressBuffer scores:register(t0); ByteAddressBuffer v:register(t1);
RWByteAddressBuffer partials:register(u0);
cbuffer Params:register(b0) {
    uint sequenceStart,sequenceLength,queryStart,queryCount;
    uint keyStart,keyCount,tileIndex,tileCount;
    uint width,heads,headDim,pad;
}
[numthreads(64,1,1)] void main(uint3 group:SV_GroupID,uint c:SV_GroupIndex) {
    uint qh=group.y,h=qh%16,qr=qh/16,tile=group.x;
    precise float sum=0.0;
    for(uint j=0;j<min(128,sequenceLength-tile*128);j++) {
        uint kr=tile*128+j;
        precise float probability=asfloat(scores.Load((qh*sequenceLength+kr)*4));
        sum+=probability*asfloat(v.Load(((sequenceStart+kr)*1024+h*64+c)*4));
    }
    uint index=((tile*queryCount+qr)*1024+h*64+c)*4;
    partials.Store(index,asuint(sum));
}
