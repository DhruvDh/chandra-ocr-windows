// New code, MPL-2.0. All Device views are RAW, including expanded float activations.
float vision_bf16(float x) {
    uint u=asuint(x);
    if ((u&0x7f800000)==0x7f800000 && (u&0x007fffff)!=0) return asfloat((u&0xffff0000)|0x00400000);
    return asfloat((u+0x7fff+((u>>16)&1))&0xffff0000);
}
float vision_weight(ByteAddressBuffer w,uint i) {
    return asfloat(((w.Load((i>>1)*4)>>((i&1)*16))&65535)<<16);
}
