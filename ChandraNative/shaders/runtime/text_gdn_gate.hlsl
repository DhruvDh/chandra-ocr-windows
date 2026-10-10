ByteAddressBuffer x:register(t0);ByteAddressBuffer z:register(t1);ByteAddressBuffer w:register(t2);RWByteAddressBuffer y:register(u0);
cbuffer Params:register(b0){uint rows,width,offset,stride,base,count,mode,pad;}
float bf(float x){uint u=asuint(x);if((u&0x7f800000)==0x7f800000&&(u&0x007fffff)!=0)return asfloat(u|0x00400000);return asfloat((u+0x7fff+((u>>16)&1))&0xffff0000);}
float wt(ByteAddressBuffer w,uint i){return asfloat(((w.Load((i>>1)*4)>>((i&1)*16))&65535)<<16);}
groupshared float sum[128];[numthreads(128,1,1)]void main(uint3 group:SV_GroupID,uint d:SV_GroupIndex){uint i=(group.y*32+group.x)*128+d;float value=asfloat(x.Load((i)*4));sum[d]=value*value;GroupMemoryBarrierWithGroupSync();for(uint step=64;step>0;step>>=1){if(d<step)sum[d]+=sum[d+step];GroupMemoryBarrierWithGroupSync();}float norm=bf(value*rsqrt(sum[0]/128+1e-6));float weighted=bf(norm*wt(w,d));float gate=asfloat(z.Load((i)*4))/(1+exp(-asfloat(z.Load((i)*4))));y.Store((i)*4,asuint(bf(weighted*gate)));}
