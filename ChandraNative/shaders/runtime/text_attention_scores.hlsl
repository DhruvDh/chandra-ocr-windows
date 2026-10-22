ByteAddressBuffer q:register(t0);ByteAddressBuffer k:register(t1);RWByteAddressBuffer scores:register(u0);
cbuffer Params:register(b0){uint rows,width,offset,stride,base,count,mode,pad;}
float bf(float x){uint u=asuint(x);if((u&0x7f800000)==0x7f800000&&(u&0x007fffff)!=0)return asfloat(u|0x00400000);return asfloat((u+0x7fff+((u>>16)&1))&0xffff0000);}
float wt(ByteAddressBuffer w,uint i){return asfloat(((w.Load((i>>1)*4)>>((i&1)*16))&65535)<<16);}
[numthreads(128,1,1)]void main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex){uint key=group.x*128+lane,h=group.y%16,r=group.y/16;if(key>=count)return;float value=-3.402823466e38;if(key<=base+r){float sum=0;for(uint d=0;d<256;d++)sum+=asfloat(q.Load(((r*16+h)*256+d)*4))*asfloat(k.Load(((key*4+h/4)*256+d)*4));value=bf(sum)*0.0625;}scores.Store(((r*16+h)*count+key)*4,asuint(value));}
