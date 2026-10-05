ByteAddressBuffer scores:register(t0);ByteAddressBuffer v:register(t1);RWByteAddressBuffer y:register(u0);
cbuffer Params:register(b0){uint rows,width,offset,stride,base,count,mode,pad;}
float bf(float x){uint u=asuint(x);if((u&0x7f800000)==0x7f800000&&(u&0x007fffff)!=0)return asfloat(u|0x00400000);return asfloat((u+0x7fff+((u>>16)&1))&0xffff0000);}
float wt(ByteAddressBuffer w,uint i){return asfloat(((w.Load((i>>1)*4)>>((i&1)*16))&65535)<<16);}
[numthreads(256,1,1)]void main(uint3 group:SV_GroupID,uint d:SV_GroupIndex){uint h=group.x,r=group.y;float sum=0;for(uint k=base;k<min(base+128,count);k++)sum+=asfloat(scores.Load(((r*16+h)*count+k)*4))*asfloat(v.Load(((k*4+h/4)*256+d)*4));y.Store((((offset*rows+r)*16+h)*256+d)*4,asuint(sum));}
