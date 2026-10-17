ByteAddressBuffer k:register(t0);ByteAddressBuffer v:register(t1);RWByteAddressBuffer keys:register(u0);RWByteAddressBuffer values:register(u1);
cbuffer Params:register(b0){uint rows,width,offset,stride,base,count,mode,pad;}
[numthreads(128,1,1)]void main(uint3 t:SV_DispatchThreadID){uint i=t.x;if(i<rows*width){keys.Store((base*width+i)*4,asuint(asfloat(k.Load((i)*4))));values.Store((base*width+i)*4,asuint(asfloat(v.Load((i)*4))));}}
