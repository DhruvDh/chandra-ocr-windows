ByteAddressBuffer x:register(t0);RWByteAddressBuffer y:register(u0);
cbuffer Params:register(b0){uint rows,width,offset,stride,base,count,mode,pad;}
[numthreads(128,1,1)]void main(uint3 t:SV_DispatchThreadID){uint i=t.x;if(i<rows*width)y.Store((base*width+i)*4,asuint(asfloat(x.Load((i)*4))));}
