// New code, MPL-2.0. Exact packed BF16 expansion, no numeric conversion to FP16.
ByteAddressBuffer tokenIds : register(t0);
ByteAddressBuffer weight : register(t1);
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) {
    uint width, rows, firstWeightRow, shardRows;
    uint tokenFirst, flags, reserved0, reserved1;
};
[numthreads(256,1,1)]
void main(uint3 group : SV_GroupID, uint3 tid : SV_DispatchThreadID) {
    uint column = tid.x;
    uint localToken = group.y;
    if (column >= width || localToken >= rows) return;
    uint row = tokenFirst + localToken;
    uint token = tokenIds.Load(row * 4);
    if (token < firstWeightRow || token - firstWeightRow >= shardRows) return;
    uint index = (token - firstWeightRow) * width + column;
    uint bits;
    if ((flags & 1) != 0) {
        uint word = weight.Load((index >> 1) * 4);
        bits = ((word >> ((index & 1) * 16)) & 0xffff) << 16;
    } else bits = weight.Load(index * 4);
    output.Store((row * width + column) * 4, bits);
}
