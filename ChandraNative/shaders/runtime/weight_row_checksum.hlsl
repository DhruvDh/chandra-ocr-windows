// New code, MPL-2.0. Opt-in diagnostic only (inference --head-row-audit; head_audit.h). One thread per
// packed weight row: checksum = sum over word i of (2i+1)*word with uint32 wrap, ascending i. Integer
// loads and arithmetic only; no float conversion. Host bounds: rows <= 1024 per dispatch, wordsPerRow
// <= 4608, every (firstRow+row)*wordsPerRow+i word inside the shard's uint32 byte range.
ByteAddressBuffer weight : register(t0);
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) {
    uint wordsPerRow, rows, firstRow, outputFirst;
    uint reserved0, reserved1, reserved2, reserved3;
};
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint row = tid.x;
    if (row >= rows) return;
    uint base = (firstRow + row) * wordsPerRow;
    uint sum = 0;
    for (uint i = 0; i < wordsPerRow; ++i) sum += (2 * i + 1) * weight.Load((base + i) * 4);
    output.Store((outputFirst + row) * 4, sum);
}
