// New code, MPL-2.0. Exact image-row replacement; no numerical conversion.
// CPU validation supplies one ordered vision row for every image-token row.
ByteAddressBuffer textEmbeddings : register(t0);
ByteAddressBuffer visionEmbeddings : register(t1);
ByteAddressBuffer visionRowForToken : register(t2);
RWByteAddressBuffer mergedEmbeddings : register(u0);
cbuffer Params : register(b0) {
    uint tokens, width, visionRows, firstElement;
    uint count, reserved0, reserved1, reserved2;
};
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= count) return;
    uint element = firstElement + thread.x;
    if (element >= tokens * width) return;
    uint tokenRow = element / width;
    uint column = element % width;
    uint visionRow = visionRowForToken.Load(tokenRow * 4);
    uint bits;
    if (visionRow == 0xffffffffu) bits = textEmbeddings.Load(element * 4);
    else {
        // The host rejects invalid indices before dispatch. Keep an access
        // bound here too; a malformed map never reads beyond the vision SRV.
        if (visionRow >= visionRows) return;
        bits = visionEmbeddings.Load((visionRow * width + column) * 4);
    }
    mergedEmbeddings.Store(element * 4, bits);
}
