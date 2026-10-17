// New code, MPL-2.0. Opt-in diagnostic only (inference --head-row-audit; head_audit.h). Copies count
// raw words of one packed weight row to output words [outputFirst, outputFirst+count). No float
// reinterpretation, so BF16 bit patterns, NaN payloads and denormals stay exact.
ByteAddressBuffer weight : register(t0);
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) {
    uint firstWord, count, outputFirst, reserved0;
    uint reserved1, reserved2, reserved3, reserved4;
};
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < count) output.Store((outputFirst + i) * 4, weight.Load((firstWord + i) * 4));
}
