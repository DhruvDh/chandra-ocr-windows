// New code, MPL-2.0. BF16 boundaries remain visible although storage is FP32.
// Unary: 0 SiLU, 1 GELU(tanh), 2 sigmoid, 3 BF16 RNE, 4 GELU(erf).
// Binary: 0 add, 1 mul, 2 SiLU(a)*b, 3 GELU(tanh,a)*b, 4 GELU(erf,a)*b.
// Binary activation modes with BF16 output round activation to BF16 BEFORE mul,
// matching a BF16 activation followed by a separate BF16 tensor multiplication.
// round=false means FP32 activation/mul; use this for gate.float() SiLU paths.
ByteAddressBuffer firstInput : register(t0);
ByteAddressBuffer secondInput : register(t1);
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) {
    uint count, first, operation, flags;
    uint secondCount, reserved0, reserved1, reserved2;
};
float bf16_rne(float value) {
    uint bits = asuint(value);
    if ((bits & 0x7fffffff) > 0x7f800000) return asfloat((bits & 0xffff0000) | 0x00400000);
    return asfloat((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
}
float sigmoid_stable(float x) {
    precise float z = exp(-abs(x));
    return x >= 0.0 ? 1.0 / (1.0+z) : z / (1.0+z);
}
float silu_stable(float x) { return x * sigmoid_stable(x); }
float tanh_stable(float x) {
    precise float z = exp(-2.0*abs(x));
    precise float y = (1.0-z) / (1.0+z);
    return x < 0.0 ? -y : y;
}
float gelu_tanh(float x) {
    precise float x2 = x*x;
    precise float x3 = x2*x;
    precise float inner = x + 0.044715*x3;
    precise float t = tanh_stable(0.7978845608028654*inner);
    precise float halfX = 0.5*x;
    return halfX * (1.0+t);
}
// Rational erf/erfc coefficients from Cephes ndtr, Stephen L. Moshier;
// source: scipy v1.15.0 scipy/special/xsf/cephes/ndtr.h (BSD source distribution).
// FP32 Horner evaluation plus SM5 exp; NOT a correctly-rounded libm claim.
// No tanh substitution for the merger's default nn.GELU erf definition.
float erf_fp32(float x) {
    float a=abs(x); precise float result;
    if (a <= 1.0) {
        precise float z=a*a;
        precise float p=9.604973739870516;
        p=p*z+90.02601972038427; p=p*z+2232.005345946843;
        p=p*z+7003.325141128051; p=p*z+55592.3013010395;
        precise float q=z+33.56171416475031;
        q=q*z+521.3579497801527; q=q*z+4594.323829709801;
        q=q*z+22629.000061389095; q=q*z+49267.39426086359;
        result=a*p/q;
    } else if (a < 6.0) {
        precise float p=2.461969814735305e-10;
        p=p*a+0.5641895648310688; p=p*a+7.463210564422699;
        p=p*a+48.63719709856814; p=p*a+196.5208329560771;
        p=p*a+526.4451949954774; p=p*a+934.5285271719576;
        p=p*a+1027.551886895157; p=p*a+557.5353353693993;
        precise float q=a+13.228195115474499;
        q=q*a+86.70721408859897; q=q*a+354.9377788878199;
        q=q*a+975.7085017432055; q=q*a+1823.9091668790973;
        q=q*a+2246.3376081871097; q=q*a+1656.6630919416135;
        q=q*a+557.5353408177277;
        precise float tail=exp(-a*a)*p/q;
        result=1.0-tail;
    } else result=1.0; // The erf tail is below half an FP32 ulp here.
    return x < 0.0 ? -result : result;
}
float gelu_erf(float x) {
    precise float e=erf_fp32(x*0.7071067811865475);
    precise float halfX=0.5*x;
    return halfX*(1.0+e);
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    uint index=first+tid.x;
    float a=asfloat(firstInput.Load(index*4));
    precise float y;
    bool rounded=(flags & 1)!=0;
    if ((flags & 2)!=0) {
        if (operation==0) y=silu_stable(a);
        else if(operation==1) y=gelu_tanh(a);
        else if(operation==2) y=sigmoid_stable(a);
        else if(operation==3) y=bf16_rne(a);
        else y=gelu_erf(a);
    } else {
        float b=asfloat(secondInput.Load((index % secondCount)*4));
        if(operation==0) y=a+b;
        else if(operation==1) y=a*b;
        else {
            precise float activated=(operation==2) ? silu_stable(a) :
                                   ((operation==3) ? gelu_tanh(a) : gelu_erf(a));
            if(rounded) activated=bf16_rne(activated);
            y=activated*b;
        }
    }
    if(rounded) y=bf16_rne(y);
    output.Store(index*4,asuint(y));
}
