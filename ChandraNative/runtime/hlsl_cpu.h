// New code, MPL-2.0. Test-only CPU execution of the runtime HLSL compute shaders.
// tests/native/hlsl_cpu.py translates an unmodified runtime .hlsl file into a C++ struct that derives from
// Shader below; this header supplies the HLSL scalar/vector types, intrinsics, RAW buffer views and
// groupshared arrays. Each workgroup lane runs as a cooperative fiber; a lane executes
// until GroupMemoryBarrierWithGroupSync, so one barrier interval of each lane is atomic and the lane
// order inside an interval is a chosen legal schedule (ascending, descending or seeded shuffle).
// Independently of the schedule, every groupshared element and every UAV word is checked for conflicting
// accesses by different lanes/threads inside one barrier interval/dispatch (a data race under the HLSL
// memory model), reads before any write, out-of-bounds and misaligned addresses, and barriers reached by
// only some lanes. Arithmetic is host IEEE FP32 (no contraction); HLSL exp/sin/cos/rsqrt implementation
// error and driver compilation are NOT modelled, so this executes source logic and indexing, not the GPU.
#pragma once
#include <ucontext.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hlsl_cpu {
using uint=uint32_t;
struct uint3 { uint x=0,y=0,z=0; };
struct uint4 { uint x=0,y=0,z=0,w=0; };
enum class Order { Ascending, Descending, Shuffled };

struct Findings {
    uint64_t total=0; std::map<std::string,uint64_t> kinds; std::vector<std::string> first;
    void add(const std::string& kind,const std::string& detail) {
        ++total;++kinds[kind];if(first.size()<32)first.push_back(kind+": "+detail);
    }
    bool clean() const { return total==0; }
    uint64_t count(const std::string& kind) const { auto it=kinds.find(kind);return it==kinds.end()?0:it->second; }
};

// One Device buffer. Contents are allocated lazily so trace-only runs of production geometry stay small;
// a lazily allocated buffer is zero-filled (zeroed) or produced by its generator (synthetic weights).
struct Memory {
    std::string label; uint32_t words=0; bool allWritten=false, zeroed=false;
    std::vector<uint32_t> data; std::vector<uint8_t> written;
    std::function<void(std::vector<uint32_t>&)> generator;
    // UAV hazard state, allocated when first bound as a UAV. A word's writer/reader/access bits are valid only
    // while stamp equals the current dispatch serial, so no per-dispatch reset or allocation is needed.
    struct Free { void operator()(void* p) const { std::free(p); } };
    std::unique_ptr<uint32_t[],Free> stamp, writer, reader; std::unique_ptr<uint8_t[],Free> access; // access bit0 written, bit1 read, bit2 several readers
    void track() {
        if(stamp)return;
        stamp.reset(static_cast<uint32_t*>(std::calloc(words,4)));writer.reset(static_cast<uint32_t*>(std::malloc(size_t(words)*4)));
        reader.reset(static_cast<uint32_t*>(std::malloc(size_t(words)*4)));access.reset(static_cast<uint8_t*>(std::malloc(words)));
        if(!stamp||!writer||!reader||!access)throw std::bad_alloc();
    }
    uint8_t& accessed(uint32_t i,uint32_t dispatch) { if(stamp[i]!=dispatch) { stamp[i]=dispatch;access[i]=0; }return access[i]; }
    // Copies contents and initialization state; hazard state belongs to the original's dispatches only.
    Memory()=default;
    Memory(const Memory& o):label(o.label),words(o.words),allWritten(o.allWritten),zeroed(o.zeroed),data(o.data),written(o.written),generator(o.generator) {}
    Memory& operator=(const Memory& o) { if(this!=&o) { Memory copy(o);*this=std::move(copy); }return *this; }
    Memory(Memory&&)=default; Memory& operator=(Memory&&)=default;
    uint32_t* words32() {
        if(data.empty()) { data.assign(words,0u);if(generator) { generator(data);allWritten=true; } }
        return data.data();
    }
    bool isWritten(uint32_t i) const { return allWritten||zeroed||(!written.empty()&&written[i]); }
    void markWritten(uint32_t i) { if(allWritten||zeroed)return;if(written.empty())written.assign(words,0);written[i]=1; }
    void markAllWritten() { allWritten=true;written.clear(); }
    bool fullyWritten() const {
        if(allWritten||zeroed)return true;if(written.empty())return false;
        return std::all_of(written.begin(),written.end(),[](uint8_t v) { return v!=0; });
    }
};

// Global execution context of the single host thread.
struct Context {
    Findings* findings=nullptr; std::string shader; uint64_t dispatch=0;
    uint32_t thread=0, lane=0, epoch=0;
};
inline Context& context() { static Context c;return c; }
inline void finding(const std::string& kind,const std::string& detail) {
    auto& c=context();if(c.findings)c.findings->add(kind,c.shader+" "+detail);
}

struct ByteAddressBuffer {
    Memory* m=nullptr; bool uav=false; std::string slot;
    uint Load(uint address) const {
        uint32_t index=0;if(!locate(address,index,"Load"))return 0;
        const uint32_t value=m->words32()[index]; // Materializes generated contents before the initialization check.
        if(!m->isWritten(index))finding("uninitialized_read",m->label+" word "+std::to_string(index));
        if(uav)trackRead(index);
        return value;
    }
    uint4 Load4(uint address) const { return {Load(address),Load(address+4),Load(address+8),Load(address+12)}; }
protected:
    bool locate(uint address,uint32_t& index,const char* what) const {
        if(!m) { finding("unbound_view",slot+" "+what);return false; }
        if(address%4) { finding("misaligned",m->label+" byte "+std::to_string(address));return false; }
        index=address/4;
        if(index>=m->words) { finding("out_of_bounds",m->label+" "+what+" word "+std::to_string(index)+" of "+std::to_string(m->words));return false; }
        return true;
    }
    void trackRead(uint32_t i) const {
        auto& c=context();auto& a=m->accessed(i,uint32_t(c.dispatch));
        if((a&1)&&m->writer[i]!=c.thread)
            finding("uav_race_write_read",m->label+" word "+std::to_string(i)+" writer "+std::to_string(m->writer[i])+" reader "+std::to_string(c.thread));
        if(!(a&2)) { a|=2;m->reader[i]=c.thread; } else if(m->reader[i]!=c.thread)a|=4;
    }
};
struct RWByteAddressBuffer : ByteAddressBuffer {
    void Store(uint address,uint value) const {
        uint32_t index=0;if(!locate(address,index,"Store"))return;
        auto& c=context();auto& a=m->accessed(index,uint32_t(c.dispatch));
        if((a&1)&&m->writer[index]!=c.thread)
            finding("uav_race_write_write",m->label+" word "+std::to_string(index)+" threads "+std::to_string(m->writer[index])+","+std::to_string(c.thread));
        if((a&2)&&((a&4)||m->reader[index]!=c.thread))
            finding("uav_race_read_write",m->label+" word "+std::to_string(index)+" writer "+std::to_string(c.thread));
        a|=1;m->writer[index]=c.thread;m->words32()[index]=value;m->markWritten(index);
    }
};

// groupshared storage with barrier-interval race tracking. Initial contents are a quiet-NaN poison.
template<class T> struct SharedState {
    std::vector<T> data; std::vector<uint32_t> writeEpoch, writer, readEpoch, reader; std::vector<uint8_t> init, multi;
    std::string name;
    void reset(size_t n,const std::string& label) {
        name=label;data.assign(n,poison());writeEpoch.assign(n,UINT32_MAX);writer.assign(n,0);
        readEpoch.assign(n,UINT32_MAX);reader.assign(n,0);init.assign(n,0);multi.assign(n,0);
    }
    static T poison() { if constexpr(std::is_same_v<T,float>) { uint32_t b=0x7fc0deadu;float f;std::memcpy(&f,&b,4);return f; } else return T(0xdeadbeefu); }
    bool check(size_t i) {
        if(i>=data.size()) { finding("groupshared_out_of_bounds",name+" index "+std::to_string(i));return false; }
        return true;
    }
    T read(size_t i) {
        if(!check(i))return T(0);
        auto& c=context();
        if(!init[i])finding("groupshared_uninitialized_read",name+"["+std::to_string(i)+"] lane "+std::to_string(c.lane));
        if(writeEpoch[i]==c.epoch&&writer[i]!=c.lane)
            finding("groupshared_race_write_read",name+"["+std::to_string(i)+"] writer lane "+std::to_string(writer[i])+" reader lane "+std::to_string(c.lane)+" epoch "+std::to_string(c.epoch));
        if(readEpoch[i]!=c.epoch) { readEpoch[i]=c.epoch;reader[i]=c.lane;multi[i]=0; } else if(reader[i]!=c.lane)multi[i]=1;
        return data[i];
    }
    void write(size_t i,T v) {
        if(!check(i))return;
        auto& c=context();
        if(writeEpoch[i]==c.epoch&&writer[i]!=c.lane)
            finding("groupshared_race_write_write",name+"["+std::to_string(i)+"] lanes "+std::to_string(writer[i])+","+std::to_string(c.lane));
        if(readEpoch[i]==c.epoch&&(multi[i]||reader[i]!=c.lane))
            finding("groupshared_race_read_write",name+"["+std::to_string(i)+"] writer lane "+std::to_string(c.lane)+" epoch "+std::to_string(c.epoch));
        writeEpoch[i]=c.epoch;writer[i]=c.lane;init[i]=1;data[i]=v;
    }
};
template<class T> struct SharedRef {
    SharedState<T>* s; size_t i;
    operator T() const { return s->read(i); }
    SharedRef& operator=(T v) { s->write(i,v);return *this; }
    SharedRef& operator=(const SharedRef& o) { T v=o;s->write(i,v);return *this; }
    SharedRef& operator+=(T v) { T cur=s->read(i);s->write(i,cur+v);return *this; }
};
template<class T,size_t N> struct Shared {
    SharedState<T> state;
    void reset(const char* name) { state.reset(N,name); }
    SharedRef<T> operator[](uint64_t i) { return {&state,size_t(i)}; }
};
template<class T,size_t N,size_t M> struct Shared2 {
    SharedState<T> state;
    struct Row { SharedState<T>* s; uint64_t row; SharedRef<T> operator[](uint64_t j) const {
        if(row>=N||j>=M) { finding("groupshared_out_of_bounds",s->name+"["+std::to_string(row)+"]["+std::to_string(j)+"]");return {s,SIZE_MAX}; }
        return {s,size_t(row*M+j)}; } };
    void reset(const char* name) { state.reset(N*M,name); }
    Row operator[](uint64_t i) { return {&state,i}; }
};

struct ThreadId { uint3 group, groupThread, dispatch; uint index=0; };

// Fiber switching. On x86-64 System V a minimal callee-saved-register switch (no signal-mask system call);
// elsewhere POSIX ucontext. Exceptions never cross a fiber boundary: a lane catches and stores its own.
#if defined(__x86_64__) && !defined(_WIN32)
extern "C" void hlsl_cpu_switch(void** save,void* load);
asm(".text\n.globl hlsl_cpu_switch\n.type hlsl_cpu_switch,@function\nhlsl_cpu_switch:\n"
    "pushq %rbp\npushq %rbx\npushq %r12\npushq %r13\npushq %r14\npushq %r15\n"
    "subq $8,%rsp\nstmxcsr (%rsp)\nfnstcw 4(%rsp)\nmovq %rsp,(%rdi)\nmovq %rsi,%rsp\n"
    "ldmxcsr (%rsp)\nfldcw 4(%rsp)\naddq $8,%rsp\n"
    "popq %r15\npopq %r14\npopq %r13\npopq %r12\npopq %rbx\npopq %rbp\nret\n"
    ".size hlsl_cpu_switch,.-hlsl_cpu_switch\n");
struct FiberContext {
    void* sp=nullptr;
    void start(char* stack,size_t bytes,void(*entry)()) {
        auto top=reinterpret_cast<uintptr_t>(stack+bytes)&~uintptr_t(15);
        auto* frame=reinterpret_cast<uint64_t*>(top)-9; // control words, six registers, entry, fake return
        uint32_t control[2]={0,0};asm volatile("stmxcsr %0\n\tfnstcw %1":"=m"(control[0]),"=m"(control[1]));
        std::memcpy(&frame[0],control,8);for(int i=1;i<=6;i++)frame[i]=0;
        frame[7]=reinterpret_cast<uint64_t>(entry);frame[8]=0;sp=frame;
    }
    static void swap(FiberContext& from,FiberContext& to) { hlsl_cpu_switch(&from.sp,to.sp); }
};
#else
struct FiberContext {
    ucontext_t context{};
    void start(char* stack,size_t bytes,void(*entry)()) {
        if(getcontext(&context)!=0)throw std::runtime_error("getcontext failed");
        context.uc_stack.ss_sp=stack;context.uc_stack.ss_size=bytes;context.uc_link=nullptr;makecontext(&context,entry,0);
    }
    static void swap(FiberContext& from,FiberContext& to) { if(swapcontext(&from.context,&to.context)!=0)throw std::runtime_error("swapcontext failed"); }
};
#endif

// Cooperative lanes of one workgroup.
class Lanes {
    struct Lane { FiberContext context; int state=0; int site=-1; std::exception_ptr error; };
    enum { ready=0, waiting=1, done=2 };
    std::vector<Lane> lanes; FiberContext scheduler;
    std::function<void(uint)> body; uint current=0;
    static constexpr size_t stackBytes=128*1024;
    static Lanes*& active() { static Lanes* a=nullptr;return a; }
    // Process-lifetime pool of uninitialized lane stacks, reused by every group and dispatch.
    static std::vector<std::unique_ptr<char[]>>& stacks() { static std::vector<std::unique_ptr<char[]>> s;return s; }
    [[noreturn]] static void entry() {
        Lanes* self=active();const uint lane=self->current;
        try { self->body(lane); } catch(...) { self->lanes[lane].error=std::current_exception(); }
        self->lanes[lane].state=done;FiberContext::swap(self->lanes[lane].context,self->scheduler);
        std::abort(); // A finished lane is never resumed.
    }
public:
    void barrier(int site) {
        auto& lane=lanes[current];lane.state=waiting;lane.site=site;
        FiberContext::swap(lane.context,scheduler);
    }
    // Returns false after a divergent barrier (reported as a finding); the group's remaining work is abandoned.
    bool run(uint count,uint64_t firstThread,Order order,std::mt19937_64& random,const std::function<void(uint)>& f) {
        body=f;lanes.assign(count,Lane{});
        while(stacks().size()<count)stacks().push_back(std::unique_ptr<char[]>(new char[stackBytes]));
        for(uint l=0;l<count;l++)lanes[l].context.start(stacks()[l].get(),stackBytes,&Lanes::entry);
        std::vector<uint> sequence(count);
        auto& c=context();c.epoch=0;
        for(;;) {
            for(uint l=0;l<count;l++)sequence[l]=order==Order::Descending?count-1-l:l;
            if(order==Order::Shuffled)std::shuffle(sequence.begin(),sequence.end(),random);
            for(uint l:sequence) {
                if(lanes[l].state!=ready)continue;
                current=l;active()=this;c.lane=l;c.thread=uint32_t(firstThread+l);
                FiberContext::swap(scheduler,lanes[l].context);
            }
            for(auto& lane:lanes)if(lane.error)std::rethrow_exception(lane.error);
            uint wait=0,finished=0;int site=-1;bool mixed=false;
            for(auto& lane:lanes) {
                if(lane.state==waiting) { if(wait&&site!=lane.site)mixed=true;site=lane.site;wait++; } else finished++;
            }
            if(!wait)return true;
            if(finished) { finding("divergent_barrier",std::to_string(finished)+" lanes returned while "+std::to_string(wait)+" wait");return false; }
            if(mixed) { finding("divergent_barrier","lanes wait at different barrier sites");return false; }
            c.epoch++;for(auto& lane:lanes)lane.state=ready;
        }
    }
};

struct Options { Order order=Order::Ascending; uint64_t seed=1; Findings* findings=nullptr; };
struct Binding {
    std::vector<Memory*> srv, uav; std::vector<uint32_t> constants=std::vector<uint32_t>(64,0); uint3 groups{1,1,1};
};

// Base of every translated shader: HLSL intrinsics resolve here before any host function.
struct Shader {
    Lanes* lanes=nullptr;
    void hlsl_barrier(int site) {
        if(!lanes)throw std::runtime_error("Barrier in a shader translated without barrier support");
        lanes->barrier(site);
    }
    static float asfloat(uint v) { float f;std::memcpy(&f,&v,4);return f; }
    static float asfloat(float v) { return v; }
    static uint asuint(float v) { uint u;std::memcpy(&u,&v,4);return u; }
    static uint asuint(uint v) { return v; }
    static float min(float a,float b) { return std::fmin(a,b); }
    static float max(float a,float b) { return std::fmax(a,b); }
    template<class A,class B,class=std::enable_if_t<std::is_integral_v<A>&&std::is_integral_v<B>>>
    static std::common_type_t<A,B> min(A a,B b) { using C=std::common_type_t<A,B>;return C(a)<C(b)?C(a):C(b); }
    template<class A,class B,class=std::enable_if_t<std::is_integral_v<A>&&std::is_integral_v<B>>>
    static std::common_type_t<A,B> max(A a,B b) { using C=std::common_type_t<A,B>;return C(a)>C(b)?C(a):C(b); }
    static float abs(float x) { return std::fabs(x); }
    static int abs(int x) { return x<0?-x:x; }
    static float exp(float x) { return std::exp(x); }
    static float log(float x) { return std::log(x); }
    static float sqrt(float x) { return std::sqrt(x); }
    static float rsqrt(float x) { return 1.0f/std::sqrt(x); }
    static float pow(float a,float b) { return std::pow(a,b); }
    static float cos(float x) { return std::cos(x); }
    static float sin(float x) { return std::sin(x); }
    static float floor(float x) { return std::floor(x); }
};

// Runs every group of one dispatch. S is a translated shader struct.
template<class S> void run(const Binding& b,const Options& o) {
    auto& c=context();c.findings=o.findings;c.shader=S::name;c.dispatch++;
    if(b.groups.x==0||b.groups.y==0||b.groups.z==0||b.groups.x>65535||b.groups.y>65535||b.groups.z>65535)
        finding("dispatch_limit","groups "+std::to_string(b.groups.x)+","+std::to_string(b.groups.y)+","+std::to_string(b.groups.z));
    auto s=std::make_unique<S>();s->bind(b);s->constants(b.constants.data());
    for(auto* m:b.uav)if(m) { m->words32();m->track(); }
    const uint3 t=S::threads;const uint groupSize=t.x*t.y*t.z;
    std::mt19937_64 random(o.seed);Lanes lanes;if(S::barriers)s->lanes=&lanes;
    std::vector<uint64_t> groupOrder;groupOrder.reserve(uint64_t(b.groups.x)*b.groups.y*b.groups.z);
    for(uint64_t g=0;g<uint64_t(b.groups.x)*b.groups.y*b.groups.z;g++)groupOrder.push_back(g);
    if(o.order==Order::Descending)std::reverse(groupOrder.begin(),groupOrder.end());
    if(o.order==Order::Shuffled)std::shuffle(groupOrder.begin(),groupOrder.end(),random);
    for(uint64_t flat:groupOrder) {
        ThreadId id;id.group={uint(flat%b.groups.x),uint((flat/b.groups.x)%b.groups.y),uint(flat/(uint64_t(b.groups.x)*b.groups.y))};
        s->reset();
        auto body=[&](uint lane) {
            ThreadId l=id;l.index=lane;l.groupThread={lane%t.x,(lane/t.x)%t.y,lane/(t.x*t.y)};
            l.dispatch={id.group.x*t.x+l.groupThread.x,id.group.y*t.y+l.groupThread.y,id.group.z*t.z+l.groupThread.z};
            s->invoke(l);
        };
        const uint64_t first=flat*groupSize;
        if(S::barriers)lanes.run(groupSize,first,o.order,random,body);
        else {
            c.epoch=0;
            for(uint i=0;i<groupSize;i++) { const uint lane=o.order==Order::Descending?groupSize-1-i:i;c.lane=lane;c.thread=uint32_t(first+lane);body(lane); }
        }
    }
    c.findings=nullptr;
}
using Runner=void(*)(const Binding&,const Options&);
struct Entry { const char* name; Runner run; bool barriers; uint3 threads; uint srvSlots, uavSlots; };
}
