// New code, MPL-2.0. CPU-only geometry/forecast tests; no model or device access.
#define main originalFixtureMain
#include "model_weights_fixture_test.cpp"
#undef main
namespace {
unsigned extra=0;
void check(bool ok,const char* why){require(ok,why);++extra;}
template<class F> void refuses(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"padding negative case accepted");}
Tensor shapeTensor(const std::string& n,const std::vector<uint32_t>& s){Tensor t;t.name=n;t.shape=s;t.rows=s.size()==1?1:s[0];t.elements=1;for(auto d:s)t.elements=multiply(t.elements,d);t.cols=static_cast<uint32_t>(t.elements/t.rows);return t;}
}
int main(int argc,char** argv){
 try {
    require(argc==1,"This check accepts no model/header/payload argument");
    const int original=originalFixtureMain(argc,argv);require(original==0,"original fixture failed");
    using Mode=chandra::dc::WeightStorageExperiment;namespace ws=chandra::dc::weight_storage;
    check(ws::parse("exact")==Mode::exact&&ws::parse("page4096")==Mode::page4096,"explicit selector differs");
    for(const std::string bad:{"","4096","PAGE4096","page4096 ","auto"})refuses([&]{ws::parse(bad);});
    const auto invalid=static_cast<Mode>(99);refuses([&]{ws::name(invalid);});refuses([&]{ws::allocationBytes(4096,invalid);});
    for(uint64_t bytes:std::initializer_list<uint64_t>{4ull,4092ull,4096ull,4100ull,134215680ull,ws::maximumBufferBytes}){
        auto physical=ws::allocationBytes(bytes,Mode::page4096);check(physical>=bytes&&physical%4096==0&&physical<=ws::maximumBufferBytes,"page allocation bounds differ");
        check(ws::allocationBytes(bytes,Mode::exact)==bytes,"exact allocation differs");
        auto g=ws::geometry(static_cast<uint32_t>(bytes/4),static_cast<uint32_t>(physical));
        struct Box{uint32_t left,top,front,right,bottom,back;};auto b=g.prefix<Box>();
        check(g.logicalWords==bytes/4&&g.logicalBytes==bytes&&g.physicalBytes==physical&&g.boxed()==(physical!=bytes),"logical geometry exposes padding");
        check(b.left==0&&b.top==0&&b.front==0&&b.right==bytes&&b.bottom==1&&b.back==1,"logical-prefix box differs");
    }
    for(uint64_t bytes:std::initializer_list<uint64_t>{0ull,2ull,UINT64_MAX,ws::maximumBufferBytes+4})refuses([&]{ws::allocationBytes(bytes,Mode::page4096);});
    refuses([&]{ws::geometry(0);});refuses([&]{ws::geometry(UINT32_MAX);});refuses([&]{ws::geometry(1024,4092);});refuses([&]{ws::geometry(1,5);});refuses([&]{ws::geometry(1,134217732);});
    auto t=shapeTensor("model.language_model.embed_tokens.weight",{248320,2560});const auto exact=shardPlan(t),padded=shardPlan(t,Mode::page4096);
    check(exact.size()==10&&padded.size()==10,"head topology changed");
    Json topology=Json::array();uint64_t headLogical=0,headPhysical=0;
    for(size_t i=0;i<exact.size();++i){const auto& a=exact[i];const auto& b=padded[i];
        check(a.firstRow==b.firstRow&&a.rows==b.rows&&a.words==b.words&&a.elements==b.elements&&a.sourceBytes==b.sourceBytes&&a.storageBytes==b.storageBytes&&a.allocationBytes==a.storageBytes,"opt-in changed head logical topology");
        check(i<9?(a.rows==26214&&a.storageBytes==134215680&&b.allocationBytes==134217728):(a.rows==12394&&a.storageBytes==63457280&&b.allocationBytes==63459328),"head physical plan differs");
        headLogical+=a.storageBytes;headPhysical+=b.allocationBytes;topology.push_back({{"shard",i},{"first_row",a.firstRow},{"rows",a.rows},{"last_row",a.firstRow+a.rows-1},{"logical_bytes",a.storageBytes},{"physical_bytes",b.allocationBytes},{"srv_words",a.words}});
    }
    check(exact[0].firstRow+exact[0].rows-1==26213&&exact[1].firstRow+exact[1].rows-1==52427,"observed/failing token topology differs");
    check(headPhysical-headLogical==20480,"head allocation padding forecast differs");
    uint64_t physical=0,logical=0,uniqueShards=0;size_t graphNames=0;
    for(const auto& item:pinnedShapes()){
        if(item.first.rfind("mtp.",0)==0)continue;
        ++graphNames;
        if(item.first=="lm_head.weight")continue;
        auto tensor=shapeTensor(item.first,item.second);auto a=shardPlan(tensor),e=shardPlan(tensor,Mode::exact),b=shardPlan(tensor,Mode::page4096);check(a.size()==e.size()&&a.size()==b.size(),"full-graph shard count changed");
        for(size_t i=0;i<a.size();++i){check(a[i].firstRow==b[i].firstRow&&a[i].rows==b[i].rows&&a[i].words==b[i].words&&a[i].sourceBytes==b[i].sourceBytes&&a[i].storageBytes==b[i].storageBytes&&a[i].allocationBytes==e[i].allocationBytes,"full-graph default/logical shard changed");check(b[i].allocationBytes>=b[i].storageBytes&&b[i].allocationBytes%4096==0&&b[i].allocationBytes<=maximumShardBytes,"full-graph padded physical limit violated");physical+=b[i].allocationBytes;logical+=a[i].storageBytes;++uniqueShards;}
    }
    check(graphNames==724,"head sharing/MTP graph count differs");
    Tensor odd=shapeTensor("norm",{3});auto o=shardPlan(odd,Mode::page4096)[0];check(o.sourceBytes==6&&o.storageBytes==8&&o.words==2&&o.allocationBytes==4096,"odd BF16 packing versus physical padding differs");
    Tensor oddRows=shapeTensor("odd",{7,33554431});auto orows=shardPlan(oddRows,Mode::page4096);check(orows.size()==4&&orows[0].rows==2&&orows[0].sourceBytes==134217724&&orows[0].storageBytes==134217724&&orows[0].allocationBytes==134217728&&orows.back().rows==1,"odd row boundary/physical ceiling differs");
    Tensor tooWide=shapeTensor("wide",{1,67108865});refuses([&]{shardPlan(tooWide,Mode::page4096);});refuses([&]{shardPlan(t,invalid);});
    check(ws::withinBudget(0,physical,physical)&&ws::withinBudget(123,physical,physical+123),"exact physical admission failed");
    check(!ws::withinBudget(0,physical,physical-1)&&!ws::withinBudget(physical+1,0,physical)&&!ws::withinBudget(UINT64_MAX,1,UINT64_MAX),"physical budget/overflow admitted");
    const auto forecast=Json::parse(chandra::dc::weightStorageForecastJson(Mode::page4096));check(forecast["logical_storage_bytes"]==logical&&forecast["physical_allocation_bytes"]==physical&&forecast["padding_bytes"]==physical-logical&&forecast["unique_shards"]==uniqueShards,"padded source forecast differs from independent sum");
    const auto base=Json::parse(chandra::dc::weightStorageForecastJson(Mode::exact));check(base["physical_allocation_bytes"]==logical&&base["padding_bytes"]==0,"baseline forecast changed physical bytes");
    std::cout<<Json{{"schema","chandra.weight-tail-padding38.cpu.v1"},{"additional_checks",extra},{"full_graph_unique_shards",uniqueShards},{"logical_storage_bytes",logical},{"physical_allocation_bytes",physical},{"padding_bytes",physical-logical},{"head_logical_bytes",headLogical},{"head_physical_bytes",headPhysical},{"head_topology",topology},{"shape_forecast",forecast},{"model_or_gpu_access",false}}.dump()<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
