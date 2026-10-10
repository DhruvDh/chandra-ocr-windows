// New ChandraNative code, MPL-2.0. Read-only pinned safetensors import.
// Model weights remain governed by the model's own supplied license.
#include "api.h"
#include "import_row_source.h"
#include "import_row_early.h"
#include "import_row_watch.h"
#include "shard1_boundary_watch.h"
#include "final_import_tail_audit.h"
#include "../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#endif

namespace chandra::dc::weight_detail {
using Json = nlohmann::json;
constexpr uint64_t maximumShardBytes = 128ull * 1024 * 1024;
constexpr uint64_t defaultBudgetBytes = 12ull * 1024 * 1024 * 1024;
constexpr uint64_t maximumHeaderBytes = 1024 * 1024;
constexpr uint64_t pinnedModelBytes = 10591220088ull;
constexpr uint64_t pinnedConfigBytes = 2773;
constexpr const char* revision = "af93b47dba1b47b6640c86ccf487ed2260ab9a09";
constexpr const char* modelHash = "0804568be9f099d6479fad9ed77a4da4611f3c1e7bc6e009af7dce45e8aa3847";
constexpr const char* configHash = "e26f17b70463de21fd68f1ef6d8f67f8e33e65d55a7a6895242130a18db60587";
void require(bool condition, const char* reason) { if (!condition) throw std::runtime_error(reason); }
uint64_t add(uint64_t a, uint64_t b) {
    require(b <= std::numeric_limits<uint64_t>::max() - a, "weight range addition overflow"); return a + b;
}
uint64_t multiply(uint64_t a, uint64_t b) {
    require(!b || a <= std::numeric_limits<uint64_t>::max() / b, "weight shape multiplication overflow"); return a * b;
}
uint64_t number(const Json& j) {
    require(j.is_number_unsigned(), "weight descriptor requires an unsigned integer"); return j.get<uint64_t>();
}
Json strictJson(const uint8_t* data, size_t bytes) {
    std::vector<std::set<std::string>> objects;
    auto duplicateCheck = [&](int, Json::parse_event_t event, Json& value) {
        if (event == Json::parse_event_t::object_start) objects.emplace_back();
        else if (event == Json::parse_event_t::key) {
            require(!objects.empty() && objects.back().insert(value.get<std::string>()).second, "duplicate safetensors/config JSON key");
        } else if (event == Json::parse_event_t::object_end) {
            require(!objects.empty(), "unbalanced weight JSON object"); objects.pop_back();
        }
        return true;
    };
    return Json::parse(data, data + bytes, duplicateCheck, true, false);
}
void keys(const Json& j, std::initializer_list<const char*> names) {
    require(j.is_object() && j.size() == names.size(), "weight descriptor fields differ");
    for (auto name : names) require(j.contains(name), "weight descriptor field missing");
}
struct Tensor {
    std::string name, dtype; std::vector<uint32_t> shape;
    uint64_t begin = 0, end = 0, elements = 0; uint32_t rows = 0, cols = 0;
    bool bf16 = true;
};
using Inventory = std::map<std::string, Tensor>;
Inventory parseHeader(const uint8_t* header, size_t bytes, uint64_t payloadBytes) {
    require(bytes > 1 && bytes <= maximumHeaderBytes && header[0] == '{', "invalid safetensors header size/prefix");
    auto root = strictJson(header, bytes); require(root.is_object(), "safetensors header must be an object");
    Inventory result;
    for (auto it = root.begin(); it != root.end(); ++it) {
        if (it.key() == "__metadata__") {
            require(it.value().is_object(), "safetensors metadata must be an object");
            for (const auto& value : it.value()) require(value.is_string(), "safetensors metadata must contain strings");
            continue;
        }
        require(!it.key().empty() && it.key().size() <= 256, "invalid tensor name");
        const auto& desc = it.value(); keys(desc, {"dtype", "shape", "data_offsets"});
        require(desc.at("dtype").is_string(), "weight dtype is not a string");
        Tensor t; t.name = it.key(); t.dtype = desc.at("dtype").get<std::string>();
        require(t.dtype == "BF16" || t.dtype == "F32", "unsupported weight dtype; no dequantization or implicit cast");
        t.bf16 = t.dtype == "BF16";
        const auto& shape = desc.at("shape");
        require(shape.is_array() && !shape.empty() && shape.size() <= 8, "unsupported weight rank");
        t.elements = 1;
        for (const auto& dimension : shape) {
            uint64_t d = number(dimension); require(d && d <= UINT32_MAX, "invalid weight dimension");
            t.elements = multiply(t.elements, d); t.shape.push_back(static_cast<uint32_t>(d));
        }
        t.rows = shape.size() == 1 ? 1 : t.shape[0];
        uint64_t cols = t.elements / t.rows; require(cols <= UINT32_MAX, "flattened weight width overflow"); t.cols = static_cast<uint32_t>(cols);
        if (!t.bf16) {
            // Explicit F32 storage is supported only for small normalization/state
            // parameters. Pinned af93 inventory below still requires original BF16.
            require(shape.size() == 1 && t.elements <= 16384 &&
                (t.name.find("norm") != std::string::npos || t.name.find(".A_log") != std::string::npos || t.name.find(".dt_bias") != std::string::npos),
                "FP32 weight is outside the small normalization/state parameter seam");
        }
        const auto& offsets = desc.at("data_offsets"); require(offsets.is_array() && offsets.size() == 2, "invalid weight offsets");
        t.begin = number(offsets[0]); t.end = number(offsets[1]);
        require(t.begin <= t.end && t.end <= payloadBytes && t.end - t.begin == multiply(t.elements, t.bf16 ? 2 : 4), "weight offsets/shape/byte length mismatch");
        require(t.begin % (t.bf16 ? 2 : 4) == 0, "weight offset alignment mismatch");
        const auto name=t.name; require(result.emplace(name, std::move(t)).second, "duplicate tensor name");
    }
    require(!result.empty(), "empty safetensors inventory");
    std::vector<const Tensor*> ranges; for (const auto& item : result) ranges.push_back(&item.second);
    std::sort(ranges.begin(), ranges.end(), [](auto a, auto b) { return a->begin < b->begin; });
    uint64_t consumed = 0;
    for (const auto* t : ranges) { require(t->begin == consumed, "safetensors gap/overlap/unaccounted bytes"); consumed = t->end; }
    require(consumed == payloadBytes, "safetensors trailing/unaccounted payload bytes");
    return result;
}
using Shapes = std::map<std::string, std::vector<uint32_t>>;
Shapes pinnedShapes() {
    Shapes s;
    auto put = [&](const std::string& name, std::initializer_list<uint32_t> dims) { require(s.emplace(name, dims).second, "duplicate source inventory name"); };
    put("lm_head.weight", {248320,2560}); put("model.language_model.embed_tokens.weight", {248320,2560}); put("model.language_model.norm.weight", {2560});
    for (unsigned i=0; i<32; ++i) {
        std::string p="model.language_model.layers."+std::to_string(i)+".";
        put(p+"input_layernorm.weight",{2560}); put(p+"post_attention_layernorm.weight",{2560});
        put(p+"mlp.down_proj.weight",{2560,9216}); put(p+"mlp.gate_proj.weight",{9216,2560}); put(p+"mlp.up_proj.weight",{9216,2560});
        if (i%4 == 3) {
            put(p+"self_attn.k_norm.weight",{256}); put(p+"self_attn.q_norm.weight",{256});
            put(p+"self_attn.k_proj.weight",{1024,2560}); put(p+"self_attn.v_proj.weight",{1024,2560});
            put(p+"self_attn.q_proj.weight",{8192,2560}); put(p+"self_attn.o_proj.weight",{2560,4096});
        } else {
            put(p+"linear_attn.A_log",{32}); put(p+"linear_attn.dt_bias",{32}); put(p+"linear_attn.conv1d.weight",{8192,1,4});
            put(p+"linear_attn.in_proj_a.weight",{32,2560}); put(p+"linear_attn.in_proj_b.weight",{32,2560});
            put(p+"linear_attn.in_proj_qkv.weight",{8192,2560}); put(p+"linear_attn.in_proj_z.weight",{4096,2560});
            put(p+"linear_attn.norm.weight",{128}); put(p+"linear_attn.out_proj.weight",{2560,4096});
        }
    }
    for (unsigned i=0; i<24; ++i) {
        std::string p="model.visual.blocks."+std::to_string(i)+".";
        put(p+"attn.proj.bias",{1024}); put(p+"attn.proj.weight",{1024,1024});
        put(p+"attn.qkv.bias",{3072}); put(p+"attn.qkv.weight",{3072,1024});
        put(p+"mlp.linear_fc1.bias",{4096}); put(p+"mlp.linear_fc1.weight",{4096,1024});
        put(p+"mlp.linear_fc2.bias",{1024}); put(p+"mlp.linear_fc2.weight",{1024,4096});
        put(p+"norm1.bias",{1024}); put(p+"norm1.weight",{1024}); put(p+"norm2.bias",{1024}); put(p+"norm2.weight",{1024});
    }
    put("model.visual.merger.linear_fc1.bias",{4096}); put("model.visual.merger.linear_fc1.weight",{4096,4096});
    put("model.visual.merger.linear_fc2.bias",{2560}); put("model.visual.merger.linear_fc2.weight",{2560,4096});
    put("model.visual.merger.norm.bias",{1024}); put("model.visual.merger.norm.weight",{1024});
    put("model.visual.patch_embed.proj.bias",{1024}); put("model.visual.patch_embed.proj.weight",{1024,3,2,16,16}); put("model.visual.pos_embed.weight",{2304,1024});
    put("mtp.fc.weight",{2560,5120}); put("mtp.norm.weight",{2560}); put("mtp.pre_fc_norm_embedding.weight",{2560}); put("mtp.pre_fc_norm_hidden.weight",{2560});
    std::string p="mtp.layers.0.";
    put(p+"input_layernorm.weight",{2560}); put(p+"post_attention_layernorm.weight",{2560});
    put(p+"mlp.down_proj.weight",{2560,9216}); put(p+"mlp.gate_proj.weight",{9216,2560}); put(p+"mlp.up_proj.weight",{9216,2560});
    put(p+"self_attn.k_norm.weight",{256}); put(p+"self_attn.q_norm.weight",{256});
    put(p+"self_attn.k_proj.weight",{1024,2560}); put(p+"self_attn.v_proj.weight",{1024,2560});
    put(p+"self_attn.q_proj.weight",{8192,2560}); put(p+"self_attn.o_proj.weight",{2560,4096});
    require(s.size()==739, "pinned tensor count differs"); return s;
}
void validatePinned(const Inventory& tensors) {
    auto expected=pinnedShapes(); require(tensors.size()==expected.size(), "pinned tensor set/count differs");
    for (const auto& item : expected) {
        auto it=tensors.find(item.first); require(it!=tensors.end(), "pinned graph/MTP tensor missing");
        require(it->second.shape==item.second && it->second.dtype=="BF16", "pinned tensor shape/dtype differs");
    }
}
struct Selection {
    std::vector<std::string> requested;
    std::set<std::string> effective;
    bool fullGraph=false, tiedAliasExpansion=false;
};
Selection selectGraph(const Inventory& tensors,const std::vector<std::string>& names) {
    Selection selection; selection.requested=names; selection.fullGraph=names.empty();
    if(names.empty()) {
        for(const auto& item:tensors)if(item.first.rfind("mtp.",0)!=0)selection.effective.insert(item.first);
    }else {
        for(const auto& name:names) {
            require(tensors.find(name)!=tensors.end(),"selected graph tensor is unknown");
            require(name.rfind("mtp.",0)!=0,"MTP tensors are outside selected graph diagnostics");
            require(selection.effective.insert(name).second,"duplicate selected graph tensor");
        }
    }
    const std::string embedding="model.language_model.embed_tokens.weight",head="lm_head.weight";
    if(selection.effective.count(embedding)||selection.effective.count(head)) {
        selection.tiedAliasExpansion=selection.effective.count(embedding)==0||selection.effective.count(head)==0;
        selection.effective.insert(embedding);selection.effective.insert(head);
    }
    return selection;
}
struct Shard { uint32_t firstRow, rows, words; uint64_t elements, sourceBytes, storageBytes, allocationBytes; };
std::vector<Shard> shardPlan(const Tensor& t, WeightStorageExperiment mode=WeightStorageExperiment::exact) {
    (void)weight_storage::name(mode);
    uint64_t rowBytes=multiply(t.cols,t.bf16?2:4); require(rowBytes && rowBytes<=maximumShardBytes, "weight row exceeds shard limit");
    uint64_t rowsPerShard=maximumShardBytes/rowBytes;
    // A BF16 shard starts at a whole uint32_t word; odd column widths require
    // an even row boundary. The final odd element receives explicit zero padding.
    if (t.bf16 && t.cols%2 && rowsPerShard>1) rowsPerShard-=rowsPerShard%2;
    require(rowsPerShard>0, "invalid rows-per-shard"); std::vector<Shard> result;
    for (uint64_t row=0; row<t.rows;) {
        uint64_t count=std::min<uint64_t>(rowsPerShard,t.rows-row), elements=multiply(count,t.cols);
        uint64_t words=t.bf16?add(elements,1)/2:elements, storage=multiply(words,4);
        // Include padding in the stated 128 MiB maximum, even at the edge.
        if (storage>maximumShardBytes) { require(count>1,"padded weight row exceeds shard limit"); --count; elements=multiply(count,t.cols); words=t.bf16?add(elements,1)/2:elements; storage=multiply(words,4); }
        require(words<=UINT32_MAX && storage<=maximumShardBytes,"weight shard allocation overflow");
        result.push_back({static_cast<uint32_t>(row),static_cast<uint32_t>(count),static_cast<uint32_t>(words),elements,multiply(elements,t.bf16?2:4),storage,weight_storage::allocationBytes(storage,mode)}); row+=count;
    }
    return result;
}
uint64_t readU64LE(const uint8_t* bytes) { uint64_t n=0; for(unsigned i=0;i<8;++i)n|=uint64_t(bytes[i])<<(8*i); return n; }
} // namespace chandra::dc::weight_detail

namespace chandra::dc {
namespace weight_srv_only {
using Json=weight_detail::Json;
static Json descriptorJson(const Descriptor& d) {
 return {{"ByteWidth",d.byteWidth},{"Usage",d.usage},{"BindFlags",d.bindFlags},{"MiscFlags",d.miscFlags},{"CPUAccessFlags",d.cpuAccess},{"StructureByteStride",d.stride},{"SRV_Format",d.srvFormat},{"SRV_ViewDimension",d.srvDimension},{"SRV_FirstElement",d.srvFirst},{"SRV_NumElements",d.srvWords},{"SRV_Flags",d.srvFlags},{"readonly_weight",d.readonlyWeight},{"SRV_present",d.srvPresent},{"UAV_present",d.uavPresent},{"SRV_same_resource",d.srvSameResource}};
}
static Json contract() {
 return {{"schema","chandra.directcompute.weight-srv-only-contract.v1"},{"requested",true},{"Usage",defaultUsage},{"BindFlags",shaderResourceBind},{"MiscFlags",rawMisc},{"CPUAccessFlags",0},{"StructureByteStride",0},{"readonly_weight",true},{"SRV_present",true},{"UAV_present",false},{"SRV_same_resource",true},{"SRV_Format",rawFormat},{"SRV_ViewDimension",bufferExDimension},{"SRV_FirstElement",0},{"SRV_Flags",rawSRVFlag},{"extent","Exact original logical words times four; SRV_NumElements equals original logical words"},{"descriptor_checks_required",import_row::final_tails::maximumOperations},{"observation","CPU descriptor queries after actual DEFAULT initialization and SRV creation return, before the original per-shard drain; no extra GPU observation"}};
}
} // namespace weight_srv_only
std::string weightStorageForecastJson(WeightStorageExperiment mode) {
    using namespace weight_detail;
    uint64_t logical=0,physical=0,padding=0,shardCount=0;
    for(const auto& item:pinnedShapes()){
        if(item.first=="lm_head.weight"||item.first.rfind("mtp.",0)==0)continue;
        Tensor t;t.name=item.first;t.shape=item.second;t.rows=t.shape.size()==1?1:t.shape[0];t.elements=1;
        for(auto d:t.shape)t.elements=multiply(t.elements,d);
        t.cols=static_cast<uint32_t>(t.elements/t.rows);
        for(const auto& s:shardPlan(t,mode)){logical=add(logical,s.storageBytes);physical=add(physical,s.allocationBytes);padding=add(padding,s.allocationBytes-s.storageBytes);++shardCount;}
    }
    return Json{{"schema","chandra.directcompute.weight-storage-forecast.v1"},{"experiment",weight_storage::name(mode)},{"logical_storage_bytes",logical},{"physical_allocation_bytes",physical},{"padding_bytes",padding},{"unique_shards",shardCount},{"maximum_buffer_bytes",weight_storage::maximumBufferBytes},{"maximum_weight_budget_bytes",defaultBudgetBytes},{"gpu_or_model_access",false},{"head_storage_shared",true},{"limits","Pinned shape-only allocation forecast; not observed live memory, driver commitment, fit or correctness"}}.dump();
}
} // namespace chandra::dc

#ifdef _WIN32
namespace chandra::dc::weight_detail {
class Hash {
    BCRYPT_ALG_HANDLE algorithm=nullptr; BCRYPT_HASH_HANDLE hash=nullptr; std::vector<uint8_t> object;
public:
    Hash() {
        require(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA256 provider failed");
        try {
            ULONG length=0,received=0;
            require(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&length),sizeof(length),&received,0)>=0,"SHA256 property failed"); object.resize(length);
            require(BCryptCreateHash(algorithm,&hash,object.data(),length,nullptr,0,0)>=0,"SHA256 create failed");
        } catch(...) { BCryptCloseAlgorithmProvider(algorithm,0); throw; }
    }
    ~Hash() { if(hash)BCryptDestroyHash(hash); if(algorithm)BCryptCloseAlgorithmProvider(algorithm,0); }
    void update(const uint8_t* data,uint32_t bytes) { require(BCryptHashData(hash,const_cast<PUCHAR>(data),bytes,0)>=0,"SHA256 update failed"); }
    std::string finish() {
        uint8_t result[32]; require(BCryptFinishHash(hash,result,sizeof(result),0)>=0,"SHA256 finish failed");
        std::ostringstream out; out<<std::hex<<std::setfill('0'); for(auto v:result)out<<std::setw(2)<<unsigned(v); return out.str();
    }
};
class Mapping {
    HANDLE file=INVALID_HANDLE_VALUE, mapping=nullptr; uint64_t length=0; uint32_t granularity=0;
public:
    class View {
        const uint8_t* base=nullptr; const uint8_t* data=nullptr;
        uint64_t mapOffset=0,mapBytes=0,prefix=0;
    public:
        View(HANDLE mapping,uint32_t granularity,uint64_t offset,uint64_t bytes) {
            require(bytes>0 && bytes<=maximumShardBytes,"mapping view exceeds bounded shard");
            uint64_t aligned=offset-offset%granularity, skip=offset-aligned, size=add(skip,bytes);
            mapOffset=aligned;mapBytes=size;prefix=skip;
            require(size<=SIZE_MAX,"mapping view size overflow");
            base=static_cast<const uint8_t*>(MapViewOfFile(mapping,FILE_MAP_READ,static_cast<DWORD>(aligned>>32),static_cast<DWORD>(aligned),static_cast<SIZE_T>(size)));
            require(base!=nullptr,"read-only model mapping view failed"); data=base+skip;
        }
        // Opt-in target release at the original iteration-end boundary. Failed
        // unmap leaves the pointer owned for the ordinary destructor cleanup.
        bool release() noexcept {if(!base || !UnmapViewOfFile(base))return false;base=nullptr;data=nullptr;return true;}
        ~View(){if(base)UnmapViewOfFile(base);}
        View(const View&)=delete; View& operator=(const View&)=delete;
        const uint8_t* get() const { return data; }
        uint64_t mappedFileOffset() const{return mapOffset;}
        uint64_t mappedBytes() const{return mapBytes;}
        uint64_t skippedPrefix() const{return prefix;}
    };
    explicit Mapping(const std::filesystem::path& path) {
        static_assert(sizeof(void*)==8,"full model importer requires a 64-bit process");
        // Only read sharing: a writer/delete-capable concurrent opener is refused.
        file=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL|FILE_FLAG_RANDOM_ACCESS,nullptr);
        require(file!=INVALID_HANDLE_VALUE,"read-only model file open failed");
        LARGE_INTEGER n;
        if(!GetFileSizeEx(file,&n) || n.QuadPart<=0){CloseHandle(file);file=INVALID_HANDLE_VALUE;throw std::runtime_error("model file size failed");}
        length=static_cast<uint64_t>(n.QuadPart); SYSTEM_INFO info;GetSystemInfo(&info);granularity=info.dwAllocationGranularity;
        mapping=CreateFileMappingW(file,nullptr,PAGE_READONLY,0,0,nullptr);
        if(!mapping){CloseHandle(file);file=INVALID_HANDLE_VALUE;throw std::runtime_error("read-only model file mapping failed");}
    }
    ~Mapping(){if(mapping)CloseHandle(mapping);if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);}
    Mapping(const Mapping&)=delete; Mapping& operator=(const Mapping&)=delete;
    uint64_t size() const{return length;}
    uint32_t allocationGranularity() const{return granularity;}
    View view(uint64_t offset,uint64_t bytes) const { require(offset<=length && bytes<=length-offset,"mapping range outside file");return View(mapping,granularity,offset,bytes); }
    std::string digest() const {
        Hash h; constexpr uint32_t chunk=64*1024*1024;
        for(uint64_t offset=0;offset<length;){uint32_t n=static_cast<uint32_t>(std::min<uint64_t>(chunk,length-offset));auto v=view(offset,n);h.update(v.get(),n);offset+=n;}
        return h.finish();
    }
    bool equal(uint64_t a,uint64_t b,uint64_t bytes) const {
        constexpr uint32_t chunk=32*1024*1024;
        for(uint64_t offset=0;offset<bytes;){uint32_t n=static_cast<uint32_t>(std::min<uint64_t>(chunk,bytes-offset));auto x=view(add(a,offset),n);auto y=view(add(b,offset),n);if(std::memcmp(x.get(),y.get(),n)!=0)return false;offset+=n;}
        return true;
    }
};
} // namespace chandra::dc::weight_detail

namespace chandra::dc {
struct ModelWeights::Impl { std::map<std::string,Weight> weights; weight_detail::Json provenance; };
ModelWeights::ModelWeights(Device& device,const std::wstring& directory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames)
    : ModelWeights(device,directory,maximumWeightBytes,selectedTensorNames,WeightStorageExperiment::exact) {}
ModelWeights::ModelWeights(Device& device,const std::wstring& directory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment storageExperiment)
    : ModelWeights(device,directory,maximumWeightBytes,selectedTensorNames,storageExperiment,nullptr) {}
ModelWeights::ModelWeights(Device& device,const std::wstring& directory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment storageExperiment,import_row::Observer* sourceObserver)
    : ModelWeights(device,directory,maximumWeightBytes,selectedTensorNames,storageExperiment,sourceObserver,nullptr) {}
ModelWeights::ModelWeights(Device& device,const std::wstring& directory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment storageExperiment,import_row::Observer* sourceObserver,import_row::EarlyObserver* earlyObserver)
    : ModelWeights(device,directory,maximumWeightBytes,selectedTensorNames,storageExperiment,sourceObserver,earlyObserver,nullptr) {}
ModelWeights::ModelWeights(Device& device,const std::wstring& directory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment storageExperiment,import_row::Observer* sourceObserver,import_row::EarlyObserver* earlyObserver,import_row::WatchObserver* watchObserver)
    : ModelWeights(device,directory,maximumWeightBytes,selectedTensorNames,storageExperiment,sourceObserver,earlyObserver,watchObserver,nullptr) {}
ModelWeights::ModelWeights(Device& device,const std::wstring& directory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment storageExperiment,import_row::Observer* sourceObserver,import_row::EarlyObserver* earlyObserver,import_row::WatchObserver* watchObserver,import_row::Shard1Observer* shard1Observer)
    : ModelWeights(device,directory,maximumWeightBytes,selectedTensorNames,storageExperiment,sourceObserver,earlyObserver,watchObserver,shard1Observer,WeightImportAPI::updateSubresource,nullptr) {}
ModelWeights::ModelWeights(Device& device,const std::wstring& directory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment storageExperiment,import_row::Observer* sourceObserver,import_row::EarlyObserver* earlyObserver,import_row::WatchObserver* watchObserver,import_row::Shard1Observer* shard1Observer,WeightImportAPI importAPI,import_row::FinalTailObserver* finalTailObserver) : ModelWeights(device,directory,maximumWeightBytes,selectedTensorNames,storageExperiment,sourceObserver,earlyObserver,watchObserver,shard1Observer,importAPI,finalTailObserver,false) {}
ModelWeights::ModelWeights(Device& device,const std::wstring& directory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment storageExperiment,import_row::Observer* sourceObserver,import_row::EarlyObserver* earlyObserver,import_row::WatchObserver* watchObserver,import_row::Shard1Observer* shard1Observer,WeightImportAPI importAPI,import_row::FinalTailObserver* finalTailObserver,bool weightSrvOnly) : impl(std::make_unique<Impl>()) {
    (void)weight_storage::name(storageExperiment);
    (void)weight_import::name(importAPI);
    weight_detail::require(weight_srv_only::admitted(weightSrvOnly,finalTailObserver!=nullptr,storageExperiment==WeightStorageExperiment::exact,importAPI==WeightImportAPI::defaultInitial,true),"SRV-only weights require separate exact final-tail DEFAULT initialization");
    weight_detail::require(!finalTailObserver || finalTailObserver->weightSrvOnly()==weightSrvOnly,"Final-tail observer/SRV-only selector pair differs");
    weight_detail::require(!finalTailObserver || finalTailObserver->importAPI()==importAPI,"Final-tail observer/import API pair differs");
    weight_detail::require(importAPI==WeightImportAPI::updateSubresource || finalTailObserver,"DEFAULT initial data is a separate explicit final-tail experiment");
    weight_detail::require(!finalTailObserver || (!sourceObserver&&!earlyObserver&&!watchObserver&&!shard1Observer&&storageExperiment==WeightStorageExperiment::exact&&selectedTensorNames.empty()),"Final-tail API experiment requires separate exact full graph; historical API observers excluded");
    using namespace weight_detail;
    require(maximumWeightBytes>0 && maximumWeightBytes<=defaultBudgetBytes,"weight budget must be positive and at most 12 GiB");
    require(!sourceObserver || (storageExperiment==WeightStorageExperiment::exact && selectedTensorNames.empty()),"Import source observer requires exact allocation and the full original graph");
    require(!earlyObserver || sourceObserver,"Early copy requires the current CPU upload-source observer");
    require(!watchObserver || (sourceObserver && !earlyObserver && storageExperiment==WeightStorageExperiment::exact && selectedTensorNames.empty()),"Import-only watch requires source audit, exact full graph and no separate early observer");
    require(!shard1Observer || (!sourceObserver && !earlyObserver && !watchObserver && storageExperiment==WeightStorageExperiment::exact && selectedTensorNames.empty()),"Shard1 boundary observer requires separate exact import-only full graph");
    auto modelDirectory=std::filesystem::path(directory);
    Mapping config(modelDirectory/L"config.json"); require(config.size()==pinnedConfigBytes,"pinned config length differs");
    require(config.digest()==configHash,"pinned config SHA256 differs"); auto configView=config.view(0,config.size());auto configJson=strictJson(configView.get(),static_cast<size_t>(config.size()));
    require(configJson.at("model_type")=="qwen3_5" && configJson.at("text_config").at("num_hidden_layers")==32 && configJson.at("vision_config").at("depth")==24,"pinned graph config differs");
    Mapping model(modelDirectory/L"model.safetensors"); require(model.size()==pinnedModelBytes,"pinned model length differs");
    uint64_t headerBytes;{auto first=model.view(0,8);headerBytes=readU64LE(first.get());}
    require(headerBytes>1 && headerBytes<=maximumHeaderBytes && headerBytes<=model.size()-8,"safetensors header length exceeds bound");
    uint64_t payloadBegin=add(8,headerBytes);Inventory tensors;
    {auto header=model.view(8,headerBytes);tensors=parseHeader(header.get(),static_cast<size_t>(headerBytes),model.size()-payloadBegin);}
    validatePinned(tensors);
    const auto selection=selectGraph(tensors,selectedTensorNames);
    // Exactly one whole-file authentication pass. Later reads are necessary
    // tied-byte comparison and direct shard uploads, never another full hash.
    require(model.digest()==modelHash,"pinned model SHA256 differs");
    const auto& embed=tensors.at("model.language_model.embed_tokens.weight");const auto& head=tensors.at("lm_head.weight");
    const bool tieSelected=selection.effective.count("lm_head.weight")!=0;
    if(tieSelected)require(embed.shape==head.shape && embed.dtype==head.dtype && model.equal(add(payloadBegin,embed.begin),add(payloadBegin,head.begin),embed.end-embed.begin),"embedding and lm_head bytes differ; tied config flag is insufficient");
    uint64_t requiredBytes=0,skippedBytes=0;
    for(const auto& item:tensors){
        const auto& t=item.second;
        if(t.name.rfind("mtp.",0)==0){skippedBytes=add(skippedBytes,t.end-t.begin);continue;}
        if(!selection.effective.count(t.name))continue;
        if(t.name=="lm_head.weight")continue;
        for(const auto& shard:shardPlan(t,storageExperiment))requiredBytes=add(requiredBytes,shard.allocationBytes);
    }
    uint64_t baseline=device.trackedBufferBytes();require(weight_storage::withinBudget(baseline,requiredBytes,maximumWeightBytes),"global tracked buffer budget would exceed weight admission");
    Json watchOperations=Json::array();uint32_t watchOrdinal=0;
    auto watchOperation=[&](const Tensor& t,const Shard& shard,uint32_t ordinal,uint32_t tensorOrdinal,uint32_t chunkIndex,uint64_t consumed){
        return Json{{"operation_ordinal",ordinal},{"tensor_ordinal",tensorOrdinal},{"tensor_name",t.name},{"dtype",t.dtype},{"shape",t.shape},{"chunk_index",chunkIndex},
            {"first_row",shard.firstRow},{"rows",shard.rows},{"cols",t.cols},{"logical_elements",shard.elements},{"words",shard.words},
            {"tensor_data_offsets",{t.begin,t.end}},{"source_file_offset",add(payloadBegin,add(t.begin,consumed))},{"source_bytes",shard.sourceBytes},
            {"logical_storage_bytes",shard.storageBytes},{"allocation_bytes",shard.allocationBytes}};
    };
    if(watchObserver || shard1Observer || finalTailObserver){
        uint32_t ordinal=0,tensorOrdinal=0;
        for(const auto& item:tensors){const auto& t=item.second;if(t.name.rfind("mtp.",0)==0 || !selection.effective.count(t.name) || t.name=="lm_head.weight")continue;
            ++tensorOrdinal;uint32_t chunkIndex=0;uint64_t consumed=0;
            for(const auto& shard:shardPlan(t,storageExperiment)){watchOperations.push_back(watchOperation(t,shard,++ordinal,tensorOrdinal,chunkIndex++,consumed));consumed=add(consumed,shard.sourceBytes);}
        }
        Json observerPlan={{"source_tensor_count",tensors.size()},{"effective_tensor_count",selection.effective.size()},{"unique_upload_tensor_count",tensorOrdinal},
            {"allocation_selector","exact"},{"allocation_bytes",requiredBytes},{"operations",watchOperations},
            {"model_binding",{{"revision",revision},{"model_bytes",model.size()},{"model_sha256",modelHash},{"config_bytes",config.size()},{"config_sha256",configHash},
                {"header_bytes",headerBytes},{"payload_begin",payloadBegin},{"whole_file_hash_performed",true},{"tied_byte_equality_performed",tieSelected},
                {"source_mapping","Held read-only Mapping; current upload view remains alive through its upload, original drain and watcher sample; views/handles close on constructor return"}}}};
        const auto observerIdentity=Json::parse(device.identityJson());
        if(watchObserver)watchObserver->prepare(observerPlan,observerIdentity);
        if(weightSrvOnly)observerPlan["weight_srv_only"]=weight_srv_only::contract();
        if(finalTailObserver)finalTailObserver->prepare(observerPlan);
        if(shard1Observer){observerPlan["model_binding"]["source_mapping"]="Held read-only Mapping; each original upload View unmaps at iteration end; mapping/file handles close on constructor return";shard1Observer->prepare(observerPlan,observerIdentity);}
    }
    Json inventory=Json::array(),skipped=Json::array(),omitted=Json::array();uint64_t uploaded=0,uploadedLogical=0;
    for(const auto& item:tensors){
        const auto& t=item.second;
        Json row={{"name",t.name},{"shape",t.shape},{"dtype",t.dtype},{"data_offsets",{t.begin,t.end}},{"logical_elements",t.elements},{"rows",t.rows},{"cols",t.cols}};
        if(t.name.rfind("mtp.",0)==0){row["reason"]="unused MTP/speculative head; original graph does not execute it";skipped.push_back(std::move(row));continue;}
        if(!selection.effective.count(t.name)){row["reason"]="graph tensor omitted from explicitly selected trained-layer diagnostic";omitted.push_back(std::move(row));continue;}
        if(t.name=="lm_head.weight"){row["shared_with"]="model.language_model.embed_tokens.weight";inventory.push_back(std::move(row));continue;}
        Weight weight;weight.rows=t.rows;weight.cols=t.cols;weight.bf16=t.bf16;Json shards=Json::array();
        uint64_t consumed=0;
        for(const auto& shard:shardPlan(t,storageExperiment)){
            if(watchObserver || shard1Observer || finalTailObserver){const auto& planned=watchOperations.at(watchOrdinal);const auto operation=watchOperation(t,shard,++watchOrdinal,planned.at("tensor_ordinal").get<uint32_t>(),planned.at("chunk_index").get<uint32_t>(),consumed);
                if(watchObserver)watchObserver->begin(operation);if(shard1Observer)shard1Observer->begin(operation);if(finalTailObserver)finalTailObserver->begin(operation);}
            require(weight_storage::withinBudget(device.trackedBufferBytes(),shard.allocationBytes,maximumWeightBytes),"tracked weight budget exhausted before upload");
            if(importAPI==WeightImportAPI::defaultInitial){
                auto source=model.view(add(payloadBegin,add(t.begin,consumed)),shard.sourceBytes);
                std::vector<uint32_t> initialScratch;
                const void* pointer=weight_import::initialPayload(source.get(),shard.sourceBytes,shard.storageBytes,shard.words,initialScratch);
                const Json mapping={{"file_offset",source.mappedFileOffset()},{"mapped_bytes",source.mappedBytes()},{"skipped_prefix",source.skippedPrefix()},{"allocation_granularity",model.allocationGranularity()}};
                finalTailObserver->beforeAPI(mapping,pointer,shard.storageBytes);
                Buffer buffer=weightSrvOnly?device.srvOnlyWeightWords(shard.words,pointer):weight_import::createInitial(device,t.bf16,shard.words,pointer);
                buffer.logicalElements=shard.elements;buffer.packedBF16=t.bf16;if(weightSrvOnly)finalTailObserver->returned(device,buffer);else finalTailObserver->returned(buffer);
                device.drain(); // Original per-shard drain; source and any bounded scratch remain alive.
                finalTailObserver->afterDrain();
                require(device.trackedBufferBytes()<=maximumWeightBytes,"tracked weight budget exceeded after allocation");
                weight.shardFirstRows.push_back(shard.firstRow);weight.shards.push_back(std::move(buffer));consumed=add(consumed,shard.sourceBytes);uploaded=add(uploaded,shard.allocationBytes);uploadedLogical=add(uploadedLogical,shard.storageBytes);
                shards.push_back({{"first_row",shard.firstRow},{"rows",shard.rows},{"source_bytes",shard.sourceBytes},{"storage_bytes",shard.storageBytes},{"words",shard.words}});
            }else{
            Buffer buffer=storageExperiment==WeightStorageExperiment::exact?(t.bf16?device.words(shard.words):device.floats(shard.words)):
                device.paddedWeightWords(shard.words,static_cast<uint32_t>(shard.allocationBytes));
            buffer.logicalElements=shard.elements;buffer.packedBF16=t.bf16;
            if(watchObserver)watchObserver->allocated(buffer);
            if(shard1Observer)shard1Observer->allocated(buffer);
            auto source=model.view(add(payloadBegin,add(t.begin,consumed)),shard.sourceBytes);
            if(watchObserver)watchObserver->mapped({{"file_offset",source.mappedFileOffset()},{"mapped_bytes",source.mappedBytes()},{"skipped_prefix",source.skippedPrefix()},{"allocation_granularity",model.allocationGranularity()}});
            if(shard1Observer)shard1Observer->mapped({{"file_offset",source.mappedFileOffset()},{"mapped_bytes",source.mappedBytes()},{"skipped_prefix",source.skippedPrefix()},{"allocation_granularity",model.allocationGranularity()}});
            const bool observeSource=sourceObserver && t.name=="model.language_model.embed_tokens.weight" && shard.firstRow==0;
            if(observeSource){
                require(t.bf16 && t.rows==248320 && t.cols==2560 && consumed==0 && shard.rows==26214 && shard.words==33553920 && shard.sourceBytes==import_row::shardBytes && shard.storageBytes==shard.sourceBytes && shard.allocationBytes==shard.sourceBytes,"Import source observer head topology differs");
                import_row::Context context;context.fileBytes=model.size();context.payloadBegin=payloadBegin;context.tensorBegin=t.begin;
                context.shardFileOffset=add(payloadBegin,t.begin);context.allocationGranularity=model.allocationGranularity();
                context.mappedFileOffset=source.mappedFileOffset();context.skippedPrefix=source.skippedPrefix();context.mappedBytes=source.mappedBytes();
                sourceObserver->beforeSubmit(source.get(),static_cast<uint32_t>(shard.storageBytes),context);sourceObserver->submitting();
            }
            if(watchObserver)watchObserver->beforeUpload(buffer,source.get(),shard.sourceBytes);
            if(shard1Observer)shard1Observer->beforeUpload(buffer,source.get(),shard.sourceBytes);
            if(watchObserver)watchObserver->submitting();
            if(shard1Observer)shard1Observer->submitting();
            if(shard.storageBytes!=shard.sourceBytes){
                // At most one bounded shard scratch; zero high halfword explicitly.
                std::vector<uint32_t> padded(shard.words,0);std::memcpy(padded.data(),source.get(),static_cast<size_t>(shard.sourceBytes));if(finalTailObserver)finalTailObserver->beforeAPI({{"file_offset",source.mappedFileOffset()},{"mapped_bytes",source.mappedBytes()},{"skipped_prefix",source.skippedPrefix()},{"allocation_granularity",model.allocationGranularity()}},padded.data(),shard.storageBytes);device.upload(buffer,padded.data(),static_cast<uint32_t>(shard.storageBytes));
            }else {if(finalTailObserver)finalTailObserver->beforeAPI({{"file_offset",source.mappedFileOffset()},{"mapped_bytes",source.mappedBytes()},{"skipped_prefix",source.skippedPrefix()},{"allocation_granularity",model.allocationGranularity()}},source.get(),shard.storageBytes);device.upload(buffer,source.get(),static_cast<uint32_t>(shard.storageBytes));}
            if(finalTailObserver)finalTailObserver->returned(buffer);
            if(watchObserver)watchObserver->returned();
            if(shard1Observer)shard1Observer->returned();
            if(observeSource)sourceObserver->afterReturn(source.get(),static_cast<uint32_t>(shard.storageBytes));
            if(observeSource)sourceObserver->draining();
            device.drain(); // Bound driver upload staging to one authenticated shard, not the whole checkpoint.
            if(finalTailObserver)finalTailObserver->afterDrain();
            if(observeSource)sourceObserver->afterDrain(source.get(),static_cast<uint32_t>(shard.storageBytes));
            if(observeSource && earlyObserver)earlyObserver->capture(device,buffer,*sourceObserver);
            if(watchObserver)watchObserver->afterDrain(device,buffer,*sourceObserver);
            if(shard1Observer)shard1Observer->afterDrain(device,buffer);
            require(device.trackedBufferBytes()<=maximumWeightBytes,"tracked weight budget exceeded after allocation");
            weight.shardFirstRows.push_back(shard.firstRow);weight.shards.push_back(std::move(buffer));consumed=add(consumed,shard.sourceBytes);uploaded=add(uploaded,shard.allocationBytes);uploadedLogical=add(uploadedLogical,shard.storageBytes);
            shards.push_back({{"first_row",shard.firstRow},{"rows",shard.rows},{"source_bytes",shard.sourceBytes},{"storage_bytes",shard.storageBytes},{"words",shard.words}});
            if(storageExperiment!=WeightStorageExperiment::exact){shards.back()["allocation_bytes"]=shard.allocationBytes;shards.back()["padding_bytes"]=shard.allocationBytes-shard.storageBytes;}
            if(shard1Observer && shard1Observer->needsOwnViewRelease()){const bool released=source.release();shard1Observer->afterSourceRelease(device,released);}
            }
        }
        require(consumed==t.end-t.begin,"upload did not consume complete tensor");row["shards"]=std::move(shards);inventory.push_back(std::move(row));impl->weights.emplace(t.name,std::move(weight));
    }
    // Copying Weight shares Buffer::storage; no second GPU allocation.
    if(tieSelected)impl->weights.emplace("lm_head.weight",impl->weights.at("model.language_model.embed_tokens.weight"));
    device.drain(); // Producer completion precedes releasing every mapped source view.
    require(!sourceObserver || sourceObserver->completed(),"Requested import upload-source observation did not complete");
    if(earlyObserver){
        earlyObserver->afterImport(impl->weights.at("model.language_model.embed_tokens.weight").shards.at(0),impl->weights.at("lm_head.weight").shards.at(0));
        require(earlyObserver->completed(),"Requested early import-row observation did not complete");
    }
    impl->provenance={{"schema","chandra.directcompute.model-weights.v2"},{"model","datalab-to/chandra-ocr-2"},{"revision",revision},{"model_bytes",model.size()},{"model_sha256",modelHash},{"config_bytes",config.size()},{"config_sha256",configHash},{"tensor_count",tensors.size()},{"full_graph_requested",selection.fullGraph},{"requested_tensor_names",selection.requested},{"effective_tensor_names",std::vector<std::string>(selection.effective.begin(),selection.effective.end())},{"uploaded_tensor_names",impl->weights.size()},{"maximum_shard_bytes",maximumShardBytes},{"maximum_tracked_weight_bytes",maximumWeightBytes},{"tracked_buffer_bytes_before",baseline},{"uploaded_storage_bytes",uploaded},{"tracked_buffer_bytes_after",device.trackedBufferBytes()},{"tie_byte_equality_performed",tieSelected},{"tie_byte_equality",tieSelected},{"tie_config_flag_used_as_proof",false},{"tied_alias_selection_expanded",selection.tiedAliasExpansion},{"inventory",std::move(inventory)},{"omitted_graph_tensors",std::move(omitted)},{"skipped_mtp_bytes",skippedBytes},{"skipped_mtp",std::move(skipped)},{"mapping_lifetime","all read-only views/handles released on constructor return after upload/drain"},{"implicit_cast_or_dequantization",false},{"numerical_acceptance",false},{"full_model_qualified",false}};
    if(finalTailObserver){impl->provenance["weight_import_api"]=weight_import::name(importAPI);impl->provenance["API_operations_completed"]=watchOrdinal;impl->provenance["UpdateSubresource_calls"]=importAPI==WeightImportAPI::updateSubresource?watchOrdinal:0;impl->provenance["DEFAULT_initial_data_Device_creations"]=importAPI==WeightImportAPI::defaultInitial?watchOrdinal:0;}
    if(weightSrvOnly)impl->provenance["weight_srv_only"]=weight_srv_only::contract();
    if(storageExperiment!=WeightStorageExperiment::exact){
        impl->provenance["weight_storage_experiment"]=weight_storage::name(storageExperiment);
        impl->provenance["uploaded_logical_storage_bytes"]=uploadedLogical;
        impl->provenance["allocated_padding_bytes"]=uploaded-uploadedLogical;
        impl->provenance["physical_allocation_bytes"]=uploaded;
        impl->provenance["shape_only_full_graph_allocation_forecast"]=Json::parse(weightStorageForecastJson(storageExperiment));
    }
}
ModelWeights::~ModelWeights()=default;
const Weight& ModelWeights::at(const std::string& name) const {
    auto it=impl->weights.find(name);if(it==impl->weights.end())throw std::out_of_range("DirectCompute graph weight not imported: "+name);return it->second;
}
bool ModelWeights::contains(const std::string& name) const {return impl->weights.find(name)!=impl->weights.end();}
std::string ModelWeights::provenanceJson() const {return impl->provenance.dump();}
} // namespace chandra::dc

// Placement-only successor02: one implementation in the existing importer TU.
namespace chandra::dc::import_row {
Json Shard1Observer::counters()const{return {{"operations_admitted",counts.admitted},{"original_allocations_completed",counts.allocated},{"original_allocation_bytes",counts.allocatedBytes},
  {"original_upload_calls_admitted",counts.uploadAdmitted},{"original_upload_returns_observed",counts.uploadReturned},{"original_upload_drains_completed",counts.drained},
  {"completed_original_upload_bytes",counts.completedUploadBytes},{"completed_samples",counts.samples},{"direct_read_call_admissions",counts.copyAdmissions},
  {"current_operation_phase",counts.phase},{"comparison_pending",counts.pending},{"own_source_view_release_confirmed",counts.ownViewReleased},{"constructor_return_sample_completed",counts.finalSample}};}
Json Shard1Observer::transport()const{const auto& r=receipt;return {{"source_first_byte",r.extent.firstByte},{"copy_bytes",r.extent.bytes},{"source_logical_bytes",r.extent.logicalBytes},{"source_physical_bytes",r.extent.physicalBytes},
  {"source_descriptor",{{"usage",r.sourceUsage},{"bind_flags",r.sourceBindFlags},{"misc_flags",r.sourceMiscFlags},{"cpu_access_flags",r.sourceCPUAccess},{"structure_byte_stride",r.sourceStride}}},
  {"copies_submitted",r.copiesSubmitted},{"before_copy_drain_completed",r.beforeCopyDrainCompleted},{"after_copy_drain_completed",r.afterCopyDrainCompleted},
  {"staging_created",r.stagingCreated},{"staging_released",r.stagingReleased},{"map_attempted",r.mapAttempted},{"map_attempts",r.readiness.attempts},{"still_drawing_results",r.readiness.stillDrawing},
  {"map_copied",r.mapAttempted&&r.readiness.result==readback::Result::copied},{"unmapped",r.unmapped},{"map_status",r.mapAttempted?Json(uint32_t(r.readiness.status)):Json(nullptr)},
  {"map_wait_nanoseconds",r.readiness.waited.count()},{"operation_nanoseconds",r.operationNanoseconds},{"copy_resubmitted",false},{"retirement_scope","Local staging only; root owns Job/process/channel retirement"}};}
void Shard1Observer::emit(Json record){shard1::require(!journalFailed&&records<shard1::maximumRecords,"Shard1 journal bound or previous failed write");
  record["schema"]="chandra.directcompute.shard1-boundary-journal.v1";record["record_index"]=records+1;record["counts"]=counters();
  shard1::require(record.dump().size()+1<=shard1::maximumRecordBytes,"Shard1 journal record extent exceeded");++writeAttempts;
  try{progress(record);}catch(...){journalFailed=true;throw;}++records;
 }
void Shard1Observer::persistGood(){if(goodAvailable&&!goodAttempted){goodAttempted=true;sink(shard1GoodFile,good.data(),shard1::rowBytes);goodWritten=true;}}
void Shard1Observer::finish(){persistGood();counts.finish();emit({{"event","terminal"},{"outcome",counts.fault?"first_fault":"no_fault_completed_boundaries"},{"first_bad",firstBad},{"last_good",lastGood}});complete=true;}
void Shard1Observer::sample(Device& device,shard1::Boundary boundary){
  counts.copy(boundary);receipt={};
  try{
   const auto words=device.readWordsRange(target,shard1::firstWord,shard1::rowWords,receipt);lastTransport=transport();const auto& r=receipt;
   shard1::require(words.size()==shard1::rowWords&&r.extent.firstByte==uint64_t(shard1::firstWord)*4&&r.extent.bytes==shard1::rowBytes&&r.extent.logicalBytes==shard1::resourceBytes&&r.extent.physicalBytes==shard1::resourceBytes&&
    r.copiesSubmitted==1&&r.beforeCopyDrainCompleted&&r.afterCopyDrainCompleted&&r.stagingReleased&&r.mapAttempted&&r.readiness.result==readback::Result::copied&&r.unmapped,"Shard1 direct-row transport incomplete or extent differs");
   const auto sha=digest(words.data(),shard1::rowBytes);shard1::require(sha.size()==64,"Shard1 GPU digest extent differs");
   const bool equal=std::equal(words.begin(),words.end(),expected.begin());counts.observed(boundary,equal);
   const char* at=boundary==shard1::Boundary::uploadDrain?"after_original_upload_drain_current_source_view_alive":boundary==shard1::Boundary::ownViewRelease?"after_target_source_view_release_before_next_import":"after_constructor_return_source_views_and_mapping_handles_released_before_arithmetic";
   Json observed={{"sample_index",counts.samples},{"at",at},{"operation_ordinal",counts.drained},{"global_row",shard1::globalRow},{"shard_index",shard1::shardIndex},
    {"source_sha256",shard1::expectedSHA},{"gpu_sha256",sha},{"byte_equal_to_actual_upload_source",equal},{"target_source_view_alive",!counts.ownViewReleased},
    {"current_upload_source_view_alive",boundary==shard1::Boundary::uploadDrain},{"constructor_return_observed",boundary==shard1::Boundary::constructorReturn}};
   if(!equal){uint32_t n=0,first=UINT32_MAX,last=0;for(uint32_t i=0;i<shard1::rowWords;++i)if(words[i]!=expected[i]){++n;first=std::min(first,i);last=i;}
    observed["different_u32_words"]=n;observed["first_different_word"]=first;observed["last_different_word"]=last;firstBad=observed;
    badAttempted=true;sink(shard1BadFile,words.data(),shard1::rowBytes);badWritten=true;
   }else{std::copy(words.begin(),words.end(),good.begin());goodAvailable=true;lastGood=observed;}
   emit({{"event","sample"},{"observation",observed},{"operation",current},{"current_source_mapping",mapping},{"transport",lastTransport}});
   if(!equal){finish();throw Shard1Stop();}if(boundary==shard1::Boundary::constructorReturn)finish();
  }catch(...){lastTransport=transport();throw;}
 }
Shard1Observer::Shard1Observer(Sink s,Digest d,Progress p):sink(std::move(s)),digest(std::move(d)),progress(std::move(p)){shard1::require(bool(sink)&&bool(digest)&&bool(progress),"Shard1 callbacks required");}
Json Shard1Observer::forecast(){return {{"schema","chandra.directcompute.shard1-boundary-plan.v1"},{"import_only",true},{"allocation_selector","exact"},{"shard_index",shard1::shardIndex},{"global_row",shard1::globalRow},
  {"first_word",shard1::firstWord},{"row_words",shard1::rowWords},{"row_bytes",shard1::rowBytes},{"original_resource_bytes",shard1::resourceBytes},{"source_row_file_offset",shard1::sourceOffset+uint64_t(shard1::firstWord)*4},
  {"expected_source_sha256",shard1::expectedSHA},{"maximum_operations",watch::operationCount},{"maximum_after_upload_samples",731},{"maximum_own_view_release_samples",1},{"maximum_constructor_return_samples",1},{"maximum_samples",shard1::maximumSamples},
  {"staging_bytes_at_one_time",shard1::rowBytes},{"CPU_row_arrays_bytes",2*shard1::rowBytes},{"maximum_plan_bytes",shard1::maximumPlanBytes},{"maximum_journal_records",shard1::maximumRecords},{"maximum_record_bytes",shard1::maximumRecordBytes},
  {"maximum_artifact_bytes",shard1::maximumArtifactBytes},{"leaves",{{"plan.json",shard1::maximumPlanBytes},{"progress.jsonl",uint64_t(shard1::maximumRecords)*shard1::maximumRecordBytes},{shard1SourceFile,shard1::rowBytes},{shard1GoodFile,shard1::rowBytes},{shard1BadFile,shard1::rowBytes}}},
  {"scope","One original shard1 row at instrumented cutoffs; observer changes timing/residency; no driver cause, repair, arithmetic, numerical/OCR or physical retirement acceptance"}};}
void Shard1Observer::prepare(Json plan,const Json& identity){
  shard1::require(!prepared&&plan.at("operations").size()==watch::operationCount&&plan.at("source_tensor_count")==watch::tensorCount&&plan.at("effective_tensor_count")==watch::graphCount&&plan.at("unique_upload_tensor_count")==watch::uniqueTensorCount&&plan.at("allocation_bytes")==watch::allocationBytes&&plan.at("allocation_selector")=="exact","Shard1 pinned plan differs");
  const auto& model=plan.at("model_binding");shard1::require(model.at("model_bytes")==watch::fileBytes&&model.at("model_sha256")=="0804568be9f099d6479fad9ed77a4da4611f3c1e7bc6e009af7dce45e8aa3847"&&model.at("payload_begin")==shard1::payloadBegin&&model.at("whole_file_hash_performed")==true&&model.at("tied_byte_equality_performed")==true,"Shard1 held-file authentication differs");
  operations=plan.at("operations");uint64_t bytes=0;uint32_t ordinal=0,tensorOrdinal=0;std::string previous;
  for(const auto& op:operations){shard1::require(op.at("operation_ordinal")==++ordinal&&op.at("tensor_name").is_string(),"Shard1 original operation order differs");const auto name=op.at("tensor_name").get<std::string>();
   shard1::require(name.size()<=256&&(previous.empty()||previous<=name),"Shard1 tensor name/order differs");if(name!=previous){previous=name;++tensorOrdinal;}
   shard1::require(op.at("tensor_ordinal")==tensorOrdinal&&op.at("allocation_bytes")==op.at("logical_storage_bytes")&&op.at("logical_storage_bytes").get<uint64_t>()==uint64_t(op.at("words").get<uint32_t>())*4,"Shard1 original tensor/extents differ");
   watch::range(op.at("source_file_offset").get<uint64_t>(),op.at("source_bytes").get<uint64_t>());const auto n=op.at("allocation_bytes").get<uint64_t>();shard1::require(n<=watch::allocationBytes-bytes,"Shard1 cumulative plan extent differs");bytes+=n;
  }
  const auto& op=operations.at(shard1::targetOperation-1);shard1::require(tensorOrdinal==watch::uniqueTensorCount&&bytes==watch::allocationBytes&&op.at("tensor_name")=="model.language_model.embed_tokens.weight"&&op.at("dtype")=="BF16"&&op.at("shape")==Json::array({248320,2560})&&
   op.at("chunk_index")==shard1::shardIndex&&op.at("first_row")==shard1::firstRow&&op.at("rows")==shard1::rows&&op.at("cols")==shard1::columns&&op.at("words")==shard1::resourceBytes/4&&op.at("logical_elements")==uint64_t(shard1::rows)*shard1::columns&&
   op.at("source_file_offset")==shard1::sourceOffset&&op.at("source_bytes")==shard1::resourceBytes&&op.at("allocation_bytes")==shard1::resourceBytes&&op.at("tensor_data_offsets")==Json::array({shard1::tensorBegin,shard1::tensorBegin+uint64_t(248320)*shard1::rowBytes}),"Shard1 target topology/source range differs");
  plan["schema"]="chandra.directcompute.shard1-boundary-upload-plan.v1";plan["observer_plan"]=forecast();plan["device_identity"]=identity;const auto raw=plan.dump()+"\n";
  shard1::require(raw.size()<=shard1::maximumPlanBytes,"Shard1 plan byte bound exceeded");sink("plan.json",raw.data(),raw.size());prepared=true;emit({{"event","plan_persisted"},{"plan_bytes",raw.size()},{"plan_sha256",digest(raw.data(),raw.size())}});
 }
void Shard1Observer::begin(const Json& op){shard1::require(prepared&&counts.admitted<operations.size()&&op==operations.at(counts.admitted),"Shard1 actual upload differs from plan");counts.begin(op.at("operation_ordinal").get<uint32_t>(),op.at("allocation_bytes").get<uint64_t>());current=op;emit({{"event","operation_admitted"},{"operation",current},{"admission_is_not_API_execution",true}});}
void Shard1Observer::allocated(const Buffer& b){shard1::require(b.storage&&b.words==current.at("words")&&b.logicalElements==current.at("logical_elements")&&b.packedBF16==(current.at("dtype")=="BF16"),"Shard1 allocated descriptor differs");counts.allocation(current.at("allocation_bytes").get<uint64_t>());}
void Shard1Observer::mapped(const Json& value){counts.mapped();mapping=value;}
void Shard1Observer::beforeUpload(const Buffer& b,const void* pointer,uint64_t bytes){if(counts.admitted!=shard1::targetOperation)return;
  shard1::require(!bound&&pointer&&bytes==shard1::resourceBytes&&b.storage&&b.packedBF16&&b.words==shard1::resourceBytes/4&&b.logicalElements==uint64_t(shard1::rows)*shard1::columns,"Shard1 actual upload CPU pointer/Buffer differs");
  const auto gran=mapping.at("allocation_granularity").get<uint32_t>();const auto offset=mapping.at("file_offset").get<uint64_t>(),prefix=mapping.at("skipped_prefix").get<uint64_t>(),mapped=mapping.at("mapped_bytes").get<uint64_t>();
  shard1::require(gran&&offset==shard1::sourceOffset-shard1::sourceOffset%gran&&prefix==shard1::sourceOffset-offset&&mapped==prefix+bytes&&offset<=watch::fileBytes&&mapped<=watch::fileBytes-offset,"Shard1 actual source mapping extent differs");
  std::memcpy(expected.data(),static_cast<const unsigned char*>(pointer)+uint64_t(shard1::firstWord)*4,shard1::rowBytes);const auto sha=digest(expected.data(),shard1::rowBytes);counts.source();target=b;bound=true;
  sourceRecord={{"sha256",sha},{"bytes",shard1::rowBytes},{"global_row",shard1::globalRow},{"source_row_file_offset",shard1::sourceOffset+uint64_t(shard1::firstWord)*4},{"mapping",mapping},{"copied_from_actual_submission_pointer",true},{"source_pointer_redirected",false},{"original_Storage_retained",true}};
  sourceAttempted=true;sink(shard1SourceFile,expected.data(),shard1::rowBytes);sourceWritten=true;emit({{"event","before_original_upload_source_snapshot"},{"source",sourceRecord},{"matches_expected_source",sha==shard1::expectedSHA}});
  shard1::require(sha==shard1::expectedSHA,"Shard1 actual source snapshot differs; target upload refused");
 }
void Shard1Observer::submitting(){counts.submit();}
void Shard1Observer::returned(){counts.returned();}
void Shard1Observer::afterDrain(Device& device,const Buffer& b){counts.drain(current.at("allocation_bytes").get<uint64_t>());if(counts.drained==1){counts.skipBeforeTarget();return;}
  shard1::require(bound&&(counts.drained!=shard1::targetOperation||target.storage==b.storage),"Shard1 original upload Buffer identity differs");sample(device,shard1::Boundary::uploadDrain);
 }
bool Shard1Observer::needsOwnViewRelease()const{return counts.drained==shard1::targetOperation&&!counts.ownViewReleased&&!counts.fault&&!counts.refused&&!counts.terminal;}
void Shard1Observer::afterSourceRelease(Device& device,bool succeeded){counts.release(succeeded);sample(device,shard1::Boundary::ownViewRelease);}
void Shard1Observer::afterImport(Device& device,const Weight& embedding,const Weight& head){importStackClosed=true;
  shard1::require(bound&&embedding.rows==248320&&embedding.cols==shard1::columns&&embedding.bf16&&head.rows==embedding.rows&&head.cols==embedding.cols&&head.bf16&&embedding.shards.size()==10&&head.shards.size()==10&&embedding.shardFirstRows.size()==10&&head.shardFirstRows==embedding.shardFirstRows&&embedding.shardFirstRows.at(1)==shard1::firstRow&&
   target.storage==embedding.shards.at(1).storage&&target.storage==head.shards.at(1).storage&&embedding.shards.at(1).words==target.words&&head.shards.at(1).words==target.words&&embedding.shards.at(1).logicalElements==target.logicalElements&&head.shards.at(1).logicalElements==target.logicalElements&&embedding.shards.at(1).packedBF16&&head.shards.at(1).packedBF16,"Shard1 original embedding/head Storage or descriptor changed");
  identitiesChecked=true;sample(device,shard1::Boundary::constructorReturn);
 }
void Shard1Observer::failure(const std::string& reason)noexcept{try{counts.refusal();complete=false;if(firstRefusal.is_null())firstRefusal={{"error",reason.substr(0,1024)},{"operation",current},{"counts",counters()},{"last_transport",lastTransport}};persistGood();if(!journalFailed&&records<shard1::maximumRecords)emit({{"event","partial_or_refused"},{"first_refusal",firstRefusal}});}catch(...){persistenceFailed=true;}}
void Shard1Observer::stackClosedAndReleaseTarget(){importStackClosed=true;target={};targetReleased=true;}
bool Shard1Observer::completed()const{return complete&&counts.terminal&&!counts.refused;}
Json Shard1Observer::report()const{return {{"schema","chandra.directcompute.shard1-boundary-watch.v1"},{"requested",true},{"complete",completed()},{"outcome",completed()?(counts.fault?"first_fault":"no_fault_completed_boundaries"):"partial_or_refused"},
  {"counts",counters()},{"current_operation",current},{"source_snapshot",sourceRecord},{"last_good",lastGood},{"first_bad",firstBad},{"first_refusal",firstRefusal},{"last_transport",lastTransport},{"journal_records_written",records},{"journal_write_attempts",writeAttempts},
  {"source_raw_write_attempted",sourceAttempted},{"source_raw_written_and_flushed",sourceWritten},{"last_good_raw_write_attempted",goodAttempted},{"last_good_raw_written_and_flushed",goodWritten},{"first_bad_raw_write_attempted",badAttempted},{"first_bad_raw_written_and_flushed",badWritten},
  {"journal_write_failed_or_torn",journalFailed},{"failure_evidence_persistence_failed",persistenceFailed},{"same_original_embedding_head_Storage_checked",identitiesChecked},{"import_stack_closed_observed",importStackClosed},{"target_reference_released",targetReleased},
  {"root_physical_retirement_accepted",false},{"arithmetic_gate_eligible",false},{"request_replayed",false},{"plan",forecast()}};}
} // namespace chandra::dc::import_row

namespace chandra::dc::import_row {
struct FinalTailObserver::Impl {
 struct SrvOnlyEvidence {uint32_t checked=0;Json selected=Json::array(),last=nullptr;};
 std::unique_ptr<SrvOnlyEvidence> srvOnly;
 WeightImportAPI api;Sink sink;Digest digest;Progress progress;final_tails::Counts counts;
 std::array<Buffer,tails::targetCount> targets{};
 std::array<std::array<uint32_t,tails::rowWords>,tails::targetCount> expected{},good{};
 static_assert(sizeof(decltype(expected))==final_tails::rowArrayBytes,"Fixed aggregate row-array layout required");
 Json operations,current=nullptr,sources=Json::array(),samples=Json::array(),firstBad=nullptr,firstRefusal=nullptr,lastTransport=nullptr;
 uint32_t records=0,writeAttempts=0;bool prepared=false,complete=false,journalFailed=false,persistenceFailed=false,identitiesChecked=false,targetsReleased=false,constructorReturned=false;
 bool sourceAttempted=false,sourceWritten=false,goodAttempted=false,goodWritten=false,badAttempted=false,badWritten=false;
 Impl(WeightImportAPI a,Sink s,Digest d,Progress p,bool weightSrvOnly):api(a),sink(std::move(s)),digest(std::move(d)),progress(std::move(p)){(void)weight_import::name(api);tails::require(bool(sink)&&bool(digest)&&bool(progress),"Final-tail callbacks required");tails::require(!weightSrvOnly||api==WeightImportAPI::defaultInitial,"SRV-only observer requires DEFAULT initial data");if(weightSrvOnly)srvOnly=std::make_unique<SrvOnlyEvidence>();}
 Json counterReport()const {
  return {{"API_operations_admitted",counts.admitted},{"API_calls_admitted",counts.APIcallsAdmitted},{"API_operations_returned",counts.returned},{"original_drains_completed",counts.drained},{"completed_storage_bytes",counts.storageBytes},
   {"Device_upload_UpdateSubresource_admissions",api==WeightImportAPI::updateSubresource?counts.APIcallsAdmitted:0},{"UpdateSubresource_returns_observed",api==WeightImportAPI::updateSubresource?counts.returned:0},
   {"DEFAULT_initial_data_Device_create_admissions",api==WeightImportAPI::defaultInitial?counts.APIcallsAdmitted:0},{"DEFAULT_initial_data_Device_create_returns",api==WeightImportAPI::defaultInitial?counts.returned:0},
   {"CPU_source_snapshots_completed",counts.sources},{"direct_read_call_admissions",counts.copyAdmissions},{"direct_read_returns_observed",counts.copyReturns},{"copies_submitted_observed",counts.copiesSubmitted},
   {"completed_samples",counts.samples},{"matched_prefix_rows",counts.matched},{"API_operation_pending",counts.APIpending},{"direct_comparison_pending",counts.copyPending}};
 }
 static Json transport(const direct_head_row::Receipt& r) {
  return {{"source_first_byte",r.extent.firstByte},{"copy_bytes",r.extent.bytes},{"source_logical_bytes",r.extent.logicalBytes},{"source_physical_bytes",r.extent.physicalBytes},
   {"source_descriptor",{{"usage",r.sourceUsage},{"bind_flags",r.sourceBindFlags},{"misc_flags",r.sourceMiscFlags},{"cpu_access_flags",r.sourceCPUAccess},{"structure_byte_stride",r.sourceStride}}},
   {"copies_submitted",r.copiesSubmitted},{"before_copy_drain_completed",r.beforeCopyDrainCompleted},{"after_copy_drain_completed",r.afterCopyDrainCompleted},{"staging_created",r.stagingCreated},{"staging_released",r.stagingReleased},
   {"map_attempted",r.mapAttempted},{"map_attempts",r.readiness.attempts},{"still_drawing_results",r.readiness.stillDrawing},{"map_copied",r.mapAttempted&&r.readiness.result==readback::Result::copied},{"unmapped",r.unmapped},{"copy_resubmitted",false}};
 }
 void emit(Json record) {
  tails::require(!journalFailed&&records<final_tails::maximumRecords,"Final-tail journal bound or prior failed write");
  record["schema"]="chandra.directcompute.final-import-tail-journal.v1";record["record_index"]=records+1;record["weight_import_api"]=weight_import::name(api);record["counts"]=counterReport();
  tails::require(record.dump().size()+1<=final_tails::maximumRecordBytes,"Final-tail record byte extent");++writeAttempts;
  try{progress(record);}catch(...){journalFailed=true;throw;}++records;
 }
 void persist() {
  if(counts.sources&&!sourceAttempted){sourceAttempted=true;sink(finalTailSourceFile,expected.data(),size_t(counts.sources)*tails::rowBytes);sourceWritten=true;}
  if(counts.matched&&!goodAttempted){goodAttempted=true;sink(finalTailGoodFile,good.data(),size_t(counts.matched)*tails::rowBytes);goodWritten=true;}
 }
 void finish() {
  persist();counts.finish();emit({{"event","terminal"},{"outcome",counts.fault?"first_fault":"all_selected_tails_match"},{"first_bad",firstBad},{"matched_prefix_rows",counts.matched}});complete=true;
 }
};
FinalTailObserver::FinalTailObserver(WeightImportAPI api,Sink s,Digest d,Progress p):FinalTailObserver(api,std::move(s),std::move(d),std::move(p),false){}
FinalTailObserver::FinalTailObserver(WeightImportAPI api,Sink s,Digest d,Progress p,bool weightSrvOnly):impl(std::make_unique<Impl>(api,std::move(s),std::move(d),std::move(p),weightSrvOnly)){}
FinalTailObserver::~FinalTailObserver()=default;
WeightImportAPI FinalTailObserver::importAPI()const noexcept{return impl->api;}
bool FinalTailObserver::weightSrvOnly()const noexcept{return bool(impl->srvOnly);}
Json FinalTailObserver::forecast(WeightImportAPI api,bool weightSrvOnly) {
 (void)weight_import::name(api);Json targets=Json::array();
 for(uint32_t i=0;i<tails::targetCount;++i){const auto t=tails::target(i);targets.push_back({{"shard_index",i},{"global_row",t.row},{"first_word",t.firstWord},{"resource_bytes",t.bytes},{"row_file_offset",t.rowFileOffset}});}
 Json result={{"schema","chandra.directcompute.final-import-tail-plan.v1"},{"weight_import_api",weight_import::name(api)},{"D3D11_usage","DEFAULT"},{"allocation_selector","exact"},{"import_only",true},{"selected_targets",targets},
  {"original_import_operation_count",final_tails::maximumOperations},{"original_import_storage_bytes",tails::originalAllocationBytes},{"maximum_direct_copies",final_tails::maximumCopies},{"early_GPU_observations",0},
  {"staging_bytes_at_one_time",tails::rowBytes},{"CPU_source_and_good_arrays_bytes",2*final_tails::rowArrayBytes},{"maximum_journal_records",final_tails::maximumRecords},{"maximum_record_bytes",final_tails::maximumRecordBytes},{"maximum_artifact_bytes",final_tails::maximumArtifactBytes},
  {"leaves",{{"progress.jsonl",uint64_t(final_tails::maximumRecords)*final_tails::maximumRecordBytes},{finalTailSourceFile,final_tails::rowArrayBytes},{finalTailGoodFile,final_tails::rowArrayBytes},{finalTailBadFile,tails::rowBytes}}},
  {"scope","Post-constructor selected tails only; no early GPU copies, whole-model equality, fix, arithmetic, OCR or throughput acceptance"}};
 if(weightSrvOnly){tails::require(api==WeightImportAPI::defaultInitial,"SRV-only forecast requires DEFAULT initial data");result["weight_srv_only"]=weight_srv_only::contract();}
 return result;
}
void FinalTailObserver::prepare(const Json& plan) {
 auto& s=*impl;tails::require(!s.prepared&&plan.at("operations").size()==final_tails::maximumOperations&&plan.at("allocation_selector")=="exact"&&plan.at("allocation_bytes")==tails::originalAllocationBytes,"Final-tail original import plan differs");
 tails::require(s.srvOnly?(plan.contains("weight_srv_only")&&plan.at("weight_srv_only")==weight_srv_only::contract()):!plan.contains("weight_srv_only"),"Final-tail SRV-only plan commitment differs");
 const auto& model=plan.at("model_binding");tails::require(model.at("model_bytes")==tails::fileBytes&&model.at("model_sha256")=="0804568be9f099d6479fad9ed77a4da4611f3c1e7bc6e009af7dce45e8aa3847"&&model.at("payload_begin")==tails::payloadBegin&&model.at("whole_file_hash_performed")==true&&model.at("tied_byte_equality_performed")==true,"Final-tail held source binding differs");
 s.operations=plan.at("operations");
 for(uint32_t i=0;i<tails::targetCount;++i){const auto t=tails::target(i);const auto& op=s.operations.at(i);
  tails::require(op.at("operation_ordinal")==i+1&&op.at("tensor_name")=="model.language_model.embed_tokens.weight"&&op.at("dtype")=="BF16"&&op.at("shape")==Json::array({tails::vocabulary,tails::columns})&&op.at("chunk_index")==i&&op.at("first_row")==t.firstRow&&op.at("rows")==t.rows&&op.at("cols")==tails::columns&&op.at("words")==t.words&&op.at("logical_elements")==uint64_t(t.rows)*tails::columns&&op.at("source_file_offset")==t.fileOffset&&op.at("source_bytes")==t.bytes&&op.at("logical_storage_bytes")==t.bytes&&op.at("allocation_bytes")==t.bytes&&op.at("tensor_data_offsets")==Json::array({tails::tensorBegin,tails::tensorBegin+tails::tensorBytes}),"Final-tail selected source topology differs");
 }
 s.emit({{"event","plan"},{"plan",forecast(s.api,bool(s.srvOnly))},{"whole_file_hash_performed",true},{"tied_byte_equality_performed",true}});s.prepared=true;
}
void FinalTailObserver::begin(const Json& op) {
 auto& s=*impl;tails::require(s.prepared&&s.counts.admitted<s.operations.size()&&op==s.operations.at(s.counts.admitted),"Final-tail actual operation differs from planned original order");
 s.counts.begin(op.at("operation_ordinal").get<uint32_t>());s.current=op;
}
void FinalTailObserver::beforeAPI(const Json& mapping,const void* pointer,uint64_t bytes) {
 auto& s=*impl;tails::require(s.counts.APIpending&&pointer&&bytes==s.current.at("logical_storage_bytes"),"Final-tail actual API source pointer/extent differs");
 if(s.current.at("tensor_name")!="model.language_model.embed_tokens.weight"){s.counts.beforeCall();return;}
 const auto i=s.current.at("chunk_index").get<uint32_t>();const auto t=tails::target(i);
 const auto gran=mapping.at("allocation_granularity").get<uint32_t>();const auto offset=mapping.at("file_offset").get<uint64_t>(),prefix=mapping.at("skipped_prefix").get<uint64_t>(),mapped=mapping.at("mapped_bytes").get<uint64_t>();
 tails::require(bytes==t.bytes&&gran&&offset==t.fileOffset-t.fileOffset%gran&&prefix==t.fileOffset-offset&&mapped==prefix+bytes&&offset<=tails::fileBytes&&mapped<=tails::fileBytes-offset,"Final-tail original mapping range differs");
 std::memcpy(s.expected[i].data(),static_cast<const unsigned char*>(pointer)+uint64_t(t.firstWord)*4,tails::rowBytes);const auto sha=s.digest(s.expected[i].data(),tails::rowBytes);tails::require(sha.size()==64,"Final-tail source digest extent");
 const Json known=i==0?Json("e696b6bda9b22dba3ad2a49132d6353302f5b20ad2627909ae5474586a46c07f"):i==1?Json("0575f9588d644d6b669a29517b1dda97ea56b4831b5578edffb99913df8e4ab7"):Json(nullptr);
 s.counts.source(i);Json record={{"shard_index",i},{"global_row",t.row},{"row_file_offset",t.rowFileOffset},{"row_bytes",tails::rowBytes},{"source_payload_offset",uint64_t(i)*tails::rowBytes},{"sha256",sha},{"mapping",mapping},{"actual_API_operation_ordinal",s.counts.admitted},{"copied_from_actual_submission_pointer",true},{"source_view_alive_at_snapshot",true},{"known_checkpoint_row_sha256",known},{"known_checkpoint_row_matches",known.is_null()?Json(nullptr):Json(sha==known.get<std::string>())}};
 s.sources.push_back(record);s.emit({{"event","before_actual_API_source_snapshot"},{"source",record},{"API_call_not_yet_returned",true}});
 tails::require(known.is_null()||sha==known.get<std::string>(),"Final-tail known CPU source hash differs; API refused");s.counts.beforeCall();
}
void FinalTailObserver::returned(const Buffer& b) {
 auto& s=*impl;tails::require(!s.srvOnly,"SRV-only weight return requires actual Device descriptor evidence");tails::require(b.storage&&b.words==s.current.at("words")&&b.logicalElements==s.current.at("logical_elements")&&b.packedBF16==(s.current.at("dtype")=="BF16"),"Final-tail returned Buffer descriptor differs");
 s.counts.APIreturned();
 if(s.current.at("tensor_name")=="model.language_model.embed_tokens.weight"){const auto i=s.current.at("chunk_index").get<uint32_t>();tails::require(i<s.counts.sources,"Final-tail initialized target lacks CPU snapshot");for(uint32_t k=0;k<i;++k)tails::require(s.targets[k].storage!=b.storage,"Final-tail unexpected target Storage alias");s.targets[i]=b;}
}
void FinalTailObserver::returned(Device& device,const Buffer& b) {
 auto& s=*impl;tails::require(bool(s.srvOnly),"SRV-only descriptor evidence requested on ordinary path");tails::require(b.storage&&b.words==s.current.at("words")&&b.logicalElements==s.current.at("logical_elements")&&b.packedBF16==(s.current.at("dtype")=="BF16"),"Final-tail returned Buffer descriptor differs");
 s.counts.APIreturned();
 if(s.current.at("tensor_name")=="model.language_model.embed_tokens.weight"){const auto i=s.current.at("chunk_index").get<uint32_t>();tails::require(i<s.counts.sources,"Final-tail initialized target lacks CPU snapshot");for(uint32_t k=0;k<i;++k)tails::require(s.targets[k].storage!=b.storage,"Final-tail unexpected target Storage alias");s.targets[i]=b;}
 const auto d=device.weightDescriptor(b);s.srvOnly->last={{"operation_ordinal",s.counts.returned},{"tensor_name",s.current.at("tensor_name")},{"chunk_index",s.current.at("chunk_index")},{"descriptor",weight_srv_only::descriptorJson(d)}};
 tails::require(weight_srv_only::matches(d,b.words),"Actual SRV-only weight descriptor or view differs");++s.srvOnly->checked;
 if(s.current.at("tensor_name")=="model.language_model.embed_tokens.weight")s.srvOnly->selected.push_back(s.srvOnly->last);
}
void FinalTailObserver::afterDrain(){auto& s=*impl;s.counts.drain(s.current.at("logical_storage_bytes").get<uint64_t>());}
void FinalTailObserver::afterImport(Device& device,const Weight& embedding,const Weight& head) {
 auto& s=*impl;s.constructorReturned=true;
 tails::require(!s.srvOnly||(s.srvOnly->checked==final_tails::maximumOperations&&s.srvOnly->selected.size()==tails::targetCount),"Final-tail SRV-only creation descriptor coverage incomplete");
 tails::require(s.prepared&&s.counts.drained==final_tails::maximumOperations&&s.counts.returned==final_tails::maximumOperations&&s.counts.sources==tails::targetCount&&embedding.rows==tails::vocabulary&&embedding.cols==tails::columns&&embedding.bf16&&head.rows==embedding.rows&&head.cols==embedding.cols&&head.bf16&&embedding.shards.size()==tails::targetCount&&head.shards.size()==tails::targetCount&&embedding.shardFirstRows.size()==tails::targetCount&&head.shardFirstRows==embedding.shardFirstRows,"Final-tail complete constructor/source/head topology differs");
 for(uint32_t i=0;i<tails::targetCount;++i){const auto t=tails::target(i);const auto& b=s.targets[i];
  tails::require(b.storage&&b.storage==embedding.shards[i].storage&&b.storage==head.shards[i].storage&&embedding.shardFirstRows[i]==t.firstRow&&b.words==t.words&&b.logicalElements==uint64_t(t.rows)*tails::columns&&b.packedBF16&&embedding.shards[i].words==t.words&&head.shards[i].words==t.words&&embedding.shards[i].logicalElements==b.logicalElements&&head.shards[i].logicalElements==b.logicalElements&&embedding.shards[i].packedBF16&&head.shards[i].packedBF16,"Final-tail original tied Storage/descriptor changed");}
 s.identitiesChecked=true;
 for(uint32_t i=0;i<tails::targetCount;++i){
  const auto t=tails::target(i);s.counts.copy(i);direct_head_row::Receipt receipt{};bool returned=false;
  try{
   const auto words=device.readWordsRange(s.targets[i],t.firstWord,tails::rowWords,receipt);s.lastTransport=Impl::transport(receipt);s.counts.copyReturned(receipt.copiesSubmitted);returned=true;
   tails::require(words.size()==tails::rowWords&&receipt.extent.firstByte==uint64_t(t.firstWord)*4&&receipt.extent.bytes==tails::rowBytes&&receipt.extent.logicalBytes==t.bytes&&receipt.extent.physicalBytes==t.bytes&&receipt.beforeCopyDrainCompleted&&receipt.afterCopyDrainCompleted&&receipt.stagingCreated&&receipt.stagingReleased&&receipt.mapAttempted&&receipt.readiness.result==readback::Result::copied&&receipt.unmapped,"Final-tail direct transport extent/completion differs");
   tails::require(!s.srvOnly||(receipt.sourceUsage==weight_srv_only::defaultUsage&&receipt.sourceBindFlags==weight_srv_only::shaderResourceBind&&receipt.sourceMiscFlags==weight_srv_only::rawMisc&&receipt.sourceCPUAccess==0&&receipt.sourceStride==0),"Final-tail direct source is not exact DEFAULT raw SRV-only");
   const bool equal=std::equal(words.begin(),words.end(),s.expected[i].begin());const auto sha=s.digest(words.data(),tails::rowBytes);tails::require(sha.size()==64,"Final-tail direct digest extent");
   s.counts.observed(equal);Json observed={{"shard_index",i},{"global_row",t.row},{"source_sha256",s.sources.at(i).at("sha256")},{"direct_sha256",sha},{"byte_equal_to_actual_API_source",equal},{"phase","after_full_import_constructor_return_before_arithmetic"},{"source_maps_released",true},{"matched_prefix_payload_offset",equal?Json(uint64_t(i)*tails::rowBytes):Json(nullptr)}};
   if(equal)std::copy(words.begin(),words.end(),s.good[i].begin());
   else{uint32_t n=0,first=UINT32_MAX,last=0;for(uint32_t k=0;k<tails::rowWords;++k)if(words[k]!=s.expected[i][k]){++n;first=std::min(first,k);last=k;}observed["different_u32_words"]=n;observed["first_different_word"]=first;observed["last_different_word"]=last;s.firstBad=observed;s.badAttempted=true;s.sink(finalTailBadFile,words.data(),tails::rowBytes);s.badWritten=true;}
   s.samples.push_back(observed);s.emit({{"event","post_import_direct_sample"},{"observation",observed},{"transport",s.lastTransport}});
   if(!equal){s.finish();return;}
  }catch(...){s.lastTransport=Impl::transport(receipt);if(!returned){tails::require(receipt.copiesSubmitted<=1,"Final-tail failed copy receipt exceeds one");s.counts.copiesSubmitted+=receipt.copiesSubmitted;}throw;}
 }
 s.finish();
}
void FinalTailObserver::failure(const std::string& reason)noexcept {
 auto& s=*impl;try{s.counts.refusal();s.complete=false;if(s.firstRefusal.is_null())s.firstRefusal={{"error",reason.substr(0,1024)},{"last_API_operation",s.current},{"counts",s.counterReport()},{"last_transport",s.lastTransport}};s.persist();if(!s.journalFailed&&s.records<final_tails::maximumRecords)s.emit({{"event","partial_or_refused"},{"first_refusal",s.firstRefusal}});}catch(...){s.persistenceFailed=true;}
}
void FinalTailObserver::releaseTargets(){for(auto& b:impl->targets)b={};impl->targetsReleased=true;}
bool FinalTailObserver::completed()const{return impl->complete&&impl->counts.terminal&&!impl->counts.refused;}
Json FinalTailObserver::report()const {
 const auto& s=*impl;
 Json result={{"schema","chandra.directcompute.final-import-tail-audit.v1"},{"weight_import_api",weight_import::name(s.api)},{"D3D11_usage","DEFAULT"},{"requested",true},{"complete",completed()},{"outcome",completed()?(s.counts.fault?"first_fault":"all_selected_tails_match"):"partial_or_refused"},{"selected_tails_match",completed()&&!s.counts.fault},
  {"counts",s.counterReport()},{"source_records",s.sources},{"post_import_samples",s.samples},{"first_bad",s.firstBad},{"first_refusal",s.firstRefusal},{"last_transport",s.lastTransport},{"constructor_return_observed",s.constructorReturned},{"same_original_embedding_head_Storage_checked",s.identitiesChecked},{"target_references_released",s.targetsReleased},
  {"source_raw_write_attempted",s.sourceAttempted},{"source_raw_written_and_flushed",s.sourceWritten},{"source_payload_bytes",s.sourceWritten?uint64_t(s.counts.sources)*tails::rowBytes:0},{"matched_prefix_raw_write_attempted",s.goodAttempted},{"matched_prefix_raw_written_and_flushed",s.goodWritten},{"matched_prefix_payload_bytes",s.goodWritten?uint64_t(s.counts.matched)*tails::rowBytes:0},{"first_bad_raw_write_attempted",s.badAttempted},{"first_bad_raw_written_and_flushed",s.badWritten},
  {"journal_records_written",s.records},{"journal_write_attempts",s.writeAttempts},{"journal_write_failed_or_torn",s.journalFailed},{"failure_evidence_persistence_failed",s.persistenceFailed},{"early_GPU_observations",0},{"arithmetic_gate_eligible",false},{"whole_model_byte_integrity_accepted",false},{"root_physical_retirement_accepted",false},{"request_replayed",false}};
 if(s.srvOnly)result["weight_srv_only"]={{"contract",weight_srv_only::contract()},{"descriptors_checked",s.srvOnly->checked},{"all_returned_descriptors_checked",s.srvOnly->checked==final_tails::maximumOperations},{"selected_target_creation_descriptors",s.srvOnly->selected},{"last_actual_descriptor",s.srvOnly->last}};
 return result;
}
Json runFinalTailImport(Device& device,const std::wstring& directory,WeightImportAPI api,FinalTailObserver& observer) {return runFinalTailImport(device,directory,api,observer,false);}
Json runFinalTailImport(Device& device,const std::wstring& directory,WeightImportAPI api,FinalTailObserver& observer,bool weightSrvOnly) {
 Json result={{"schema","chandra.directcompute.final-import-tail-result.v1"},{"weight_import_api",weight_import::name(api)},{"D3D11_usage","DEFAULT"},{"passed",false},{"import_only",true},{"actual_inference",false},{"complete_inference",false},{"qualified_full_graph",false},{"qualified_OCR",false},{"performance_claim",false},{"request_replayed",false}};std::string error;
 if(weightSrvOnly)result["weight_srv_only_requested"]=true;
 try{
  tails::require(observer.importAPI()==api&&observer.weightSrvOnly()==weightSrvOnly&&(!weightSrvOnly||api==WeightImportAPI::defaultInitial),"Final-tail run API/SRV-only selector pair differs");
  {ModelWeights weights(device,directory,12ull*1024*1024*1024,{},WeightStorageExperiment::exact,nullptr,nullptr,nullptr,nullptr,api,&observer,weightSrvOnly);
   const auto provenance=Json::parse(weights.provenanceJson());Json summary=Json::object();
   for(const auto* key:{"model","revision","model_bytes","model_sha256","config_bytes","config_sha256","tensor_count","full_graph_requested","uploaded_tensor_names","tracked_buffer_bytes_after","uploaded_storage_bytes","tie_byte_equality","mapping_lifetime","weight_import_api","API_operations_completed","UpdateSubresource_calls","DEFAULT_initial_data_Device_creations"})summary[key]=provenance.at(key);
   if(weightSrvOnly)summary["weight_srv_only"]=provenance.at("weight_srv_only");
   result["model_import_completed"]=true;result["model_provenance"]=summary;
   observer.afterImport(device,weights.at("model.language_model.embed_tokens.weight"),weights.at("lm_head.weight"));}
 }catch(const std::exception& e){error=e.what();observer.failure(error);}catch(...){error="Non-standard final-import-tail exception";observer.failure(error);}
 observer.releaseTargets();result["final_import_tail_audit"]=observer.report();result["diagnostic_completed"]=observer.completed();result["outcome"]=observer.report().at("outcome");
 if(!error.empty())result["error"]=error;
 try{device.drain();result["final_drain_completed"]=true;result["owned_buffer_zero"]=device.trackedBufferBytes()==0;result["final_memory"]=Json::parse(device.memoryJson());}
 catch(const std::exception& e){result["final_drain_completed"]=false;result["final_drain_error"]=e.what();}
 result["passed"]=observer.completed()&&error.empty()&&result.value("final_drain_completed",false)&&result.value("owned_buffer_zero",false);
 result["passed_scope"]="Bounded import observation and local owned-buffer drain only; a completed byte fault is not model correctness or physical Job retirement";
 return result;
}
} // namespace chandra::dc::import_row
#endif
