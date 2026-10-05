// New ChandraNative code, MPL-2.0. Read-only pinned safetensors import.
// Model weights remain governed by the model's own supplied license.
#include "api.h"
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
struct Shard { uint32_t firstRow, rows, words; uint64_t elements, sourceBytes, storageBytes; };
std::vector<Shard> shardPlan(const Tensor& t) {
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
        result.push_back({static_cast<uint32_t>(row),static_cast<uint32_t>(count),static_cast<uint32_t>(words),elements,multiply(elements,t.bf16?2:4),storage}); row+=count;
    }
    return result;
}
uint64_t readU64LE(const uint8_t* bytes) { uint64_t n=0; for(unsigned i=0;i<8;++i)n|=uint64_t(bytes[i])<<(8*i); return n; }
} // namespace chandra::dc::weight_detail

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
    public:
        View(HANDLE mapping,uint32_t granularity,uint64_t offset,uint64_t bytes) {
            require(bytes>0 && bytes<=maximumShardBytes,"mapping view exceeds bounded shard");
            uint64_t aligned=offset-offset%granularity, skip=offset-aligned, size=add(skip,bytes);
            require(size<=SIZE_MAX,"mapping view size overflow");
            base=static_cast<const uint8_t*>(MapViewOfFile(mapping,FILE_MAP_READ,static_cast<DWORD>(aligned>>32),static_cast<DWORD>(aligned),static_cast<SIZE_T>(size)));
            require(base!=nullptr,"read-only model mapping view failed"); data=base+skip;
        }
        ~View(){if(base)UnmapViewOfFile(base);}
        View(const View&)=delete; View& operator=(const View&)=delete;
        const uint8_t* get() const { return data; }
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
ModelWeights::ModelWeights(Device& device,const std::wstring& directory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames) : impl(std::make_unique<Impl>()) {
    using namespace weight_detail;
    require(maximumWeightBytes>0 && maximumWeightBytes<=defaultBudgetBytes,"weight budget must be positive and at most 12 GiB");
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
        for(const auto& shard:shardPlan(t))requiredBytes=add(requiredBytes,shard.storageBytes);
    }
    uint64_t baseline=device.trackedBufferBytes();require(baseline<=maximumWeightBytes && requiredBytes<=maximumWeightBytes-baseline,"global tracked buffer budget would exceed weight admission");
    Json inventory=Json::array(),skipped=Json::array(),omitted=Json::array();uint64_t uploaded=0;
    for(const auto& item:tensors){
        const auto& t=item.second;
        Json row={{"name",t.name},{"shape",t.shape},{"dtype",t.dtype},{"data_offsets",{t.begin,t.end}},{"logical_elements",t.elements},{"rows",t.rows},{"cols",t.cols}};
        if(t.name.rfind("mtp.",0)==0){row["reason"]="unused MTP/speculative head; original graph does not execute it";skipped.push_back(std::move(row));continue;}
        if(!selection.effective.count(t.name)){row["reason"]="graph tensor omitted from explicitly selected trained-layer diagnostic";omitted.push_back(std::move(row));continue;}
        if(t.name=="lm_head.weight"){row["shared_with"]="model.language_model.embed_tokens.weight";inventory.push_back(std::move(row));continue;}
        Weight weight;weight.rows=t.rows;weight.cols=t.cols;weight.bf16=t.bf16;Json shards=Json::array();
        uint64_t consumed=0;
        for(const auto& shard:shardPlan(t)){
            require(device.trackedBufferBytes()<=maximumWeightBytes && shard.storageBytes<=maximumWeightBytes-device.trackedBufferBytes(),"tracked weight budget exhausted before upload");
            Buffer buffer=t.bf16?device.words(shard.words):device.floats(shard.words);
            buffer.logicalElements=shard.elements;buffer.packedBF16=t.bf16;
            auto source=model.view(add(payloadBegin,add(t.begin,consumed)),shard.sourceBytes);
            if(shard.storageBytes!=shard.sourceBytes){
                // At most one bounded shard scratch; zero high halfword explicitly.
                std::vector<uint32_t> padded(shard.words,0);std::memcpy(padded.data(),source.get(),static_cast<size_t>(shard.sourceBytes));device.upload(buffer,padded.data(),static_cast<uint32_t>(shard.storageBytes));
            }else device.upload(buffer,source.get(),static_cast<uint32_t>(shard.storageBytes));
            device.drain(); // Bound driver upload staging to one authenticated shard, not the whole checkpoint.
            require(device.trackedBufferBytes()<=maximumWeightBytes,"tracked weight budget exceeded after allocation");
            weight.shardFirstRows.push_back(shard.firstRow);weight.shards.push_back(std::move(buffer));consumed=add(consumed,shard.sourceBytes);uploaded=add(uploaded,shard.storageBytes);
            shards.push_back({{"first_row",shard.firstRow},{"rows",shard.rows},{"source_bytes",shard.sourceBytes},{"storage_bytes",shard.storageBytes},{"words",shard.words}});
        }
        require(consumed==t.end-t.begin,"upload did not consume complete tensor");row["shards"]=std::move(shards);inventory.push_back(std::move(row));impl->weights.emplace(t.name,std::move(weight));
    }
    // Copying Weight shares Buffer::storage; no second GPU allocation.
    if(tieSelected)impl->weights.emplace("lm_head.weight",impl->weights.at("model.language_model.embed_tokens.weight"));
    device.drain(); // Producer completion precedes releasing every mapped source view.
    impl->provenance={{"schema","chandra.directcompute.model-weights.v2"},{"model","datalab-to/chandra-ocr-2"},{"revision",revision},{"model_bytes",model.size()},{"model_sha256",modelHash},{"config_bytes",config.size()},{"config_sha256",configHash},{"tensor_count",tensors.size()},{"full_graph_requested",selection.fullGraph},{"requested_tensor_names",selection.requested},{"effective_tensor_names",std::vector<std::string>(selection.effective.begin(),selection.effective.end())},{"uploaded_tensor_names",impl->weights.size()},{"maximum_shard_bytes",maximumShardBytes},{"maximum_tracked_weight_bytes",maximumWeightBytes},{"tracked_buffer_bytes_before",baseline},{"uploaded_storage_bytes",uploaded},{"tracked_buffer_bytes_after",device.trackedBufferBytes()},{"tie_byte_equality_performed",tieSelected},{"tie_byte_equality",tieSelected},{"tie_config_flag_used_as_proof",false},{"tied_alias_selection_expanded",selection.tiedAliasExpansion},{"inventory",std::move(inventory)},{"omitted_graph_tensors",std::move(omitted)},{"skipped_mtp_bytes",skippedBytes},{"skipped_mtp",std::move(skipped)},{"mapping_lifetime","all read-only views/handles released on constructor return after upload/drain"},{"implicit_cast_or_dequantization",false},{"numerical_acceptance",false},{"full_model_qualified",false}};
}
ModelWeights::~ModelWeights()=default;
const Weight& ModelWeights::at(const std::string& name) const {
    auto it=impl->weights.find(name);if(it==impl->weights.end())throw std::out_of_range("DirectCompute graph weight not imported: "+name);return it->second;
}
bool ModelWeights::contains(const std::string& name) const {return impl->weights.find(name)!=impl->weights.end();}
std::string ModelWeights::provenanceJson() const {return impl->provenance.dump();}
} // namespace chandra::dc
#endif
