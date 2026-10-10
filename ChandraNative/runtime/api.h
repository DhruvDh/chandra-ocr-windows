// New code, MPL-2.0. DirectCompute graph contract; GPU execution stays D3D11.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "weight_storage_experiment.h"
#include "direct_head_row.h"
#include "weight_import_api.h"
#include "weight_srv_only.h"
namespace chandra::dc {
namespace import_row { class Observer; class EarlyObserver; class WatchObserver; class Shard1Observer; class FinalTailObserver; }
struct Storage;
struct Buffer { std::shared_ptr<Storage> storage; uint32_t words=0; uint64_t logicalElements=0; bool packedBF16=false; };
struct Weight { std::vector<Buffer> shards; std::vector<uint32_t> shardFirstRows; uint32_t rows=0,cols=0; bool bf16=true; };
class Device {
 struct Impl; std::unique_ptr<Impl> impl;
public:
 Device(const std::wstring& shaderDirectory,const std::string& pci,const std::string& luid);
 ~Device(); Device(const Device&)=delete; Device& operator=(const Device&)=delete;
 Buffer floats(uint32_t count,const float* initial=nullptr);
 Buffer words(uint32_t count,const uint32_t* initial=nullptr);
 // Explicit weight-only experiment: initialized DEFAULT raw SRV, no UAV and no subsequent writes.
 Buffer srvOnlyWeightWords(uint32_t count,const void* initial);
 weight_srv_only::Descriptor weightDescriptor(const Buffer&) const;
 // Weight-only experiment: views and transfers expose count words; physicalBytes is accounted.
 Buffer paddedWeightWords(uint32_t count,uint32_t physicalBytes);
 void upload(Buffer&,const void*,uint32_t bytes);
 void dispatch(const std::string& shader,const std::vector<const Buffer*>& inputs,const std::vector<Buffer*>& outputs,const void* parameters,uint32_t parameterBytes,uint32_t x,uint32_t y=1,uint32_t z=1);
 std::vector<float> readFloats(const Buffer&);
 std::vector<uint32_t> readWords(const Buffer&);
 // Opt-in observer: one bounded underlying-buffer copy, never a shader/SRV read.
 std::vector<uint32_t> readWordsRange(const Buffer&,uint32_t firstWord,uint32_t count,direct_head_row::Receipt&);
 void zero(Buffer&); void drain(uint32_t timeoutMilliseconds=10000);
 // Optional experimental-route cancellation only. Sticky; never acknowledges a
 // completed drain or releases Storage. May be called from another thread while
 // this Device stays alive; caller must join that thread before destroying it.
 // Existing worker cancellation/safe boundaries are unchanged and do not call it.
 void cancelExperimentalDrainWait() noexcept;
 uint64_t trackedBufferBytes() const; std::string identityJson() const;
 std::string memoryJson() const;
 void beginProfile(); std::string finishProfile();
};
class ModelWeights {
 struct Impl; std::unique_ptr<Impl> impl;
public:
 ModelWeights(Device&,const std::wstring& modelDirectory,uint64_t maximumWeightBytes=12ull*1024*1024*1024,const std::vector<std::string>& selectedTensorNames={});
 ModelWeights(Device&,const std::wstring& modelDirectory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment);
 ModelWeights(Device&,const std::wstring& modelDirectory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment,import_row::Observer*);
 ModelWeights(Device&,const std::wstring& modelDirectory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment,import_row::Observer*,import_row::EarlyObserver*);
 ModelWeights(Device&,const std::wstring& modelDirectory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment,import_row::Observer*,import_row::EarlyObserver*,import_row::WatchObserver*);
 ModelWeights(Device&,const std::wstring& modelDirectory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment,import_row::Observer*,import_row::EarlyObserver*,import_row::WatchObserver*,import_row::Shard1Observer*);
 ModelWeights(Device&,const std::wstring& modelDirectory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment,import_row::Observer*,import_row::EarlyObserver*,import_row::WatchObserver*,import_row::Shard1Observer*,WeightImportAPI,import_row::FinalTailObserver*);
 ModelWeights(Device&,const std::wstring& modelDirectory,uint64_t maximumWeightBytes,const std::vector<std::string>& selectedTensorNames,WeightStorageExperiment,import_row::Observer*,import_row::EarlyObserver*,import_row::WatchObserver*,import_row::Shard1Observer*,WeightImportAPI,import_row::FinalTailObserver*,bool weightSrvOnly);
 ~ModelWeights(); ModelWeights(const ModelWeights&)=delete;
 const Weight& at(const std::string&) const;
 bool contains(const std::string&) const;
 std::string provenanceJson() const;
};
// Activations are R32_FLOAT buffers. BF16 boundaries are explicit round-to-nearest-even
// within shader operators; weights stay packed BF16. Recurrent/normalization reductions
// remain FP32. No precision policy is inferred from storage alone.
Buffer linear(Device&,const Buffer& input,const Weight&,uint32_t batchRows,bool roundOutputBF16=true,const Weight* bias=nullptr);
Buffer embedding(Device&,const Weight&,const std::vector<uint32_t>& tokenIds);
Buffer rmsNorm(Device&,const Buffer&,const Weight&,uint32_t rows,uint32_t width,float epsilon,bool onePlusWeight,bool roundOutputBF16=true);
Buffer layerNorm(Device&,const Buffer&,const Weight& scale,const Weight& bias,uint32_t rows,uint32_t width,float epsilon,bool roundOutputBF16=true);
Buffer binary(Device&,const Buffer&,const Buffer&,uint32_t count,uint32_t operation,bool roundOutputBF16=true); // RHS broadcast if its logicalElements divides count. 0 add, 1 multiply, 2 silu(first)*second, 3 gelu_tanh(first)*second, 4 gelu_erf(first)*second
Buffer unary(Device&,const Buffer&,uint32_t count,uint32_t operation,bool roundOutputBF16=true); // 0 silu, 1 gelu_tanh, 2 sigmoid, 3 BF16 rounding, 4 gelu_erf
}
