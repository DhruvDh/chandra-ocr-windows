// New code, MPL-2.0. Optional bounded raw dumps of completed native graph values.
// Inactive unless the inference CLI requests --diagnostic-dump. Recorder callbacks read existing
// completed buffers synchronously on the Device thread; they never write, retain or dispatch.
// A dump is unqualified evidence: comparison against a declared reference is separate work.
#pragma once
#include "text_model.h"
#include "vision_model.h"
#include "../vendor/nlohmann/json.hpp"
#include <array>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace chandra::dc::diagnostics {
using Json=nlohmann::json;
constexpr const char* planSchema="chandra.directcompute.diagnostic-plan.v1";
constexpr const char* progressSchema="chandra.directcompute.diagnostic-progress.v2";
constexpr const char* recordSchema="chandra.directcompute.diagnostic-record.v2";
// Canonical consumed-prefix stream: "<schema>\ninput_manifest_sha256 <hex>\nprompt_rows <P>\n" followed by
// one "<absolute_row> <token_id> <t> <h> <w>\n" line per consumed request row in order. prefix_sha256(n)
// is the SHA-256 of the header and the first n row lines; scripts/native/compare_diagnostics.py recomputes it.
constexpr const char* prefixSchema="chandra.directcompute.consumed-prefix.v1";
constexpr uint64_t defaultByteLimit=64ull<<20, maximumByteLimit=256ull<<20;
constexpr uint64_t maximumRecords=4096, maximumReadbackBytes=8ull<<30, maximumSingleReadbackBytes=128ull<<20;
constexpr uint32_t maximumSelectedRows=64, maximumDecodeSteps=64, maximumStateSelections=16, maximumStateLengths=16;
constexpr uint32_t defaultDecodeSteps=8, textWidth=2560, visionWidth=1024, vocabulary=248320;

class Sha256 {
    uint32_t state[8]; uint8_t block[64]; uint64_t bytes=0; size_t used=0;
    void compress(const uint8_t*);
public:
    Sha256();
    void update(const void*,size_t);
    std::string finish() const; // Lowercase hex; the running state remains usable.
};
std::string sha256(const void*,size_t);

// Authenticated CPU input facts used to validate selections before any Device exists.
struct Geometry {
    uint32_t patchRows=0, mergedRows=0, generationLimit=0;
    std::vector<VisionGrid> grids;
    std::vector<uint32_t> ids, visionRowMap; // visionRowMap is UINT32_MAX for text rows.
    TextPositions positions;
};
struct StateSelection { uint32_t layer=0; bool conv=false; std::vector<uint32_t> heads, cacheLengths; };
struct Plan {
    std::vector<uint32_t> visionRows, visionBlocks, mergerRows, embeddingRows, prefillRows, prefillLayers, decodeSteps, decodeLayers;
    bool patchEmbedding=true, positionAdded=true, prefillFinalNorm=true, prefillLogits=true, decodeFinalNorm=true, decodeLogits=true;
    std::vector<StateSelection> states;
    uint64_t byteLimit=defaultByteLimit, records=0, payloadBytes=0, readbacks=0, readbackBytes=0, largestReadbackBytes=0;
    std::set<std::string> keys; // Exact identity of every planned record.
    Json resolved; std::string sha256;
};
// Resolves an explicit plan object, or safe small defaults for nullptr, against the authenticated
// geometry. Throws on unknown keys, out-of-range or duplicate selections and any budget breach.
Plan resolve(const Json* requested,const Geometry&);

// Exclusive output creation. createNew must fail if the name already exists.
struct File { virtual ~File()=default; virtual void write(const void*,size_t)=0; virtual void flush()=0; };
struct Directory { virtual ~Directory()=default; virtual std::unique_ptr<File> createNew(const std::string& name)=0; };
bool safeName(const std::string&);
using Reader=std::function<std::vector<float>(const Buffer&)>;

// Writes progress.jsonl incrementally plus one raw little-endian FP32 payload per selected row.
// Any write, budget or provenance failure throws, so the caller's existing request poisoning
// and retirement apply. finish() appends the terminal status; dumps always remain unqualified.
// Every text observation, selected or not, passes through one request-call state machine that
// advances the consumed prefix exactly once per prefill/decode call, so each text record commits to
// its complete conditioning prefix rather than to the current token or the producer's tiling.
class Recorder {
public:
    struct Spec;
    Recorder(Plan,Geometry,std::unique_ptr<Directory>,Reader,Json commitments,Json producer);
    ~Recorder();
    Recorder(const Recorder&)=delete; Recorder& operator=(const Recorder&)=delete;
    void authenticatedModel(const Json& provenance,const Json& device);
    void boundary(const std::string& name);
    void vision(const std::string& name,const Buffer&,uint32_t rows,uint32_t width);
    void merged(const Buffer&,uint32_t rows,uint32_t width);
    void text(const TextObservation&);
    Json finish(bool runCompleted,const std::string& error);
private:
    Plan plan; Geometry geometry; std::unique_ptr<Directory> directory; Reader reader;
    Json commitments, producer; std::unique_ptr<File> progress; Sha256 progressHash;
    std::set<std::string> written; std::string started;
    uint64_t sequence=0, payloadBytes=0, readbacks=0, readbackBytes=0;
    bool modelCommitted=false, broken=false, finished=false; Json summary;
    // Consumed request rows (token, t, h, w), prefix digests indexed by row count, and rows already
    // written to conditioning lines. call identifies the current graph call for repeated callbacks.
    Sha256 prefixHash; std::vector<std::array<uint32_t,4>> consumed; std::vector<std::string> prefixDigests; uint64_t conditioned=0;
    struct Call { bool active=false; uint32_t before=0, generated=0, tokens=0; } call;
    void enter(const TextObservation&,bool decode);
    void condition();
    void line(const Json&);
    std::vector<float> read(const Buffer&,uint64_t elements);
    void put(const Spec&,const float*,size_t);
};
}
