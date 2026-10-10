// New code, MPL-2.0. Opt-in weight allocation experiment; no arithmetic or shader change.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
namespace chandra::dc {
enum class WeightStorageExperiment { exact, page4096 };
namespace weight_storage {
constexpr uint64_t maximumBufferBytes = 128ull * 1024 * 1024;
inline const char* name(WeightStorageExperiment mode) {
    switch (mode) {
    case WeightStorageExperiment::exact: return "exact";
    case WeightStorageExperiment::page4096: return "page4096";
    }
    throw std::invalid_argument("Unknown weight storage experiment");
}
inline WeightStorageExperiment parse(const std::string& value) {
    if (value == "exact") return WeightStorageExperiment::exact;
    if (value == "page4096") return WeightStorageExperiment::page4096;
    throw std::invalid_argument("--weight-storage-experiment requires exact or page4096");
}
inline uint64_t allocationBytes(uint64_t logicalBytes, WeightStorageExperiment mode) {
    if (!logicalBytes || logicalBytes % 4 || logicalBytes > maximumBufferBytes)
        throw std::invalid_argument("Weight storage must be nonempty, whole words and at most 128 MiB");
    (void)name(mode); // Refuse a malformed enum even when the size is already page-aligned.
    if (mode == WeightStorageExperiment::exact) return logicalBytes;
    // Bounds are checked before addition; the 128 MiB ceiling is itself page-aligned.
    const uint64_t rounded = (logicalBytes + 4095) / 4096 * 4096;
    if (rounded > maximumBufferBytes) throw std::invalid_argument("Padded weight exceeds 128 MiB");
    return rounded;
}
struct Geometry {
    uint32_t logicalWords, logicalBytes, physicalBytes;
    bool boxed() const { return physicalBytes != logicalBytes; }
    // D3D11_BOX coordinates are byte coordinates for buffers; both prefix transfers use these.
    template<class Box> Box prefix() const { return Box{0, 0, 0, logicalBytes, 1, 1}; }
};
inline Geometry geometry(uint32_t logicalWords, uint32_t physicalBytes = 0) {
    const uint64_t logicalBytes = uint64_t(logicalWords) * 4;
    if (!logicalWords || logicalBytes > maximumBufferBytes)
        throw std::invalid_argument("D3D11 buffer must be nonempty and at most 128 MiB");
    if (!physicalBytes) physicalBytes = static_cast<uint32_t>(logicalBytes);
    if (physicalBytes < logicalBytes || physicalBytes % 4 || physicalBytes > maximumBufferBytes)
        throw std::invalid_argument("Physical buffer extent must contain its exact logical words within 128 MiB");
    return {logicalWords, static_cast<uint32_t>(logicalBytes), physicalBytes};
}
inline bool withinBudget(uint64_t live, uint64_t additional, uint64_t ceiling) {
    return live <= ceiling && additional <= ceiling - live;
}
} // namespace weight_storage
// Shape-only forecast; no file, model, Device or GPU access. Original MTP omission and head sharing apply.
std::string weightStorageForecastJson(WeightStorageExperiment);
} // namespace chandra::dc
