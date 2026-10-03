# Current work

Started October 3, 2026. The commissioned outcome is a correctness-validated Chandra OCR 2 endpoint on Windows Intel GPU, selectable or load-balanced with the existing ROCm endpoint. This record distinguishes implemented components from deployment acceptance.

The public repository and upstream history are established. Waystone's Arc A770 was enumerated through Vulkan and DXGI; it has 15.88 GiB dedicated memory and an observed process budget of 15.13 GiB. Its existing Python 3.12.14, uv, Visual Studio 2022, MSVC and Windows SDK provide a native installation route. These observations do not establish model compatibility or current free GPU capacity.

Work is divided into the benchmark and numerical gates under `benchmarks/`, the exact-model XPU reference under `runtime/waystone/`, native DirectCompute operator work under `ChandraNative/`, and the HTTP worker/router under `chandra_service/`. Builds and installations are project-local. GPU work on each machine has one coordinating owner to avoid overlapping model allocations.

Full-model Intel inference, numerical acceptance, repeated OCR benchmarks and deployed routing remain in progress. Do not advertise a native Chandra graph or a validated Windows OCR service until the corresponding retained results exist. The next acceptance boundary is one complete synthetic OCR page on the explicitly selected Intel GPU, followed by representative pages and repeated-request correctness.
