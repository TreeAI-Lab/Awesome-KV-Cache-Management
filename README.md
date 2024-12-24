# Awesome-KV-Cache-Management

This repo aims to record a survey on LLM acceleration based on KV Cache Management.

- [Awesome-KV-Cache-Management](#awesome-kv-cache-management)
    - [Token-level Optimization](#token-level-optimization)
        - [KV Cache Selection](#kv-cache-selection)
        - [KV Cache Budget Allocation](#kv-cache-budget-allocation)
        - [KV Cache Merging](#kv-cache-merging)
        - [KV Cache Quantization](#kv-cache-quantization)
        - [KV Cache Low-rank Decomposition](#kv-cache-low-rank-decomposition)
    - [Model-level Optimization](#model-level-optimization)
        - [Intra later](#intra-later)
        - [Cross Layer](#cross-layer)
        - [Non-transformer Architecture](#non-transformer-architecture)
    - [System-level Optimization](#system-level-optimization)
        - [Memory Management](#memory-management)
        - [Scheduling](#scheduling)
        - [Hardware-aware Design](#hardware-aware-design)
    - [Datasets and Benchmarks](#datasets-and-benchmarks)
        - [Text Datasets](#text-datasets)
        - [Multi-modal Datasets](#multi-modal-datasets)

---

## Token-level Optimization

### KV Cache Selection

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | :--: | :---: | ----- | ---- |
|| FastGen | Static KV Cache Selection ||||
|| SnapKV | Static KV Cache Selection ||||
|| Attention-Gate | Static KV Cache Selection ||||
|| H2O | Dynamic Selection with Permanent Eviction ||||
|| BUZZ | Dynamic Selection with Permanent Eviction ||||
|| NACL | Dynamic Selection with Permanent Eviction ||||
|| Scissorhands | Dynamic Selection with Permanent Eviction ||||
|| Keyformer | Dynamic Selection with Permanent Eviction ||||
|| InfLLM | Dynamic Selection without Permanent Eviction ||||
|| Quest | Dynamic Selection without Permanent Eviction ||||
|| PQCache | Dynamic Selection without Permanent Eviction ||||
|| SqueezedAttention | Dynamic Selection without Permanent Eviction ||||
|| RetrievalAttention | Dynamic Selection without Permanent Eviction ||||
|| EM-LLM             | Dynamic Selection without Permanent Eviction ||||

### KV Cache Budget Allocation

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | :--: | ----- | ----- | ---- |
||PyramidKV|Layer-wise Budget Allocation||||
||PyrimidInfer|Layer-wise Budget Allocation||||
||DynamicKV|Layer-wise Budget Allocation||||
||PrefixKV|Layer-wise Budget Allocation||||
||SimLayerKV|Layer-wise Budget Allocation||||
||AdaKV|Head-wise Budget Allocation||||
||CriticalKV|Head-wise Budget Allocation||||
||LeanKV|Head-wise Budget Allocation||||
||RazorAttention|Head-wise Budget Allocation||||
||HeadKV|Head-wise Budget Allocation||||
||DuoAttention|Head-wise Budget Allocation||||

### KV Cache Merging

| Year | Title | Type | Venue | Paper | code |
| ---- | :---: | ---- | ----- | ----- | ---- |
||CCM|Intra-layer Merging||||
||CaM|Intra-layer Merging||||
||D2O|Intra-layer Merging||||
||AIM|Intra-layer Merging||||
||Look-M|Intra-layer Merging||||
||KVMerger|Intra-layer Merging||||
||MiniCache|Cross-layer Merging||||
|| KVSharer  |Cross-layer Merging||||

### KV Cache Quantization

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
||ZeroQuant|Fixed-precision Quantization||||
||FlexGen|Fixed-precision Quantization||||
||QJL|Fixed-precision Quantization||||
||PQCache|Fixed-precision Quantization||||
||KVQuant|Mixed-precision Quantization||||
||IntactKV|Mixed-precision Quantization||||
||SKVQ|Mixed-precision Quantization||||
||KIVI|Mixed-precision Quantization||||
||WKVQuant|Mixed-precision Quantization||||
||GEAR|Mixed-precision Quantization||||
||MiKV|Mixed-precision Quantization||||
||ZIPVL|Mixed-precision Quantization||||
||ZipCache|Mixed-precision Quantization||||
||PrefixQuant|Mixed-precision Quantization||||
||MiniKV|Mixed-precision Quantization||||
||MassiveActivation,|Outlier Redistribution||||
||QuaRot,|Outlier Redistribution||||
||Qserve,|Outlier Redistribution||||
||Q-INT4,|Outlier Redistribution||||
||SpinQuant,|Outlier Redistribution||||
||DuQuant,|Outlier Redistribution||||
||SmoothQuant,|Outlier Redistribution||||
||OS+,|Outlier Redistribution||||
||AffineQuant,|Outlier Redistribution||||
||FlatQuant,|Outlier Redistribution||||
||AWQ,|Outlier Redistribution||||
||OmniQuant|Outlier Redistribution||||

### KV Cache Low-rank Decomposition

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | :--: | ----- | ----- | ---- |
||ECKVH|Singular Value Decomposition||||
||EigenAttention|Singular Value Decomposition||||
||ZDC|Singular Value Decomposition||||
||LoRC|Singular Value Decomposition||||
||ShadowKV|Singular Value Decomposition||||
||Palu|Singular Value Decomposition||||
||DecoQuant|Tensor Decomposition||||
||LESS|Learned Low-rank Approximation||||

---

## Model-level Optimization

### Intra later

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2019 | MQA                     | Grouped Attention        |      |      |      |
| 2023 | GQA                     | Grouped Attention        |      |      |      |
| 2024 | AsymGQA                 | Grouped Attention        |      |      |      |
| 2024 | Weighted GQA            | Grouped Attention        |      |      |      |
| 2024 | QCQA                    | Grouped Attention        |      |      |      |
| 2024 | KDGQA                   | Grouped Attention        |      |      |      |
| 2023 | GQKVA                   | Grouped Attention        |      |      |      |
| 2024 | MLA                     | Compression              |      |      |      |
| 2024 | MatryoshkaKV            | Compression              |      |      |      |
| 2022 | FLASH                   | Extended Mechanism       |      |      |      |
| 2024 | Infini-Attention        | Extended Mechanism       |      |      |      |

### Cross Layer 

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | CLA                     | Cache Sharing            |      |      |      |
| 2024 | LCKV                    | Cache Sharing            |      |      |      |
| 2024 | SA                      | Cache Sharing            |      |      |      |
| 2024 | DHA                     | Cache Sharing            |      |      |      |
| 2024 | MLKV                    | Cache Sharing            |      |      |      |
| 2024 | Wu et al.               | Cache Sharing            |      |      |      |
| 2024 | LISA                    | Cache Sharing            |      |      |      |
| 2024 | SVFormer                | Cache Sharing            |      |      |      |
| 2024 | YOCO                    | Compression              |      |      |      |
| 2024 | CLLA                    | Compression              |      |      |      |
| 2024 | CEPE                    | Augmented Architectures  |      |      |      |
| 2024 | XC-Cache                | Augmented Architectures  |      |      |      |
| 2024 | Block Transformer       | Augmented Architectures  |      |      |      |

### Non-transformer Architecture

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2023 | RWKV                    | New Architecture         |      |      |      |
| 2024 | Mamba                   | New Architecture         |      |      |      |
| 2023 | RetNet                  | New Architecture         |      |      |      |
| 2024 | MCSD                    | New Architecture         |      |      |      |
| 2024 | MixCon                  | Hybrid Architecture      |      |      |      |
| 2024 | GoldFinch               | Hybrid Architecture      |      |      |      |
| 2024 | RecurFormer             | Hybrid Architecture      |      |      |      |

---

## System-level Optimization

### Memory Management

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2023 | vLLM                    | Architectural Design      |      |      |      |
| 2024 | vTensor                 | Architectural Design      |      |      |      |
| 2024 | LeanKV                  | Architectural Design      |      |      |      |
| 2024 | ChunkAttention          | Prefix-aware Design       |      |      |      |
| 2024 | MemServe                | Prefix-aware Design       |      |      |      |



### Scheduling

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | BatchLLM                | Prefix-aware Scheduling   |      |      |      |
| 2024 | RadixAttention          | Prefix-aware Scheduling   |      |      |      |
| 2024 | FastServe               | Preemptive and Fairness-oriented Scheduling |      |      |      |
| 2024 | FastSwitch              | Preemptive and Fairness-oriented Scheduling |      |      |      |
| 2024 | LayerKV                 | Layer-specific and Hierarchical Scheduling  |      |      |      |
| 2024 | CachedAttention         | Layer-specific and Hierarchical Scheduling  |      |      |      |
| 2024 | ALISA                   | Layer-specific and Hierarchical Scheduling  |      |      |      |
| 2024 | LAMPS                   | Layer-specific and Hierarchical Scheduling  |      |      |      |


### Hardware-aware Design

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
 2024 | HydraGen                | Single/Multi-GPU Design   |      |      |      |
| 2024 | DeFT                   | Single/Multi-GPU Design   |      |      |      |
| 2023 | vLLM                   | Single/Multi-GPU Design   |      |      |      |
| 2022 | ORCA                   | Single/Multi-GPU Design   |      |      |      |
| 2024 | DistServe              | Single/Multi-GPU Design   |      |      |      |
| 2024 | Multi-Bin Batching     | Single/Multi-GPU Design   |      |      |      |
| 2024 | Tree Attention         | Single/Multi-GPU Design   |      |      |      |
| 2022 | FlashAttention          | I/O-based Design         |      |      |      |
| 2024 | Bifurcated Attention    | I/O-based Design         |      |      |      |
| 2024 | PartKVRec               | I/O-based Design         |      |      |      |
| 2024 | HCache                  | I/O-based Design         |      |      |      |
| 2024 | Cake                    | I/O-based Design         |      |      |      |
| 2024 | FastSwitch              | I/O-based Design         |      |      |      |
| 2024 | NEO                     | Heterogeneous Design      |      |      |      |
| 2024 | FastDecode              | Heterogeneous Design      |      |      |      |
| 2024 | FlexInfer               | Heterogeneous Design      |      |      |      |
| 2024 | InfiniGen               | Heterogeneous Design      |      |      |      |
| 2023 | Pensieve                | Heterogeneous Design      |      |      |      |
| 2024 | FastServe               | Heterogeneous Design      |      |      |      |
| 2024 | PartKVRec               | Heterogeneous Design      |      |      |      |
| 2023 | FlexGen                 | SSD-based Design         |      |      |      |
| 2024 | InstInfer               | SSD-based Design         |      |      |      |

---

## Datasets and Benchmarks

### Text Datasets

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|||||||

### Multi-modal Datasets

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|||||||

---