# Awesome-KV-Cache-Management

This repo aims to record a survey on LLM acceleration based on KV Cache Management.

- [Token-level Optimization](#token-level-optimization)
    - [KV Cache Selection](#kv-cache-selection)
        - [Static KV Cache Selection](#static-kv-cache-selection)
        - [Dynamic Selection with Permanent Eviction](#dynamic-selection-with-permanent-eviction)
        - [Dynamic Selection without Permanent Eviction](#dynamic-selection-without-permanent-eviction)
    - [KV Cache Budget Allocation](#kv-cache-budget-allocation)
        - [Layer-wise Budget Allocation](#layer-wise-budget-allocation)
        - [Head-wise Budget Allocation](#head-wise-budget-allocation)
    - [KV Cache Merging](#kv-cache-merging)
        - [Intra-layer Merging](#intra-layer-merging)
        - [Cross-layer Merging](#cross-layer-merging)
    - [KV Cache Quantization](#kv-cache-quantization)
        - [Fixed-precision Quantization](#fixed-precision-quantization)
        - [Mixed-precision Quantization](#mixed-precision-quantization)
        - [Outlier Redistribution](#outlier-redistribution)
    - [KV Cache Low-rank Decomposition](#kv-cache-low-rank-decomposition)
        - [Singular Value Decomposition](#singular-value-decomposition)
        - [Tensor Decomposition](#tensor-decomposition)
        - [Learned Low-rank Approximation](#learned-low-rank-approximation)
- [Model-level Optimization](#model-level-optimization)
    - [Intra later](#intra-later)
        - [Grouped Attention](#grouped-attention)
        - [Compression](#compression)
        - [Extended Mechanism](#extended-mechanism)
    - [Cross Layer](#cross-layer)
        - [Cache Sharing](#cache-sharing)
        - [Compression](#compression)
        - [Augmented Architectures](#augmented-architectures)
    - [Non-transformer Architecture](#non-transformer-architecture)
        - [New Architecture](#new-architecture)
        - [Hybrid Architecture](#hybrid-architecture)
- [System-level Optimization](#system-level-optimization)
    - [Memory Management](#memory-management)
        - [Prefix-aware Design](#prefix-aware-design)
        - [Architectural Design](#architectural-design)
    - [Scheduling](#scheduling)
        - [Prefix-aware Scheduling](#prefix-aware-scheduling)
        - [Preemptive and Fairness-oriented Scheduling](#preemptive-and-fairness-oriented-scheduling)
        - [Layer-specific and Hierarchical Scheduling](#layer-specific-and-hierarchical-scheduling)
    - [Hardware-aware Design](#hardware-aware-design)
        - [Single/Multi-GPU Design](#singlemulti-gpu-design)
        - [I/O-based Design](#io-based-design)
        - [Heterogeneous Design](#heterogeneous-design)
        - [SSD-based Design](#ssd-based-design)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
    - [Text Datasets](#text-datasets)
    - [Multi-modal Datasets](#multi-modal-datasets)

---

# Token-level Optimization

## KV Cache Selection

### Static KV Cache Selection

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024| Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs | Static KV Cache Selection |arXiv|[Link](https://arxiv.org/abs/2310.01801)||
|2024| SnapKV: LLM Knows What You are Looking for Before Generation | Static KV Cache Selection |arXiv|[Link](https://arxiv.org/abs/2404.14469)|[Link](https://github.com/FasterDecoding/SnapKV)|
|2024| Attention-Gate | Static KV Cache Selection |arXiv|[Link](https://arxiv.org/abs/2410.12876)||

### Dynamic Selection with Permanent Eviction

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2023| H2O | Dynamic Selection with Permanent Eviction |arXiv|[Link](https://arxiv.org/abs/2306.14048)|[Link](https://github.com/FMInference/H2O)|
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

## KV Cache Budget Allocation

### Layer-wise Budget Allocation

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
||PyramidKV|Layer-wise Budget Allocation||||
||PyrimidInfer|Layer-wise Budget Allocation||||
||DynamicKV|Layer-wise Budget Allocation||||
||PrefixKV|Layer-wise Budget Allocation||||
||SimLayerKV|Layer-wise Budget Allocation||||

### Head-wise Budget Allocation

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
||AdaKV|Head-wise Budget Allocation||||
||CriticalKV|Head-wise Budget Allocation||||
||LeanKV|Head-wise Budget Allocation||||
||RazorAttention|Head-wise Budget Allocation||||
||HeadKV|Head-wise Budget Allocation||||
||DuoAttention|Head-wise Budget Allocation||||

## KV Cache Merging

### Intra-layer Merging

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
||CCM|Intra-layer Merging||||
||CaM|Intra-layer Merging||||
||D2O|Intra-layer Merging||||
||AIM|Intra-layer Merging||||
||Look-M|Intra-layer Merging||||
||KVMerger|Intra-layer Merging||||

### Cross-layer Merging

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
||MiniCache|Cross-layer Merging||||
|| KVSharer  |Cross-layer Merging||||

## KV Cache Quantization

### Fixed-precision Quantization

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
||ZeroQuant|Fixed-precision Quantization||||
||FlexGen|Fixed-precision Quantization||||
||QJL|Fixed-precision Quantization||||
||PQCache|Fixed-precision Quantization||||

### Mixed-precision Quantization

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
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

### Outlier Redistribution

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
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

## KV Cache Low-rank Decomposition

### Singular Value Decomposition

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
||ECKVH|Singular Value Decomposition||||
||EigenAttention|Singular Value Decomposition||||
||ZDC|Singular Value Decomposition||||
||LoRC|Singular Value Decomposition||||
||ShadowKV|Singular Value Decomposition||||
||Palu|Singular Value Decomposition||||

### Tensor Decomposition

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
||DecoQuant|Tensor Decomposition||||

### Learned Low-rank Approximation

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
||LESS|Learned Low-rank Approximation||||

---

# Model-level Optimization

## Intra later

### Grouped Attention

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2019 | MQA                     | Grouped Attention        |      |      |      |
| 2023 | GQA                     | Grouped Attention        |      |      |      |
| 2024 | AsymGQA                 | Grouped Attention        |      |      |      |
| 2024 | Weighted GQA            | Grouped Attention        |      |      |      |
| 2024 | QCQA                    | Grouped Attention        |      |      |      |
| 2024 | KDGQA                   | Grouped Attention        |      |      |      |
| 2023 | GQKVA                   | Grouped Attention        |      |      |      |

### Compression 

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | MLA                     | Compression              |      |      |      |
| 2024 | MatryoshkaKV            | Compression              |      |      |      |

### Extended Mechanism

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2022 | FLASH                   | Extended Mechanism       |      |      |      |
| 2024 | Infini-Attention        | Extended Mechanism       |      |      |      |

## Cross Layer 

### Cache Sharing

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

### Compression

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | YOCO                    | Compression              |      |      |      |
| 2024 | CLLA                    | Compression              |      |      |      |

### Augmented Architectures

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | CEPE                    | Augmented Architectures  |      |      |      |
| 2024 | XC-Cache                | Augmented Architectures  |      |      |      |
| 2024 | Block Transformer       | Augmented Architectures  |      |      |      |

## Non-transformer Architecture

### New Architecture

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2023 | RWKV                    | New Architecture         |      |      |      |
| 2024 | Mamba                   | New Architecture         |      |      |      |
| 2023 | RetNet                  | New Architecture         |      |      |      |
| 2024 | MCSD                    | New Architecture         |      |      |      |

### Hybrid Architecture

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | MixCon                  | Hybrid Architecture      |      |      |      |
| 2024 | GoldFinch               | Hybrid Architecture      |      |      |      |
| 2024 | RecurFormer             | Hybrid Architecture      |      |      |      |

---

# System-level Optimization

## Memory Management

### Architectural Design

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2023 | vLLM                    | Architectural Design      |      |      |      |
| 2024 | vTensor                 | Architectural Design      |      |      |      |
| 2024 | LeanKV                  | Architectural Design      |      |      |      |

### Prefix-aware Design

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | ChunkAttention          | Prefix-aware Design       |      |      |      |
| 2024 | MemServe                | Prefix-aware Design       |      |      |      |

## Scheduling

### Prefix-aware Scheduling

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | BatchLLM                | Prefix-aware Scheduling   |      |      |      |
| 2024 | RadixAttention          | Prefix-aware Scheduling   |      |      |      |

### Preemptive and Fairness-oriented Scheduling

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | FastServe               | Preemptive and Fairness-oriented Scheduling |      |      |      |
| 2024 | FastSwitch              | Preemptive and Fairness-oriented Scheduling |      |      |      |

### Layer-specific and Hierarchical Scheduling

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | LayerKV                 | Layer-specific and Hierarchical Scheduling  |      |      |      |
| 2024 | CachedAttention         | Layer-specific and Hierarchical Scheduling  |      |      |      |
| 2024 | ALISA                   | Layer-specific and Hierarchical Scheduling  |      |      |      |
| 2024 | LAMPS                   | Layer-specific and Hierarchical Scheduling  |      |      |      |


## Hardware-aware Design

### Single/Multi-GPU Design

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | HydraGen               | Single/Multi-GPU Design   |      |      |      |
| 2024 | DeFT                   | Single/Multi-GPU Design   |      |      |      |
| 2023 | vLLM                   | Single/Multi-GPU Design   |      |      |      |
| 2022 | ORCA                   | Single/Multi-GPU Design   |      |      |      |
| 2024 | DistServe              | Single/Multi-GPU Design   |      |      |      |
| 2024 | Multi-Bin Batching     | Single/Multi-GPU Design   |      |      |      |
| 2024 | Tree Attention         | Single/Multi-GPU Design   |      |      |      |

### I/O-based Design

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2022 | FlashAttention          | I/O-based Design         |      |      |      |
| 2024 | Bifurcated Attention    | I/O-based Design         |      |      |      |
| 2024 | PartKVRec               | I/O-based Design         |      |      |      |
| 2024 | HCache                  | I/O-based Design         |      |      |      |
| 2024 | Cake                    | I/O-based Design         |      |      |      |
| 2024 | FastSwitch              | I/O-based Design         |      |      |      |

### Heterogeneous Design

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2024 | NEO                     | Heterogeneous Design      |      |      |      |
| 2024 | FastDecode              | Heterogeneous Design      |      |      |      |
| 2024 | FlexInfer               | Heterogeneous Design      |      |      |      |
| 2024 | InfiniGen               | Heterogeneous Design      |      |      |      |
| 2023 | Pensieve                | Heterogeneous Design      |      |      |      |
| 2024 | FastServe               | Heterogeneous Design      |      |      |      |
| 2024 | PartKVRec               | Heterogeneous Design      |      |      |      |

### SSD-based Design

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
| 2023 | FlexGen                 | SSD-based Design         |      |      |      |
| 2024 | InstInfer               | SSD-based Design         |      |      |      |

---

# Datasets and Benchmarks

## Text Datasets

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|||||||

## Multi-modal Datasets

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|||||||

---