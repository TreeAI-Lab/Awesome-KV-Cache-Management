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
|2024| Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs | Static KV Cache Selection |ICLR|[Link](https://arxiv.org/pdf/2310.01801)||
|2024| SnapKV: LLM Knows What You are Looking for Before Generation | Static KV Cache Selection |NeurIPS|[Link](https://arxiv.org/pdf/2404.14469)|[Link](https://github.com/FasterDecoding/SnapKV)|
|2024| In-context KV-Cache Eviction for LLMs via Attention-Gate | Static KV Cache Selection |arXiv|[Link](https://arxiv.org/pdf/2410.12876)||

### Dynamic Selection with Permanent Eviction

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024| Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference | Dynamic Selection with Permanent Eviction |MLSys|[Link](https://arxiv.org/pdf/2403.09054)||
|2024| BUZZ: Beehive-structured Sparse KV Cache with Segmented Heavy Hitters for Efficient LLM Inference | Dynamic Selection with Permanent Eviction |arXiv|[Link](https://arxiv.org/pdf/2410.23079)|[Link](https://github.com/JunqiZhao888/buzz-llm)|
|2024| NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time | Dynamic Selection with Permanent Eviction |ACL|[Link](https://arxiv.org/pdf/2408.03675)|[Link](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2024-NACL)|
|2023| H2O: heavy-hitter oracle for efficient generative inference of large language models | Dynamic Selection with Permanent Eviction |NeurIPS|[Link](https://arxiv.org/pdf/2306.14048)|[Link](https://github.com/FMInference/H2O)|
|2023| Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time | Dynamic Selection with Permanent Eviction |NeurIPS|[Link](https://arxiv.org/pdf/2305.17118)||

### Dynamic Selection without Permanent Eviction

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024| InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory | Dynamic Selection without Permanent Eviction |arXiv|[Link](https://arxiv.org/pdf/2402.04617)|[Link](https://github.com/thunlp/InfLLM)|
|2024| Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference | Dynamic Selection without Permanent Eviction |ICML|[Link](https://arxiv.org/pdf/2406.10774)|[Link](https://github.com/mit-han-lab/Quest)|
|2024| PQCache: Product Quantization-based KVCache for Long Context LLM Inference | Dynamic Selection without Permanent Eviction |arXiv|[Link](https://arxiv.org/pdf/2407.12820)||
|2024| Squeezed Attention: Accelerating Long Context Length LLM Inference | Dynamic Selection without Permanent Eviction |arXiv|[Link](https://arxiv.org/pdf/2411.09688)|[Link](https://github.com/SqueezeAILab/SqueezedAttention)|
|2024| RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval | Dynamic Selection without Permanent Eviction |arXiv|[Link](https://arxiv.org/pdf/2409.10516)|[Link](https://github.com/jzbjyb/ReAtt)|
|2024| Human-like Episodic Memory for Infinite Context LLMs | Dynamic Selection without Permanent Eviction |arXiv|[Link](https://arxiv.org/pdf/2407.09450)||

## KV Cache Budget Allocation

### Layer-wise Budget Allocation

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024|PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling|Layer-wise Budget Allocation|arXiv|[Link](https://arxiv.org/pdf/2406.02069)|[Link](https://github.com/Zefan-Cai/KVCache-Factory)|
|2024|PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference|Layer-wise Budget Allocation|ACL|[Link](https://arxiv.org/pdf/2405.12532)|[Link](https://github.com/mutonix/pyramidinfer)|
|2024| DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs |Layer-wise Budget Allocation|ICLR sub.|[Link](https://arxiv.org/pdf/2412.14838)||
|2024|PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation|Layer-wise Budget Allocation|arXiv|[Link](https://arxiv.org/pdf/2412.03409)|[Link](https://github.com/THU-MIG/PrefixKV)|
|2024|SimLayerKV: A Simple Framework for Layer-Level KV Cache Reduction|Layer-wise Budget Allocation|arXiv|[Link](https://arxiv.org/pdf/2410.13846)|[Link](https://github.com/sail-sg/SimLayerKV)|

### Head-wise Budget Allocation

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024|Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference|Head-wise Budget Allocation|arXiv|[Link](https://arxiv.org/pdf/2407.11550)||
|2024|Identify Critical KV Cache in LLM Inference from an Output Perturbation Perspective|Head-wise Budget Allocation|ICLR sub.|[Link](https://openreview.net/forum?id=lRTDMGYCpy)||
|2024|Unifying KV Cache Compression for Large Language Models with LeanKV|Head-wise Budget Allocation|arXiv|[Link](Unifying KV Cache Compression for Large Language Models with LeanKV)||
|2024|RazorAttention: Efficient KV Cache Compression Through Retrieval Heads|Head-wise Budget Allocation|arXiv|[Link](https://arxiv.org/pdf/2407.15891)||
|2024| Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning |Head-wise Budget Allocation|arXiv|[Link](https://arxiv.org/pdf/2410.19258)|[Link](https://github.com/FYYFU/HeadKV/tree/main)|
|2024|DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads|Head-wise Budget Allocation|arXiv|[Link](https://arxiv.org/pdf/2410.10819)|[Link](https://github.com/mit-han-lab/duo-attention)|

## KV Cache Merging

### Intra-layer Merging

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024|Compressed Context Memory for Online Language Model Interaction|Intra-layer Merging|ICLR|[Link](https://openreview.net/forum?id=64kSvC4iPg)|[Link](https://github.com/snu-mllab/context-memory)|
|2024|LoMA: Lossless Compressed Memory Attention|Intra-layer Merging|arXiv|[Link](https://arxiv.org/pdf/2401.09486)||
|2024|Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference|Intra-layer Merging|ICML|[Link](https://openreview.net/forum?id=tDRYrAkOB7)|[Link](https://github.com/NVIDIA/Megatron-LM/tree/dmc)|
|2024|CaM: Cache Merging for Memory-efficient LLMs Inference|Intra-layer Merging|ICML|[Link](https://openreview.net/forum?id=LCTmppB165)|[Link](https://github.com/zyxxmu/cam)|
|2024|D2O: Dynamic Discriminative Operations for Efficient Generative Inference of Large Language Models|Intra-layer Merging|arXiv|[Link](https://arxiv.org/pdf/2406.13035)||
|2024|AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning|Intra-layer Merging|arXiv|[Link](https://arxiv.org/pdf/2412.03248)|[Link](https://github.com/LaVi-Lab/AIM)|
|2024|LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference|Intra-layer Merging|EMNLP|[Link](https://aclanthology.org/2024.findings-emnlp.235/)|[Link](https://github.com/SUSTechBruce/LOOK-M)|
|2024|Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks|Intra-layer Merging|arXiv|[Link](https://arxiv.org/pdf/2407.08454)||
|2024|CHAI: Clustered Head Attention for Efficient LLM Inference|Intra-layer Merging|arXiv|[Link](https://arxiv.org/pdf/2403.08058)||

### Cross-layer Merging

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024|MiniCache: KV Cache Compression in Depth Dimension for Large Language Models|Cross-layer Merging|arXiv|[Link](https://arxiv.org/pdf/2405.14366)|[Link](https://github.com/AkideLiu/MiniCache)|
|2024| KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing |Cross-layer Merging|arXiv|[Link](https://arxiv.org/pdf/2410.18517)|[Link](https://github.com/yangyifei729/KVSharer)|

## KV Cache Quantization

### Fixed-precision Quantization

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024|QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead|Fixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2406.03482)|[Link](https://github.com/amirzandieh/QJL)|
|2024|PQCache: Product Quantization-based KVCache for Long Context LLM Inference|Fixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2407.12820)||
|2023|FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU|Fixed-precision Quantization|ICML|[Link](https://proceedings.mlr.press/v202/sheng23a.html)|[Link](https://github.com/FMInference/FlexLLMGen)|
|2022|ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers|Fixed-precision Quantization|NIPS|[Link](https://arxiv.org/pdf/2206.01861)|[Link](https://github.com/microsoft/DeepSpeed)|


### Mixed-precision Quantization

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024|KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2401.18079)|[Link](https://github.com/SqueezeAILab/KVQuant)|
|2024|IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2403.01241)|[Link](https://github.com/ruikangliu/IntactKV)|
|2024|SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2405.06219)|[Link](https://github.com/cat538/SKVQ)|
|2024|KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2402.02750)|[Link](https://github.com/jy-yuan/KIVI)|
|2024|WKVQuant: Quantizing Weight and Key/Value Cache for Large Language Models Gains More|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2402.12065)||
|2024|GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2403.05527)|[Link](https://github.com/opengear-project/GEAR)|
|2024|No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2402.18096)||
|2024|ZipVL: Efficient Large Vision-Language Models with Dynamic Token Sparsification|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2410.08584)||
|2024|ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2405.14256)|[Link](https://github.com/ThisisBillhe/ZipCache)|
|2024|PrefixQuant: Static Quantization Beats Dynamic through Prefixed Outliers in LLMs|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2410.05265)|[Link](https://github.com/ChenMnZ/PrefixQuant)|
|2024|MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache|Mixed-precision Quantization|arXiv|[Link](https://arxiv.org/pdf/2411.18077)||

### Outlier Redistribution

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024|Massive Activations in Large Language Models|Outlier Redistribution|arXiv|[Link](https://arxiv.org/pdf/2402.17762)|[Link](https://github.com/locuslab/massive-activations)|
|2024|QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs|Outlier Redistribution|arXiv|[Link](https://arxiv.org/pdf/2404.00456)|[Link](https://github.com/spcl/QuaRot)|
|2024|QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving|Outlier Redistribution|arXiv|[Link](https://arxiv.org/pdf/2405.04532)|[Link](https://github.com/mit-han-lab/qserve)|
|2024|SpinQuant: LLM Quantization with Learned Rotations|Outlier Redistribution|arXiv|[Link](https://arxiv.org/pdf/2405.16406)|[Link](https://github.com/facebookresearch/SpinQuant)|
|2024| DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs |Outlier Redistribution|NeurIPS|[Link](https://arxiv.org/pdf/2406.01721)|[Link](https://github.com/Hsu1023/DuQuant)|
|2024|SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models|Outlier Redistribution|ICML|[Link](https://proceedings.mlr.press/v202/xiao23c.html)|[Link](https://github.com/mit-han-lab/smoothquant)|
|2024|Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling|Outlier Redistribution|ACL|[Link](https://arxiv.org/pdf/2304.09145)|[Link](https://github.com/ModelTC/Outlier_Suppression_Plus)|
|2024|AffineQuant: Affine Transformation Quantization for Large Language Models|Outlier Redistribution|arXiv|[Link](https://arxiv.org/pdf/2403.12544)|[Link](https://github.com/bytedance/AffineQuant)|
|2024|FlatQuant: Flatness Matters for LLM Quantization|Outlier Redistribution|arXiv|[Link](https://arxiv.org/pdf/2410.09426)|[Link](https://github.com/ruikangliu/FlatQuant)|
|2024|AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration|Outlier Redistribution|MLSys|[Link](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-pdftract-Conference.html)|[Link](https://github.com/mit-han-lab/llm-awq)|
|2023|OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models|Outlier Redistribution|arXiv|[Link](https://arxiv.org/pdf/2308.13137)|[Link](https://github.com/OpenGVLab/OmniQuant)|
|2023| Training Transformers with 4-bit Integers                    |Outlier Redistribution|NeurIPS|[Link](https://proceedings.neurips.cc//paper_files/paper/2023/hash/99fc8bc48b917c301a80cb74d91c0c06-pdftract-Conference.html)|[Link](https://github.com/xijiu9/Train_Transformers_with_INT4)|


## KV Cache Low-rank Decomposition

### Singular Value Decomposition

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024|Effectively Compress KV Heads for LLM|Singular Value Decomposition|arXiv|[Link](https://arxiv.org/pdf/2406.07056)||
|2024|Eigen Attention: Attention in Low-Rank Space for KV Cache Compression|Singular Value Decomposition|arXiv|[Link](https://arxiv.org/pdf/2408.05646)|[Link](https://github.com/UtkarshSaxena1/EigenAttn/tree/main)|
|2024|Zero-Delay QKV Compression for Mitigating KV Cache and Network Bottlenecks in LLM Inference|Singular Value Decomposition|arXiv|[Link](https://arxiv.org/pdf/2408.04107)||
|2024|LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy|Singular Value Decomposition|arXiv|[Link](https://arxiv.org/pdf/2410.03111)||
|2024|ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference|Singular Value Decomposition|arXiv|[Link](https://arxiv.org/pdf/2410.21465)|[Link](https://github.com/bytedance/ShadowKV)|
|2024|Palu: Compressing KV-Cache with Low-Rank Projection|Singular Value Decomposition|arXiv|[Link](https://arxiv.org/pdf/2407.21118)|[Link](https://github.com/shadowpa0327/Palu)|

### Tensor Decomposition

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024|Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression|Tensor Decomposition|ACL|[Link](https://arxiv.org/pdf/2405.12591)|[Link](https://github.com/lpyhdzx/DecoQuant_code)|

### Learned Low-rank Approximation

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|2024|Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference|Learned Low-rank Approximation|arXiv|[Link](https://arxiv.org/pdf/2402.09398)|[Link](https://github.com/hdong920/LESS)|
|2024|MatryoshkaKV: Adaptive KV Compression via Trainable Orthogonal Projection|Learned Low-rank Approximation|arXiv|[Link](https://arxiv.org/pdf/2410.14731)||

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
