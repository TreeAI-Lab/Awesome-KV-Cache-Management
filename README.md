# Awesome-KV-Cache-Management


### News

**📢 Multi-turn KV Strategies and Benchmark Released (2025-07-18):** *"LoopServe: An Adaptive Dual-phase LLM Inference Acceleration System for Multi-Turn Dialogues [[PDF]](https://www.arxiv.org/abs/2507.13681)[[Dataset]](https://huggingface.co/datasets/TreeAILab/Multi-turn_Long-context_Benchmark_for_LLMs)"*  🚀


**📢 Two Papers Accepted (2025-05-20):** Our survey has been accepted by TMLR 2025, and our numerical benchmark paper has been accepted by ACL 2025! 🚀

**📢 Numerical Benchmark  Released (2025-02-18):** *"Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models [[PDF]](https://arxiv.org/pdf/2502.11075)[[Dataset]](https://github.com/TreeAI-Lab/NumericBench)"* — proposing a long NumericBench to assess LLMs' numerical reasoning! 🚀




## A Survey on Large Language Model Acceleration based on KV Cache Management [[PDF]](https://arxiv.org/pdf/2412.19442) 

> *Haoyang Li <sup>1</sup>, Yiming Li <sup>2</sup>, Anxin Tian <sup>2</sup>, Tinahao Tang <sup>2</sup>, Zhanchao Xu <sup>4</sup>, Xuejia Chen <sup>4</sup>, Nicole Hu <sup>3</sup>, Wei Dong <sup>5</sup>, Qing Li <sup>1</sup>, Lei Chen <sup>2</sup>*

> *<sup>1</sup>Hong Kong Polytechnic University, <sup>2</sup>Hong Kong University of Science and Technology, <sup>3</sup>The Chinese University of Hong Kong, <sup>4</sup>Huazhong University of Science and Technology, <sup>5</sup>Nanyang Technological University.*

- This repository is dedicated to recording KV Cache Management papers for LLM acceleration. The survey will be updated regularly. If you find this survey helpful for your work, please consider citing it.
```
  @article{li2024surveylargelanguagemodel,
      title={A Survey on Large Language Model Acceleration based on KV Cache Management}, 
      author={Haoyang Li and Yiming Li and Anxin Tian and Tianhao Tang and Zhanchao Xu and Xuejia Chen and Nicole Hu and Wei Dong and Qing Li and Lei Chen},
      journal={arXiv preprint arXiv:2412.19442},
      year={2024}
  }
  ```
- If you would like to include your paper or  any modifications in this survey and repository, please feel free to send email to (haoyang-comp.li@polyu.edu.hk) or open an issue with your paper's title, category, and a brief summary highlighting its key techniques. Thank you!


## Toxonomy and Papers

- [Awesome-KV-Cache-Management](#awesome-kv-cache-management)
- [Token-level Optimization](#token-level-optimization)
  - [KV Cache Selection](#kv-cache-selection)
    - [Static KV Cache Selection (To Top👆🏻)](#static-kv-cache-selection-to-top)
    - [Dynamic Selection with Permanent Eviction (To Top👆🏻)](#dynamic-selection-with-permanent-eviction-to-top)
    - [Dynamic Selection without Permanent Eviction (To Top👆🏻)](#dynamic-selection-without-permanent-eviction-to-top)
  - [KV Cache Budget Allocation](#kv-cache-budget-allocation)
    - [Layer-wise Budget Allocation (To Top👆🏻)](#layer-wise-budget-allocation-to-top)
    - [Head-wise Budget Allocation (To Top👆🏻)](#head-wise-budget-allocation-to-top)
  - [KV Cache Merging](#kv-cache-merging)
    - [Intra-layer Merging (To Top👆🏻)](#intra-layer-merging-to-top)
    - [Cross-layer Merging (To Top👆🏻)](#cross-layer-merging-to-top)
  - [KV Cache Quantization](#kv-cache-quantization)
    - [Fixed-precision Quantization (To Top👆🏻)](#fixed-precision-quantization-to-top)
    - [Mixed-precision Quantization (To Top👆🏻)](#mixed-precision-quantization-to-top)
    - [Outlier Redistribution (To Top👆🏻)](#outlier-redistribution-to-top)
  - [KV Cache Low-rank Decomposition](#kv-cache-low-rank-decomposition)
    - [Singular Value Decomposition (To Top👆🏻)](#singular-value-decomposition-to-top)
    - [Tensor Decomposition (To Top👆🏻)](#tensor-decomposition-to-top)
    - [Learned Low-rank Approximation (To Top👆🏻)](#learned-low-rank-approximation-to-top)
- [Model-level Optimization](#model-level-optimization)
  - [Attention Grouping and Sharing](#attention-grouping-and-sharing)
    - [Intra-Layer Grouping (To Top👆🏻)](#intra-layer-grouping-to-top)
    - [Cross-Layer Sharing (To Top👆🏻)](#cross-layer-sharing-to-top)
  - [Architecture Alteration](#architecture-alteration)
    - [Enhanced Attention (To Top👆🏻)](#enhanced-attention-to-top)
    - [Augmented Architecture (To Top👆🏻)](#augmented-architecture-to-top)
  - [Non-transformer Architecture](#non-transformer-architecture)
    - [Adaptive Sequence Processing Architecture (To Top👆🏻)](#adaptive-sequence-processing-architecture-to-top)
    - [Hybrid Architecture (To Top👆🏻)](#hybrid-architecture-to-top)
- [System-level Optimization](#system-level-optimization)
  - [Memory Management](#memory-management)
    - [Architectural Design (To Top👆🏻)](#architectural-design-to-top)
    - [Prefix-aware Design (To Top👆🏻)](#prefix-aware-design-to-top)
  - [Scheduling](#scheduling)
    - [Prefix-aware Scheduling (To Top👆🏻)](#prefix-aware-scheduling-to-top)
    - [Preemptive and Fairness-oriented Scheduling (To Top👆🏻)](#preemptive-and-fairness-oriented-scheduling-to-top)
    - [Layer-specific and Hierarchical Scheduling (To Top👆🏻)](#layer-specific-and-hierarchical-scheduling-to-top)
  - [Hardware-aware Design](#hardware-aware-design)
    - [Single/Multi-GPU Design (To Top👆🏻)](#singlemulti-gpu-design-to-top)
    - [I/O-based Design (To Top👆🏻)](#io-based-design-to-top)
    - [Heterogeneous Design (To Top👆🏻)](#heterogeneous-design-to-top)
    - [SSD-based Design (To Top👆🏻)](#ssd-based-design-to-top)
- [Datasets and Benchmarks](#datasets-and-benchmarks)

---

# Token-level Optimization

## KV Cache Selection

### Static KV Cache Selection ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                   | Type                      | Venue   | Paper                                 | code                                                                                                                        |
| ---- | ----------------------------------------------------------------------- | ------------------------- | ------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 2024 | Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs | Static KV Cache Selection | ICLR    | [Link](https://arxiv.org/pdf/2310.01801) |                                                                                                                             |
| 2024 | SnapKV: LLM Knows What You are Looking for Before Generation            | Static KV Cache Selection | NeurIPS | [Link](https://arxiv.org/pdf/2404.14469) | [Link](https://github.com/FasterDecoding/SnapKV) ![](https://img.shields.io/github/stars/FasterDecoding/SnapKV.svg?style=social) |
| 2024 | A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression  | Static KV Cache Selection | EMNLP   | [Link](https://arxiv.org/pdf/2406.11430) | [Link](https://github.com/alessiodevoto/l2compress) ![](https://img.shields.io/github/stars/alessiodevoto/l2compress.svg?style=social) |
| 2024 | In-context KV-Cache Eviction for LLMs via Attention-Gate                | Static KV Cache Selection | arXiv   | [Link](https://arxiv.org/pdf/2410.12876) |                                                                                                                             |

### Dynamic Selection with Permanent Eviction ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                                       | Type                                      | Venue   | Paper                                 | code                                                                                                                                                     |
| ---- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2025 | SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator                      | Dynamic Selection with Permanent Eviction | MLSys   | [Link](https://arxiv.org/pdf/2412.12094) |                                                                                                                                                          |
| 2024 | Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference               | Dynamic Selection with Permanent Eviction | MLSys   | [Link](https://arxiv.org/pdf/2403.09054) |                                                                                                                                                          |
| 2024 | BUZZ: Beehive-structured Sparse KV Cache with Segmented Heavy Hitters for Efficient LLM Inference           | Dynamic Selection with Permanent Eviction | arXiv   | [Link](https://arxiv.org/pdf/2410.23079) | [Link](https://github.com/JunqiZhao888/buzz-llm) ![](https://img.shields.io/github/stars/JunqiZhao888/buzz-llm.svg?style=social)                              |
| 2024 | NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time                        | Dynamic Selection with Permanent Eviction | ACL     | [Link](https://arxiv.org/pdf/2408.03675) | [Link](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2024-NACL) ![](https://img.shields.io/github/stars/PaddlePaddle/Research.svg?style=social) |
| 2023 | H2O: heavy-hitter oracle for efficient generative inference of large language models                        | Dynamic Selection with Permanent Eviction | NeurIPS | [Link](https://arxiv.org/pdf/2306.14048) | [Link](https://github.com/FMInference/H2O) ![](https://img.shields.io/github/stars/FMInference/H2O.svg?style=social)                                          |
| 2023 | Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time | Dynamic Selection with Permanent Eviction | NeurIPS | [Link](https://arxiv.org/pdf/2305.17118) |                                                                                                                                                          |

### Dynamic Selection without Permanent Eviction ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                      | Type                                         | Venue | Paper                                 | code                                                                                                                                          |
| ---- | ------------------------------------------------------------------------------------------ | -------------------------------------------- | ----- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 2025 | LoopServe: An Adaptive Dual-phase LLM Inference Acceleration System for Multi-Turn Dialogues | Dynamic Selection without Permanent Eviction | arXiv | [Link](https://www.arxiv.org/pdf/2507.13681) |                                    |
| 2024 | InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory | Dynamic Selection without Permanent Eviction | arXiv | [Link](https://arxiv.org/pdf/2402.04617) | [Link](https://github.com/thunlp/InfLLM) ![](https://img.shields.io/github/stars/thunlp/InfLLM.svg?style=social)                                   |
| 2024 | Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference                       | Dynamic Selection without Permanent Eviction | ICML  | [Link](https://arxiv.org/pdf/2406.10774) | [Link](https://github.com/mit-han-lab/Quest) ![](https://img.shields.io/github/stars/mit-han-lab/Quest.svg?style=social)                           |
| 2024 | PQCache: Product Quantization-based KVCache for Long Context LLM Inference                 | Dynamic Selection without Permanent Eviction | arXiv | [Link](https://arxiv.org/pdf/2407.12820) |                                                                                                                                               |
| 2024 | Squeezed Attention: Accelerating Long Context Length LLM Inference                         | Dynamic Selection without Permanent Eviction | arXiv | [Link](https://arxiv.org/pdf/2411.09688) | [Link](https://github.com/SqueezeAILab/SqueezedAttention) ![](https://img.shields.io/github/stars/SqueezeAILab/SqueezedAttention.svg?style=social) |
| 2024 | RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval           | Dynamic Selection without Permanent Eviction | arXiv | [Link](https://arxiv.org/pdf/2409.10516) | [Link](https://github.com/jzbjyb/ReAtt) ![](https://img.shields.io/github/stars/jzbjyb/ReAtt.svg?style=social)                                     |
| 2024 | Human-like Episodic Memory for Infinite Context LLMs                                       | Dynamic Selection without Permanent Eviction | arXiv | [Link](https://arxiv.org/pdf/2407.09450) |                                                                                                                                               |
| 2024 | ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression          | Dynamic Selection without Permanent Eviction | arXiv | [Link](https://arxiv.org/pdf/2412.03213) |             |                                                                                                                                
| 2024 | Loki: Low-rank Keys for Efficient Sparse Attention                                         | Dynamic Selection without Permanent Eviction | arXiv | [Link](https://arxiv.org/pdf/2406.02542) | [Link](https://github.com/hpcgroup/loki) ![](https://img.shields.io/github/stars/hpcgroup/loki.svg?style=social)            |  

## KV Cache Budget Allocation

### Layer-wise Budget Allocation ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                        | Type                         | Venue     | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | ---------------------------- | --------- | ---------------------------------------- | ------------------------------------------------------------ |
| 2024 | PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling | Layer-wise Budget Allocation | arXiv     | [Link](https://arxiv.org/pdf/2406.02069) | [Link](https://github.com/Zefan-Cai/KVCache-Factory) ![](https://img.shields.io/github/stars/Zefan-Cai/KVCache-Factory.svg?style=social) |
| 2024 | PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference | Layer-wise Budget Allocation | Findings  | [Link](https://arxiv.org/pdf/2405.12532) | [Link](https://github.com/mutonix/pyramidinfer) ![](https://img.shields.io/github/stars/mutonix/pyramidinfer.svg?style=social) |
| 2024 | DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs | Layer-wise Budget Allocation | ICLR sub. | [Link](https://arxiv.org/pdf/2412.14838) |                                                              |
| 2024 | PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation | Layer-wise Budget Allocation | arXiv     | [Link](https://arxiv.org/pdf/2412.03409) | [Link](https://github.com/THU-MIG/PrefixKV) ![](https://img.shields.io/github/stars/THU-MIG/PrefixKV.svg?style=social) |
| 2024 | SimLayerKV: A Simple Framework for Layer-Level KV Cache Reduction | Layer-wise Budget Allocation | arXiv     | [Link](https://arxiv.org/pdf/2410.13846) | [Link](https://github.com/sail-sg/SimLayerKV) ![](https://img.shields.io/github/stars/sail-sg/SimLayerKV.svg?style=social) |

### Head-wise Budget Allocation ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                                  | Type                        | Venue     | Paper                                                                       | code                                                                                                                                |
| ---- | ------------------------------------------------------------------------------------------------------ | --------------------------- | --------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 2024 | Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference         | Head-wise Budget Allocation | arXiv     | [Link](https://arxiv.org/pdf/2407.11550)                                       |                                                                                                                                     |
| 2024 | Identify Critical KV Cache in LLM Inference from an Output Perturbation Perspective                    | Head-wise Budget Allocation | ICLR sub. | [Link](https://openreview.net/forum?id=lRTDMGYCpy)                             |                                                                                                                                     |
| 2024 | Unifying KV Cache Compression for Large Language Models with LeanKV                                    | Head-wise Budget Allocation | arXiv     | [Link](https://arxiv.org/pdf/2412.03131) |                                                                                                                                     |
| 2024 | RazorAttention: Efficient KV Cache Compression Through Retrieval Heads                                 | Head-wise Budget Allocation | arXiv     | [Link](https://arxiv.org/pdf/2407.15891)                                       |                                                                                                                                     |
| 2024 | Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning | Head-wise Budget Allocation | arXiv     | [Link](https://arxiv.org/pdf/2410.19258)                                       | [Link](https://github.com/FYYFU/HeadKV/tree/main) ![](https://img.shields.io/github/stars/FYYFU/HeadKV.svg?style=social)                 |
| 2024 | DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads                  | Head-wise Budget Allocation | arXiv     | [Link](https://arxiv.org/pdf/2410.10819)                                       | [Link](https://github.com/mit-han-lab/duo-attention) ![](https://img.shields.io/github/stars/mit-han-lab/duo-attention.svg?style=social) |

## KV Cache Merging

### Intra-layer Merging ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                              | Type                | Venue | Paper                                                  | code                                                                                                                              |
| ---- | -------------------------------------------------------------------------------------------------- | ------------------- | ----- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| 2025 | ZeroMerge: Parameter-Free KV Cache Compression for Memory-Efficient Long-Context LLMs              | Intra-layer Merging | arXiv  | [Link](https://arxiv.org/pdf/2503.10714v1)        | [Link](https://github.com/SusCom-Lab/ZSMerge) ![](https://img.shields.io/github/stars/SusCom-Lab/ZSMerge.svg?style=social) |
| 2024 | Compressed Context Memory for Online Language Model Interaction                                    | Intra-layer Merging | ICLR  | [Link](https://openreview.net/forum?id=64kSvC4iPg)        | [Link](https://github.com/snu-mllab/context-memory) ![](https://img.shields.io/github/stars/snu-mllab/context-memory.svg?style=social) |
| 2024 | LoMA: Lossless Compressed Memory Attention                                                         | Intra-layer Merging | arXiv | [Link](https://arxiv.org/pdf/2401.09486)                  |                                                                                                                                   |
| 2024 | Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference                            | Intra-layer Merging | ICML  | [Link](https://openreview.net/forum?id=tDRYrAkOB7)        | [Link](https://github.com/NVIDIA/Megatron-LM/tree/dmc) ![](https://img.shields.io/github/stars/NVIDIA/Megatron-LM.svg?style=social)    |
| 2024 | CaM: Cache Merging for Memory-efficient LLMs Inference                                             | Intra-layer Merging | ICML  | [Link](https://openreview.net/forum?id=LCTmppB165)        | [Link](https://github.com/zyxxmu/cam) ![](https://img.shields.io/github/stars/zyxxmu/cam.svg?style=social)                             |
| 2024 | D2O: Dynamic Discriminative Operations for Efficient Generative Inference of Large Language Models | Intra-layer Merging | arXiv | [Link](https://arxiv.org/pdf/2406.13035)                  |                                                                                                                                   |
| 2024 | AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning                          | Intra-layer Merging | arXiv | [Link](https://arxiv.org/pdf/2412.03248)                  | [Link](https://github.com/LaVi-Lab/AIM) ![](https://img.shields.io/github/stars/LaVi-Lab/AIM.svg?style=social)                         |
| 2024 | LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference         | Intra-layer Merging | EMNLP | [Link](https://aclanthology.org/2024.findings-emnlp.235/) | [Link](https://github.com/SUSTechBruce/LOOK-M) ![](https://img.shields.io/github/stars/SUSTechBruce/LOOK-M.svg?style=social)           |
| 2024 | Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks           | Intra-layer Merging | arXiv | [Link](https://arxiv.org/pdf/2407.08454)                  |                                                                                                                                   |
| 2024 | CHAI: Clustered Head Attention for Efficient LLM Inference                                         | Intra-layer Merging | arXiv | [Link](https://arxiv.org/pdf/2403.08058)                  |                                                                                                                                   |

### Cross-layer Merging ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                        | Type                | Venue | Paper                                 | code                                                                                                                        |
| ---- | ---------------------------------------------------------------------------- | ------------------- | ----- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 2024 | MiniCache: KV Cache Compression in Depth Dimension for Large Language Models | Cross-layer Merging | arXiv | [Link](https://arxiv.org/pdf/2405.14366) | [Link](https://github.com/AkideLiu/MiniCache) ![](https://img.shields.io/github/stars/AkideLiu/MiniCache.svg?style=social)       |
| 2024 | KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cross-Layer Sharing     | Cross-layer Merging | arXiv | [Link](https://arxiv.org/pdf/2410.18517) | [Link](https://github.com/yangyifei729/KVSharer) ![](https://img.shields.io/github/stars/yangyifei729/KVSharer.svg?style=social) |

## KV Cache Quantization

### Fixed-precision Quantization ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                       | Type                         | Venue | Paper                                                 | code                                                                                                                          |
| ---- | ------------------------------------------------------------------------------------------- | ---------------------------- | ----- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 2024 | QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead              | Fixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2406.03482)                 | [Link](https://github.com/amirzandieh/QJL) ![](https://img.shields.io/github/stars/amirzandieh/QJL.svg?style=social)               |
| 2024 | PQCache: Product Quantization-based KVCache for Long Context LLM Inference                  | Fixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2407.12820)                 |                                                                                                                               |
| 2023 | FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU    | Fixed-precision Quantization | ICML  | [Link](https://proceedings.mlr.press/v202/sheng23a.html) | [Link](https://github.com/FMInference/FlexLLMGen) ![](https://img.shields.io/github/stars/FMInference/FlexLLMGen.svg?style=social) |
| 2022 | ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers | Fixed-precision Quantization | NIPS  | [Link](https://arxiv.org/pdf/2206.01861)                 | [Link](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)       |

### Mixed-precision Quantization ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                                 | Type                         | Venue | Paper                                 | code                                                                                                                        |
| ---- | ----------------------------------------------------------------------------------------------------- | ---------------------------- | ----- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 2025 | Quantize What Counts: Bit Allocation Insights Informed by Spectral Gaps in Keys and Values            | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2502.15075) | [Link](https://github.com/mohsenhariri/spectral-kv) ![](https://img.shields.io/github/stars/mohsenhariri/spectral-kv.svg?style=social)   |
| 2024 | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization                   | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2401.18079) | [Link](https://github.com/SqueezeAILab/KVQuant) ![](https://img.shields.io/github/stars/SqueezeAILab/KVQuant.svg?style=social)   |
| 2024 | IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact                  | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2403.01241) | [Link](https://github.com/ruikangliu/IntactKV) ![](https://img.shields.io/github/stars/ruikangliu/IntactKV.svg?style=social)     |
| 2024 | SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models                       | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2405.06219) | [Link](https://github.com/cat538/SKVQ) ![](https://img.shields.io/github/stars/cat538/SKVQ.svg?style=social)                     |
| 2024 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache                                         | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2402.02750) | [Link](https://github.com/jy-yuan/KIVI) ![](https://img.shields.io/github/stars/jy-yuan/KIVI.svg?style=social)                   |
| 2024 | WKVQuant: Quantizing Weight and Key/Value Cache for Large Language Models Gains More                  | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2402.12065) |                                                                                                                             |
| 2024 | GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM          | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2403.05527) | [Link](https://github.com/opengear-project/GEAR) ![](https://img.shields.io/github/stars/opengear-project/GEAR.svg?style=social) |
| 2024 | No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2402.18096) |                                                                                                                             |
| 2024 | ZipVL: Efficient Large Vision-Language Models with Dynamic Token Sparsification                       | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2410.08584) |                                                                                                                             |
| 2024 | ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification              | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2405.14256) | [Link](https://github.com/ThisisBillhe/ZipCache) ![](https://img.shields.io/github/stars/ThisisBillhe/ZipCache.svg?style=social) |
| 2024 | PrefixQuant: Static Quantization Beats Dynamic through Prefixed Outliers in LLMs                      | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2410.05265) | [Link](https://github.com/ChenMnZ/PrefixQuant) ![](https://img.shields.io/github/stars/ChenMnZ/PrefixQuant.svg?style=social)     |
| 2024 | MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache                   | Mixed-precision Quantization | arXiv | [Link](https://arxiv.org/pdf/2411.18077) |                                                                                                                             |

### Outlier Redistribution ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                        | Type                   | Venue   | Paper                                                        | code                                                         |
| ---- | ------------------------------------------------------------ | ---------------------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2024 | Massive Activations in Large Language Models                 | Outlier Redistribution | arXiv   | [Link](https://arxiv.org/pdf/2402.17762)                     | [Link](https://github.com/locuslab/massive-activations) ![](https://img.shields.io/github/stars/locuslab/massive-activations.svg?style=social) |
| 2024 | QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs         | Outlier Redistribution | arXiv   | [Link](https://arxiv.org/pdf/2404.00456)                     | [Link](https://github.com/spcl/QuaRot) ![](https://img.shields.io/github/stars/spcl/QuaRot.svg?style=social) |
| 2024 | QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving | Outlier Redistribution | arXiv   | [Link](https://arxiv.org/pdf/2405.04532)                     | [Link](https://github.com/mit-han-lab/qserve) ![](https://img.shields.io/github/stars/mit-han-lab/qserve.svg?style=social) |
| 2024 | SpinQuant: LLM Quantization with Learned Rotations           | Outlier Redistribution | arXiv   | [Link](https://arxiv.org/pdf/2405.16406)                     | [Link](https://github.com/facebookresearch/SpinQuant) ![](https://img.shields.io/github/stars/facebookresearch/SpinQuant.svg?style=social) |
| 2024 | DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs | Outlier Redistribution | NeurIPS | [Link](https://arxiv.org/pdf/2406.01721)                     | [Link](https://github.com/Hsu1023/DuQuant) ![](https://img.shields.io/github/stars/Hsu1023/DuQuant.svg?style=social) |
| 2024 | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | Outlier Redistribution | ICML    | [Link](https://proceedings.mlr.press/v202/xiao23c.html)      | [Link](https://github.com/mit-han-lab/smoothquant) ![](https://img.shields.io/github/stars/mit-han-lab/smoothquant.svg?style=social) |
| 2024 | Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling | Outlier Redistribution | EMNLP   | [Link](https://arxiv.org/pdf/2304.09145)                     | [Link](https://github.com/ModelTC/Outlier_Suppression_Plus) ![](https://img.shields.io/github/stars/ModelTC/Outlier_Suppression_Plus.svg?style=social) |
| 2024 | AffineQuant: Affine Transformation Quantization for Large Language Models | Outlier Redistribution | arXiv   | [Link](https://arxiv.org/pdf/2403.12544)                     | [Link](https://github.com/bytedance/AffineQuant) ![](https://img.shields.io/github/stars/bytedance/AffineQuant.svg?style=social) |
| 2024 | FlatQuant: Flatness Matters for LLM Quantization             | Outlier Redistribution | arXiv   | [Link](https://arxiv.org/pdf/2410.09426)                     | [Link](https://github.com/ruikangliu/FlatQuant) ![](https://img.shields.io/github/stars/ruikangliu/FlatQuant.svg?style=social) |
| 2024 | AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration | Outlier Redistribution | MLSys   | [Link](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-pdftract-Conference.html) | [Link](https://github.com/mit-han-lab/llm-awq) ![](https://img.shields.io/github/stars/mit-han-lab/llm-awq.svg?style=social) |
| 2023 | OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models | Outlier Redistribution | arXiv   | [Link](https://arxiv.org/pdf/2308.13137)                     | [Link](https://github.com/OpenGVLab/OmniQuant) ![](https://img.shields.io/github/stars/OpenGVLab/OmniQuant.svg?style=social) |
| 2023 | Training Transformers with 4-bit Integers                    | Outlier Redistribution | NeurIPS | [Link](https://proceedings.neurips.cc//paper_files/paper/2023/hash/99fc8bc48b917c301a80cb74d91c0c06-pdftract-Conference.html) | [Link](https://github.com/xijiu9/Train_Transformers_with_INT4) ![](https://img.shields.io/github/stars/xijiu9/Train_Transformers_with_INT4.svg?style=social) |

## KV Cache Low-rank Decomposition

### Singular Value Decomposition ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                       | Type                         | Venue | Paper                                 | code                                                                                                                                        |
| ---- | ------------------------------------------------------------------------------------------- | ---------------------------- | ----- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 2025 | Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression                        | Singular Value Decomposition | arXiv | [Link](https://arxiv.org/pdf/2503.02812) |                                                                                                                                             |
| 2024 | Effectively Compress KV Heads for LLM                                                       | Singular Value Decomposition | arXiv | [Link](https://arxiv.org/pdf/2406.07056) |                                                                                                                                             |
| 2024 | Eigen Attention: Attention in Low-Rank Space for KV Cache Compression                       | Singular Value Decomposition | arXiv | [Link](https://arxiv.org/pdf/2408.05646) | [Link](https://github.com/UtkarshSaxena1/EigenAttn/tree/main) ![](https://img.shields.io/github/stars/UtkarshSaxena1/EigenAttn.svg?style=social) |
| 2024 | Zero-Delay QKV Compression for Mitigating KV Cache and Network Bottlenecks in LLM Inference | Singular Value Decomposition | arXiv | [Link](https://arxiv.org/pdf/2408.04107) |                                                                                                                                             |
| 2024 | LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy        | Singular Value Decomposition | arXiv | [Link](https://arxiv.org/pdf/2410.03111) |                                                                                                                                             |
| 2024 | ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference                | Singular Value Decomposition | arXiv | [Link](https://arxiv.org/pdf/2410.21465) | [Link](https://github.com/bytedance/ShadowKV) ![](https://img.shields.io/github/stars/bytedance/ShadowKV.svg?style=social)                       |
| 2024 | Palu: Compressing KV-Cache with Low-Rank Projection                                         | Singular Value Decomposition | arXiv | [Link](https://arxiv.org/pdf/2407.21118) | [Link](https://github.com/shadowpa0327/Palu) ![](https://img.shields.io/github/stars/shadowpa0327/Palu.svg?style=social)                         |
| 2024 | Loki: Low-rank Keys for Efficient Sparse Attention                                         | Dynamic Selection without Permanent Eviction | arXiv | [Link](https://arxiv.org/pdf/2406.02542) | [Link](https://github.com/hpcgroup/loki) ![](https://img.shields.io/github/stars/hpcgroup/loki.svg?style=social)            |

### Tensor Decomposition ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                       | Type                 | Venue | Paper                                 | code                                                                                                                          |
| ---- | ------------------------------------------------------------------------------------------- | -------------------- | ----- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 2024 | Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression | Tensor Decomposition | ACL   | [Link](https://arxiv.org/pdf/2405.12591) | [Link](https://github.com/lpyhdzx/DecoQuant_code) ![](https://img.shields.io/github/stars/lpyhdzx/DecoQuant_code.svg?style=social) |

### Learned Low-rank Approximation ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                             | Type                           | Venue | Paper                                 | code                                                                                                        |
| ---- | ------------------------------------------------------------------------------------------------- | ------------------------------ | ----- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 2024 | Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference | Learned Low-rank Approximation | arXiv | [Link](https://arxiv.org/pdf/2402.09398) | [Link](https://github.com/hdong920/LESS) ![](https://img.shields.io/github/stars/hdong920/LESS.svg?style=social) |
| 2024 | MatryoshkaKV: Adaptive KV Compression via Trainable Orthogonal Projection                         | Learned Low-rank Approximation | arXiv | [Link](https://arxiv.org/pdf/2410.14731) |                                                                                                             |

---

# Model-level Optimization

## Attention Grouping and Sharing

### Intra-Layer Grouping ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                | Type              | Venue | Paper                                 | code                                                           |
| ---- | ------------------------------------------------------------------------------------ | ----------------- | ----- | ------------------------------------- | -------------------------------------------------------------- |
| 2024 | Optimised Grouped-Query Attention Mechanism for Transformers | Intra-Layer Grouping | ICML | [Link](https://openreview.net/pdf?id=13MMghY6Kh) |                                                                |
| 2024 | Weighted Grouped Query Attention in Transformers | Intra-Layer Grouping | arXiv | [Link](https://arxiv.org/pdf/2407.10855) |                                                                |
| 2024 | QCQA: Quality and Capacity-aware grouped Query Attention  | Intra-Layer Grouping | arXiv | [Link](https://arxiv.org/pdf/2406.10247) | [Non-official Link](https://github.com/vinayjoshi22/qcqa) ![](https://img.shields.io/github/stars/vinayjoshi22/qcqa.svg?style=social)|      |
| 2024 | Beyond Uniform Query Distribution: Key-Driven Grouped Query Attention | Intra-Layer Grouping | arXiv | [Link](https://arxiv.org/pdf/2408.08454) | [Link](https://github.com/zohaib-khan5040/key-driven-gqa) ![](https://img.shields.io/github/stars/zohaib-khan5040/key-driven-gqa.svg?style=social)|
| 2023 | GQKVA: Efficient Pre-training of Transformers by Grouping Queries, Keys, and Values | Intra-Layer Grouping | NeurIPS | [Link](https://arxiv.org/pdf/2311.03426) |                                                                |
| 2023 | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Intra-Layer Grouping | EMNLP | [Link](https://arxiv.org/pdf/2305.13245) | [Link](https://github.com/fkodom/grouped-query-attention-pytorch) ![](https://img.shields.io/github/stars/fkodom/grouped-query-attention-pytorch.svg?style=social)|
| 2019 | Fast Transformer Decoding: One Write-Head is All You Need                            | Intra-Layer Grouping | arXiv | [Link](https://arxiv.org/pdf/1911.02150) |                                                                |

### Cross-Layer Sharing ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title     | Type          | Venue | Paper | code |
| ---- | --------- | ------------- | ----- | ----- | ---- |
| 2024 | Reducing Transformer Key-Value Cache Size with Cross-Layer Attention | Cross-Layer Sharing | arXiv | [Link](https://arxiv.org/pdf/2405.12981) | [Non-official Link](https://github.com/JerryYin777/Cross-Layer-Attention) ![](https://img.shields.io/github/stars/JerryYin777/Cross-Layer-Attention.svg?style=social) |
| 2024 | Layer-Condensed KV Cache for Efficient Inference of Large Language Models | Cross-Layer Sharing | ACL | [Link](https://aclanthology.org/2024.acl-long.602.pdf) | [Link](https://github.LCKVcom/whyNLP/) ![](https://img.shields.io/github/stars/whyNLP/LCKV.svg?style=social) |
| 2024 | Beyond KV Caching: Shared Attention for Efficient LLMs | Cross-Layer Sharing | arXiv | [Link](https://arxiv.org/pdf/2407.12866) | [Link](https://github.com/metacarbon/shareAtt) ![](https://img.shields.io/github/stars/metacarbon/shareAtt?style=social) |
| 2024 | MLKV: Multi-Layer Key-Value Heads for Memory Efficient Transformer Decoding | Cross-Layer Sharing | arXiv | [Link](https://arxiv.org/pdf/2406.09297) | [Link](https://github.com/zaydzuhri/mlkv) ![](https://img.shields.io/github/stars/zaydzuhri/mlkv?style=social) |
| 2024 | Cross-layer Attention Sharing for Large Language Models | Cross-Layer Sharing | arXiv | [Link](https://arxiv.org/pdf/2408.01890) |      |
| 2024 | A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference | Cross-Layer Sharing | arXiv | [Link](https://arxiv.org/pdf/2410.14442) |      |
| 2024 | Lossless KV Cache Compression to 2% | Cross-Layer Sharing | arXiv | [Link](https://arxiv.org/pdf/2410.15252) |      |
| 2024 | DHA: Learning Decoupled-Head Attention from Transformer Checkpoints via Adaptive Heads Fusion | Cross-Layer Sharing |  NeurIPS | [Link](https://arxiv.org/pdf/2406.06567) |      |
| 2024 | Value Residual Learning For Alleviating Attention Concentration In Transformers | Cross-Layer Sharing | arXiv | [Link](https://arxiv.org/pdf/2410.17897) | [Link](https://github.com/Zcchill/Value-Residual-Learning) ![](https://img.shields.io/github/stars/Zcchill/Value-Residual-Learning?style=social) |

## Architecture Alteration

### Enhanced Attention ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title        | Type        | Venue | Paper | code |
| ---- | ------------ | ----------- | ----- | ----- | ---- |
| 2024 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | Enhanced Attention | arXiv | [Link](https://arxiv.org/pdf/2405.04434) | [Link](https://github.com/deepseek-ai/DeepSeek-V2) ![](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-V2?style=social) |
| 2022 | Transformer Quality in Linear Time | Enhanced Attention | ICML | [Link](https://proceedings.mlr.press/v162/hua22a/hua22a.pdf) |      |
| 2024 | Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention | Enhanced Attention | arXiv | [Link](https://arxiv.org/pdf/2404.07143) |      |

### Augmented Architecture ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title | Type        | Venue | Paper | code |
| ---- | ----- | ----------- | ----- | ----- | ---- |
| 2024 | You Only Cache Once: Decoder-Decoder Architectures for Language Models | Augmented Architecture | arXiv | [Link](https://arxiv.org/pdf/2405.05254) | [Link](https://github.com/microsoft/unilm/tree/master/YOCO) ![](https://img.shields.io/github/stars/microsoft/unilm?style=social) |
| 2024 | Long-Context Language Modeling with Parallel Context Encoding | Augmented Architectures | ACL | [Link](https://aclanthology.org/2024.acl-long.142.pdf) | [Link](https://github.com/princeton-nlp/CEPE) ![](https://img.shields.io/github/stars/princeton-nlp/CEPE?style=social) |
| 2024 | XC-CACHE: Cross-Attending to Cached Context for Efficient LLM Inference | Augmented Architectures | Findings | [Link](https://aclanthology.org/2024.findings-emnlp.896.pdf) |      |
| 2024 | Block Transformer: Global-to-Local Language Modeling for Fast Inference | Augmented Architectures | arXiv | [Link](https://arxiv.org/pdf/2406.02657) | [Link](https://github.com/itsnamgyu/block-transformer) ![](https://img.shields.io/github/stars/itsnamgyu/block-transformer?style=social) |

## Non-transformer Architecture

### Adaptive Sequence Processing Architecture ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                        | Type                                      | Venue    | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | ----------------------------------------- | -------- | ---------------------------------------- | ------------------------------------------------------------ |
| 2023 | RWKV: Reinventing RNNs for the Transformer Era               | Adaptive Sequence Processing Architecture | Findings | [Link](https://arxiv.org/pdf/2305.13048) | [Link](https://github.com/BlinkDL/RWKV-LM) ![](https://img.shields.io/github/stars/BlinkDL/RWKV-LM?style=social) |
| 2024 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Adaptive Sequence Processing Architecture | arXiv    | [Link](https://arxiv.org/pdf/2312.00752) | [Link](https://github.com/state-spaces/mamba) ![](https://img.shields.io/github/stars/state-spaces/mamba?style=social) |
| 2023 | Retentive Network: A Successor to Transformer for Large Language Models | Adaptive Sequence Processing Architecture | arXiv    | [Link](https://arxiv.org/pdf/2307.08621) | [Link](https://github.com/microsoft/unilm/tree/master/retnet) ![](https://img.shields.io/github/stars/microsoft/unilm?style=social) |
| 2024 | MCSD: An Efficient Language Model with Diverse Fusion        | Adaptive Sequence Processing Architecture | arXiv    | [Link](https://arxiv.org/pdf/2406.12230) |                                                              |

### Hybrid Architecture ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                        | Type                | Venue     | Paper                                                        | code                                                |
| ---- | ------------------------------------------------------------ | ------------------- | --------- | ------------------------------------------------------------ | --------------------------------------------------- |
| 2024 | MixCon: A Hybrid Architecture for Efficient and Adaptive Sequence Modeling | Hybrid Architecture | IOS Press | [Link](https://zhouchenlin.github.io/Publications/2024-ECAI-MixCon.pdf) |                                                     |
| 2024 | GoldFinch: High Performance RWKV/Transformer Hybrid with Linear Pre-Fill and Extreme KV-Cache Compression | Hybrid Architecture | arXiv     | [Link](https://arxiv.org/pdf/2407.12077)                     | [Link](https://github.com/recursal/GoldFinch-paper) ![](https://img.shields.io/github/stars/recursal/GoldFinch-paper?style=social) |
| 2024 | RecurFormer: Not All Transformer Heads Need Self-Attention   | Hybrid Architecture | arXiv     | [Link](https://arxiv.org/pdf/2410.12850)                     |                                                     |

---

# System-level Optimization

## Memory Management

### Architectural Design ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                        | Type                 | Venue | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | -------------------- | ----- | ---------------------------------------- | ------------------------------------------------------------ |
| 2025 | eLLM: Elastic Memory Management Framework for Efficient LLM Serving   | Architectural Design | arXiv | [Link](https://arxiv.org/pdf/2506.15155) | |
| 2025 | Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving | Architectural Design | ACM | [Link](https://dl.acm.org/doi/10.1145/3725394) |  |
| 2024 | vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving | Architectural Design | arXiv | [Link](https://arxiv.org/pdf/2407.15309) | [Link](https://github.com/antgroup/glake) ![](https://img.shields.io/github/stars/antgroup/glake.svg?style=social) |
| 2024 | Unifying KV Cache Compression for Large Language Models with LeanKV | Architectural Design | arXiv | [Link](https://arxiv.org/pdf/2412.03131) |                                                              |
| 2023 | Efficient Memory Management for Large Language Model Serving with PagedAttention | Architectural Design | SOSP  | [Link](https://arxiv.org/pdf/2309.06180) | [Link](https://github.com/vllm-project/vllm) ![](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social) |

### Prefix-aware Design ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                        | Type                | Venue | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | ------------------- | ----- | ---------------------------------------- | ------------------------------------------------------------ |
| 2025 | FlashForge: Ultra-Efficient Prefix-Aware Attention for LLM Decoding                         | Prefix-aware Design | arXiv | [Link](https://arxiv.org/pdf/2505.17694) |                                                              |
| 2024 | ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition | Prefix-aware Design | ACL   | [Link](https://arxiv.org/pdf/2402.15220) | [Link](https://github.com/microsoft/chunk-attention) ![](https://img.shields.io/github/stars/microsoft/chunk-attention.svg?style=social) |
| 2024 | MemServe:FlexibleMemPoolforBuilding DisaggregatedLLMServingwithCaching | Prefix-aware Design | arXiv | [Link](https://arxiv.org/pdf/2406.17565) |                                                              |

## Scheduling

### Prefix-aware Scheduling ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                        | Type                    | Venue   | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | ----------------------- | ------- | ---------------------------------------- | ------------------------------------------------------------ |
| 2025 | Echo: Efficient Co-Scheduling of Hybrid Online-Offline Tasks for Large Language Model Serving                      | Prefix-aware Scheduling | arXiv   | [Link](https://arxiv.org/pdf/2504.03651) |                                                              |
| 2024 | BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching | Prefix-aware Scheduling | arXiv   | [Link](https://arxiv.org/pdf/2412.03594) |                                                              |
| 2024 | SGLang: Efficient Execution of Structured Language Model Programs | Prefix-aware Scheduling | NeurIPS | [Link](https://arxiv.org/pdf/2312.07104) | [Link](https://github.com/sgl-project/sglang) ![](https://img.shields.io/github/stars/sgl-project/sglang.svg?style=social) |

### Preemptive and Fairness-oriented Scheduling ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                        | Type                                        | Venue | Paper                                    | code |
| ---- | ------------------------------------------------------------ | ------------------------------------------- | ----- | ---------------------------------------- | ---- |
| 2025 | FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling | Preemptive and Fairness-oriented Scheduling | arXiv | [Link]([https://arxiv.org/pdf/2305.05920](https://arxiv.org/pdf/2504.03775)) |      |
| 2024 | FASTSWITCH: OPTIMIZING CONTEXT SWITCHING EFFICIENCY IN FAIRNESS-AWARE LARGE LANGUAGE MODEL SERVING | Preemptive and Fairness-oriented Scheduling | arXiv | [Link](https://arxiv.org/pdf/2411.18424) |      |
| 2023 | Fast Distributed Inference Serving for Large Language Models | Preemptive and Fairness-oriented Scheduling | arXiv | [Link](https://arxiv.org/pdf/2305.05920) |      |

### Layer-specific and Hierarchical Scheduling ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                        | Type                                       | Venue      | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------ | ---------- | ---------------------------------------- | ------------------------------------------------------------ |
| 2025 | Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving | Architectural Design | ACM | [Link](https://dl.acm.org/doi/10.1145/3725394) |  |
| 2025 | Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints          | Architectural Design | arXiv | [Link](https://arxiv.org/pdf/2504.11320) |  |
| 2024 | LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Management | Layer-specific and Hierarchical Scheduling | arXiv      | [Link](https://arxiv.org/pdf/2410.00428) | [Link](https://github.com/antgroup/glake) ![](https://img.shields.io/github/stars/antgroup/glake.svg?style=social) |
| 2024 | Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention | Layer-specific and Hierarchical Scheduling | USENIX ATC | [Link](https://arxiv.org/pdf/2403.19708) |                                                              |
| 2024 | ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching | Layer-specific and Hierarchical Scheduling | ISCA       | [Link](https://arxiv.org/pdf/2403.17312) |                                                              |
| 2024 | Fast Inference for Augmented Large Language Models           | Layer-specific and Hierarchical Scheduling | arXiv      | [Link](https://arxiv.org/pdf/2410.18248) |                                                              |

## Hardware-aware Design

### Single/Multi-GPU Design ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                             | Type                    | Venue | Paper                                                            | code                                                                                                                                      |
| ---- | ------------------------------------------------------------------------------------------------- | ----------------------- | ----- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 2025 | gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling | Single/Multi-GPU Design | arXiv | [Link](https://arxiv.org/pdf/2504.14775)                            |  |
| 2025 | FairKV: Balancing Per-Head KV Cache for Fast Multi-GPU Inference                                    | Single/Multi-GPU Design | arXiv | [Link](https://arxiv.org/pdf/2502.15804)                            |  |
| 2025 | Mell: Memory-Efficient Large Language Model Serving via Multi-GPU KV Cache Management               | Single/Multi-GPU Design | arXiv | [Link](https://arxiv.org/pdf/2501.06709)                            |  |
| 2024 | Hydragen: High-Throughput LLM Inference with Shared Prefixes                                      | Single/Multi-GPU Design | arXiv | [Link](https://arxiv.org/pdf/2402.05099)                            | [Link](https://github.com/ScalingIntelligence/hydragen) ![](https://img.shields.io/github/stars/ScalingIntelligence/hydragen.svg?style=social) |
| 2024 | DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference              | Single/Multi-GPU Design | arXiv | [Link](https://arxiv.org/pdf/2404.00242)                            |                                                                                                                                           |
| 2024 | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | Single/Multi-GPU Design | OSDI  | [Link](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) | [Link](https://github.com/LLMServe/DistServe) ![](https://img.shields.io/github/stars/LLMServe/DistServe.svg?style=social)                     |
| 2024 | Multi-Bin Batching for Increasing LLM Inference Throughput                                        | Single/Multi-GPU Design | arXiv | [Link](https://openreview.net/pdf?id=WVmarX0RNd)                    |                                                                                                                                           |
| 2024 | Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters                | Single/Multi-GPU Design | arXiv | [Link](https://arxiv.org/pdf/2408.04093)                            | [Link](https://github.com/Zyphra/tree_attention) ![](https://img.shields.io/github/stars/Zyphra/tree_attention.svg?style=social)               |
| 2023 | Efficient Memory Management for Large Language Model Serving with PagedAttention                  | Single/Multi-GPU Design | SOSP  | [Link](https://arxiv.org/pdf/2309.06180)                            | [Link](https://github.com/vllm-project/vllm) ![](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social)                       |
| 2022 | Orca: A Distributed Serving System for Transformer-Based Generative Models                        | Single/Multi-GPU Design | OSDI  | [Link](https://www.usenix.org/system/files/osdi22-yu.pdf)           |                                                                                                                                           |


### I/O-based Design ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                              | Type             | Venue   | Paper                                                                                                                 | code                                                                                                                                                                                      |
| ---- | -------------------------------------------------------------------------------------------------- | ---------------- | ------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2024 | Bifurcated Attention: Accelerating Massively Parallel Decoding with Shared Prefixes in LLMs        | I/O-based Design | arXiv   | [Link](https://arxiv.org/pdf/2403.08845)                                                                                 | [Link](https://github.com/bifurcated-attn-icml-2024/gpt-fast-parallel-sampling) ![](https://img.shields.io/github/stars/bifurcated-attn-icml-2024/gpt-fast-parallel-sampling.svg?style=social) |
| 2024 | Efficient LLM Inference with I/O-Aware Partial KV Cache Recomputation                              | I/O-based Design | arXiv   | [Link](https://arxiv.org/pdf/2411.17089)                                                                                 |                                                                                                                                                                                           |
| 2024 | Fast State Restoration in LLM Serving with HCache                                                  | I/O-based Design | arXiv   | [Link](https://arxiv.org/pdf/2410.05004)                                                                                 |                                                                                                                                                                                           |
| 2024 | Compute Or Load KV Cache? Why Not Both?                                                            | I/O-based Design | arXiv   | [Link](https://arxiv.org/pdf/2410.03065)                                                                                 |                                                                                                                                                                                           |
| 2024 | FastSwitch: Optimizing Context Switching Efficiency in Fairness-aware Large Language Model Serving | I/O-based Design | arXiv   | [Link](https://arxiv.org/pdf/2411.18424)                                                                                 |                                                                                                                                                                                           |
| 2024 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision                    | I/O-based Design | arXiv   | [Link](https://arxiv.org/pdf/2407.08608)                                                                                 | [Link](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)                                                       |
| 2023 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning                   | I/O-based Design | arXiv   | [Link](https://arxiv.org/pdf/2307.08691)                                                                                 | [Link](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)                                                       |
| 2022 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness                        | I/O-based Design | NeurIPS | [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf) | [Link](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)                                                       |

### Heterogeneous Design ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                               | Type                 | Venue | Paper                                 | code |
| ---- | --------------------------------------------------------------------------------------------------- | -------------------- | ----- | ------------------------------------- | ---- |
| 2025 | HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading                                   | Heterogeneous Design | arXiv | [Link](https://arxiv.org/pdf/2502.12574) |      |
| 2025 | Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs                                    | Heterogeneous Design | arXiv | [Link](https://arxiv.org/abs/2506.03296) |      |
| 2024 | NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference                          | Heterogeneous Design | arXiv | [Link](https://arxiv.org/pdf/2411.01142) |      |
| 2024 | FASTDECODE: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines                 | Heterogeneous Design | arXiv | [Link](https://arxiv.org/pdf/2403.11421) |      |
| 2024 | vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving                               | Heterogeneous Design | arXiv | [Link](https://arxiv.org/pdf/2407.15309) |      |
| 2024 | InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management | Heterogeneous Design | arXiv | [Link](https://arxiv.org/pdf/2406.19707) |      |
| 2024 | Fast Distributed Inference Serving for Large Language Models                                        | Heterogeneous Design | arXiv | [Link](https://arxiv.org/pdf/2305.05920) |      |
| 2024 | Efficient LLM Inference with I/O-Aware Partial KV Cache Recomputation                               | Heterogeneous Design | arXiv | [Link](https://arxiv.org/pdf/2411.17089) |      |
| 2023 | Stateful Large Language Model Serving with Pensieve                                                 | Heterogeneous Design | arXiv | [Link](https://arxiv.org/pdf/2312.05516) |      |

### SSD-based Design ([To Top👆🏻](#awesome-kv-cache-management))

| Year | Title                                                                                    | Type             | Venue | Paper                                                         | code                                                                                                                          |
| ---- | ---------------------------------------------------------------------------------------- | ---------------- | ----- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 2024 | InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference | SSD-based Design | arXiv | [Link](https://arxiv.org/pdf/2409.04992)                         |                                                                                                                               |
| 2023 | FlexGen: High-Throughput Generative Inference of Large Language Models                   | SSD-based Design | ICML  | [Link](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf) | [Link](https://github.com/FMInference/FlexLLMGen) ![](https://img.shields.io/github/stars/FMInference/FlexLLMGen.svg?style=social) |

---

# Datasets and Benchmarks

Please refer to our paper for detailed information on this section.

---
