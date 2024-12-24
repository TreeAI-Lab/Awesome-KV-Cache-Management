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
|||||||

### Cross Layer 

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|||||||

### Non-transformer Architecture

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|||||||

---

## System-level Optimization

### Memory Management

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|||||||

### Scheduling

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|||||||

### Hardware-aware Design

| Year | Title | Type | Venue | Paper | code |
| ---- | ----- | ---- | ----- | ----- | ---- |
|||||||

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