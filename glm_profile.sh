#!/usr/bin/env bash
# Shared topology and memory defaults. Source from either serve script.
PP=${PP:-12}
TP=${TP:-1}
SPEC_TOKENS=${SPEC_TOKENS:-3}
if ! [[ "$PP" =~ ^[1-9][0-9]*$ && "$TP" =~ ^[1-9][0-9]*$ &&
        "$SPEC_TOKENS" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "PP/TP must be positive integers; SPEC_TOKENS must be a nonnegative integer." >&2
  exit 1
fi
if [ "$TP" -ne 1 ]; then
  echo "This CMP profile requires TP=1 (the sparse-MLA backend and PP patches target TP=1)." >&2
  exit 1
fi

_world_size=$((PP * TP))
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
  CUDA_VISIBLE_DEVICES=0
  for (( _gpu=1; _gpu<_world_size; _gpu++ )); do
    CUDA_VISIBLE_DEVICES+=",$_gpu"
  done
fi
if ! [[ "$CUDA_VISIBLE_DEVICES" =~ ^[^,[:space:]]+(,[^,[:space:]]+)*$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must be a nonempty comma-separated device list." >&2
  exit 1
fi
IFS=',' read -r -a _cuda_devices <<< "$CUDA_VISIBLE_DEVICES"
_seen_devices=,
for _gpu in "${_cuda_devices[@]}"; do
  if [ "$_gpu" = -1 ] || [[ "$_seen_devices" == *",$_gpu,"* ]]; then
    echo "CUDA_VISIBLE_DEVICES contains a disabled or repeated device: $_gpu" >&2
    exit 1
  fi
  _seen_devices+="$_gpu,"
done
if [ "${#_cuda_devices[@]}" -ne "$_world_size" ]; then
  echo "CUDA_VISIBLE_DEVICES exposes ${#_cuda_devices[@]} device(s), but PP*TP=$_world_size." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES

# Retain the indexer-aware split with MTP off too: vLLM's even split can
# put more cache on rank 0 and silently reduce the advertised capacity.
if [ -z "${VLLM_PP_LAYER_PARTITION:-}" ]; then
  case "$PP" in
    12) VLLM_PP_LAYER_PARTITION=6,7,7,7,7,7,7,7,7,6,6,4 ;;
    8)  VLLM_PP_LAYER_PARTITION=11,10,10,10,10,10,10,7 ;;
    *)
      echo "Set VLLM_PP_LAYER_PARTITION for PP=$PP." >&2
      exit 1 ;;
  esac
fi
if ! [[ "$VLLM_PP_LAYER_PARTITION" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  echo "VLLM_PP_LAYER_PARTITION must be comma-separated positive integers (no leading zeros)." >&2
  exit 1
fi
IFS=',' read -r -a _pp_layers <<< "$VLLM_PP_LAYER_PARTITION"
_layer_sum=0
for _layers in "${_pp_layers[@]}"; do
  _layer_sum=$((_layer_sum + _layers))
done
if [ "${#_pp_layers[@]}" -ne "$PP" ] || [ "$_layer_sum" -ne 78 ]; then
  echo "VLLM_PP_LAYER_PARTITION must contain $PP entries summing to 78." >&2
  exit 1
fi
export VLLM_PP_LAYER_PARTITION

# 20 GiB is a starting estimate for the built-in twelve-card layout.
# Seven MoE layers replace ten on the formerly tight middle ranks, freeing
# roughly 15 GiB of weights; this spends 12.31 GiB more than the old PP=8 KV
# budget, retaining a margin for graph capture and long-prefill transients.
# Custom partitions use memory profiling unless the user supplies a budget.
if [ -z "${KV_CACHE_MEM+x}" ]; then
  case "$PP:$VLLM_PP_LAYER_PARTITION" in
    12:6,7,7,7,7,7,7,7,7,6,6,4) KV_CACHE_MEM=21474836480 ;;
    8:11,10,10,10,10,10,10,7) KV_CACHE_MEM=8258584576 ;;
    *) KV_CACHE_MEM= ;;
  esac
fi
if [ -n "$KV_CACHE_MEM" ] && ! [[ "$KV_CACHE_MEM" =~ ^[1-9][0-9]*$ ]]; then
  echo "KV_CACHE_MEM must be a positive integer in bytes, or empty for GPU_UTIL profiling." >&2
  exit 1
fi
