#!/usr/bin/env bash
# Smoke-test portable FP32 Vulkan Conv2D and gradients on a physical AMD GPU.
# Never silently accept Lavapipe or a different vendor as an AMD test.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python executable not found: %s\nBuild/install the backend first (see README.md).\n' "$PYTHON_BIN" >&2
  exit 2
fi

# Current Arch vulkan-radeon includes radeon_icd.json. Respect caller's
# selection and avoid silently forcing the wrong ICD on other distributions.
if [[ -z "${VK_DRIVER_FILES:-}" && -z "${VK_ICD_FILENAMES:-}" \
      && -f /usr/share/vulkan/icd.d/radeon_icd.json ]]; then
  export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json
fi

export TORCH_VULKAN_PHYSICAL_DEVICE_INDEX="${TORCH_VULKAN_PHYSICAL_DEVICE_INDEX:-0}"
export TORCH_VULKAN_GPU_RESIDENT=1
export TORCH_VULKAN_DEFER_SUBMISSIONS=1
export TORCH_VULKAN_FORBID_CPU_FALLBACK=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
try:
    import torch_vulkan
    name = torch_vulkan.device_name()
except Exception as exc:
    raise SystemExit(
        "Failed to initialize the AMD Vulkan driver; inspect VK_DRIVER_FILES, "
        "run 'vulkaninfo --summary', and verify vulkan-radeon is installed. "
        f"Original error: {exc}"
    ) from exc
print("Selected Vulkan GPU:", name, flush=True)
assert torch_vulkan.is_hardware_accelerated(), "Software Vulkan is not an AMD test"
assert any(token in name.lower() for token in ("radeon", "amd", "navi", "gfx")), (
    "The selected Vulkan GPU is not identified as AMD; choose the proper ICD/device"
)
PY

"$PYTHON_BIN" -m pytest -q \
  tests/test_conv_dispatch.py \
  tests/test_conv_universality.py \
  tests/test_convolution.py

if [[ "${AMD_VULKAN_BENCH_OUTPUT:-}" != "" ]]; then
  "$PYTHON_BIN" benchmarks/probe_conv_dispatch.py \
    --backend vulkan \
    --warmup 8 --repeats 25 \
    --out "$AMD_VULKAN_BENCH_OUTPUT"
fi
