# Benchmarking and model-zoo smoke tests

**[GPU comparison report: Vulkan vs CUDA vs CPU](GPU_COMPARISON.md)** includes 8 architectures, measured FP32 inference and full AdamW training, medium-resolution image inputs, and optimizer-stage profiling.


This project intentionally **imports the actual model definitions** from
[mistrjirka/pytorch-XDNA2](https://github.com/mistrjirka/pytorch-XDNA2);
no copies of its network implementations are vendored.

The upstream benchmark compares **BF16 CPU vs BF16 XDNA2**, with larger
batch/spatial sizes. Vulkan doesn't support BF16 yet, so this is a
**reduced-size FP32 correctness and speed probe, not an equivalent benchmark**.
It compares forward output, gradients, one AdamW step, and forward+backward
CPU versus real Vulkan. Errors are absolute maxima.

```sh
git clone https://github.com/mistrjirka/pytorch-XDNA2.git ../pytorch-XDNA2
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    'torchvision==0.29.1+cpu'
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=0 \
python benchmarks/model_zoo_smoke.py --zoo ../pytorch-XDNA2 \
    --models mlp,unet,embedding,style_net --repeats 3 \
    --json benchmarks/results/rtx2080ti_model_zoo_fp32.json
```

The sample result in `results/` uses XDNA2 commit
`c7c063101ec5209caf8022f5b4ea6948d1a8d6ed` from 2026-10-15.
The scripts require PyTorch 2.14.1, torchvision for StyleNet, and a
functional Vulkan driver.

Known unsupported models: LSTM (missing fused LSTM cell),
convolutional autoencoder (missing transposed convolution). The reduced
ResNet18 probe can show substantial gradient sensitivity at tiny spatial
sizes; it is excluded from the validated four-model summary.
