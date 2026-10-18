"""Capture untouched-weight output, loss and every gradient for matched backends."""
import argparse
import hashlib
from pathlib import Path
import sys
import torch
from compare_cuda_vulkan import CONFIG, data_for_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cpu', 'cuda', 'vulkan'], required=True)
    p.add_argument('--model', choices=CONFIG, required=True)
    p.add_argument('--scale', choices=['tiny', 'small', 'medium', 'reference'], default='medium')
    p.add_argument('--zoo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.backend == 'vulkan':
        import torch_vulkan
    sys.path.insert(0, str((a.zoo / 'benchmarks').resolve()))
    import model_zoo
    torch.manual_seed(42)
    model, _, loss_fn = model_zoo.MODELS[a.model]()
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode()); digest.update(tensor.contiguous().numpy().tobytes())
    x, y = data_for_case(a.model, CONFIG[a.model][a.scale], torch.Generator().manual_seed(123))
    input_hash = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
    model.to(a.backend).train()
    output = model(x.to(a.backend))
    loss = loss_fn(output, y.to(a.backend))
    loss.backward()
    result = dict(model=a.model, backend=a.backend, model_seed=42, input_seed=123,
                  initial_state_sha256=digest.hexdigest(), input_sha256=input_hash,
                  output=output.detach().cpu(), loss=loss.detach().cpu(),
                  gradients={n: p.grad.cpu() for n, p in model.named_parameters() if p.grad is not None})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, a.out)
    print(a.model, a.backend, 'loss', float(result['loss']), 'gradients', len(result['gradients']))


if __name__ == '__main__':
    main()
