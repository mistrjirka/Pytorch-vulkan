"""Rebuild and validate the portable activation, mutation and loss shaders."""
from pathlib import Path
import subprocess

SHADERS = (
    'leaky_relu_native', 'tensor_add_inplace', 'loss_unary', 'loss_binary',
    'loss_mutate', 'loss_log_sigmoid', 'loss_log_sigmoid_backward',
    'loss_mean_reduce', 'nll_loss_forward', 'nll_loss_backward',
)


def main():
    directory = Path(__file__).resolve().parents[1] / 'csrc' / 'shaders'
    for name in SHADERS:
        target = directory / f'{name}.spv'
        subprocess.run(['glslangValidator', '-V', str(directory / f'{name}.comp'), '-o', str(target)], check=True)
        subprocess.run(['spirv-val', str(target)], check=True)


if __name__ == '__main__':
    main()
