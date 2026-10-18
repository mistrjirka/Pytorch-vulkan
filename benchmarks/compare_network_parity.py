"""Validate two captures, including every gradient; write compact error metrics."""
import argparse
import json
from pathlib import Path
import torch


def metrics(actual, expected):
    assert actual.shape == expected.shape
    a, e = actual.double(), expected.double()
    diff = a - e
    return dict(max_abs=float(diff.abs().max()) if diff.numel() else 0.,
                relative_l2=float(diff.norm() / e.norm().clamp_min(1e-30)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rtol', type=float, default=5e-4)
    p.add_argument('--atol', type=float, default=2e-5)
    args = p.parse_args()
    a, e = (torch.load(path, map_location='cpu', weights_only=True)
            for path in (args.actual, args.reference))
    for key in ('model', 'initial_state_sha256', 'input_sha256'):
        assert a[key] == e[key], f'{key} differs'
    assert a['gradients'].keys() == e['gradients'].keys()
    result = dict(model=a['model'], actual=str(args.actual), reference=str(args.reference),
                  rtol=args.rtol, atol=args.atol, initial_state_sha256=a['initial_state_sha256'],
                  input_sha256=a['input_sha256'], gradients={})
    pairs = [('output', a['output'], e['output']), ('loss', a['loss'], e['loss'])]
    pairs += [(f'gradient/{n}', a['gradients'][n], e['gradients'][n]) for n in a['gradients']]
    failures = []
    for name, got, expected in pairs:
        stat = metrics(got, expected)
        try:
            torch.testing.assert_close(got, expected, rtol=args.rtol, atol=args.atol)
            stat['passed'] = True
        except AssertionError as error:
            stat['passed'] = False
            failures.append(f'{name}: {error}')
        if name.startswith('gradient/'):
            result['gradients'][name[9:]] = stat
        else:
            result[name] = stat
    result['passed'] = not failures
    result['gradient_count'] = len(a['gradients'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    if failures:
        raise AssertionError('\n'.join(failures))
    print(a['model'], result['gradient_count'], 'gradients matched', str(args.reference))


if __name__ == '__main__':
    main()
