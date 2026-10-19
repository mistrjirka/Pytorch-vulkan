"""Report scope blockers without changing any original upstream test outcome.

This is the agreed FP32/integer/bool audit scope, including complex64 spectral
and eigen intermediates. It does not declare skips/xfails/xpasses to be passes.
Fallback counters are inspected for every status, including expected failures.
The alpha-dropout exception has matched exact-fixture CUDA/Vulkan evidence in
checkpoint 11. Weighted integral bincount's Float64 output proof is in checkpoint
12. Neither exception suppresses a numerical CPU fallback in an in-scope case.
"""
import argparse
import collections
import json
from pathlib import Path
import re


ALPHA_MASK_CASE = 'TestCommonPRIVATEUSE1.test_compare_cpu_nn_functional_alpha_dropout_vulkan_float32'
DEFERRED = re.compile(r'(?:target dtype: |input dtype: |got )(Double|Half|BFloat16|ComplexDouble|ComplexHalf|Float8_\w+)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    args = parser.parse_args()
    rows = [json.loads(s) for s in (args.audit/'results.jsonl').read_text().splitlines()]
    summary = json.loads((args.audit/'summary.json').read_text())
    assert len(rows) == summary['total_cases'] == summary['original_selection_count']
    assert len({r['id'] for r in rows}) == len(rows)
    assert summary['complete_duplicate_free_selection']
    ledger = []
    upstream_expected = []
    fallback_by_status = collections.Counter()
    successful_fallbacks = 0
    for row in rows:
        status = row['status']
        calls = row.get('execution', {}).get('cpu_fallback_calls', 0)
        fallback_by_status[status] += calls
        if status == 'expected_failure' and not calls:
            upstream_expected.append(row['id'])
            continue
        if status in ('passed', 'unexpected_success', 'skipped') and not calls:
            continue
        if status in ('passed', 'unexpected_success'):
            successful_fallbacks += calls
        details = '\n'.join(row.get('details', []))
        category = 'unresolved'
        reason = 'An in-scope failure or fallback attempt requires investigation.'
        if 'Jiterator is only supported on CUDA and ROCm' in details:
            category = 'cuda_source_jit_gate'
            reason = 'Upstream literal-source Jiterator API is CUDA/ROCm-only, outside the audited ATen/default-compile contracts.'
        elif row['id'] == ALPHA_MASK_CASE and not calls and 'Mismatched elements:' in details:
            category = 'cross_device_rng_mask_comparison'
            reason = 'The exact original fixture also differs from CPU on CUDA; affine outputs and mask gradients pass on both GPUs. Upstream xfails CUDA/XPU.'
        elif 'Vulkan weighted bincount requires FP32 weights' in details:
            category = 'deferred_precision'
            reason = 'Original integral-weight samples require a Float64 weighted output on CPU; independent original-sample proof is retained.'
        else:
            match = DEFERRED.search(details)
            if match:
                category = 'deferred_precision'
                reason = 'The failing operation explicitly requires '+match[1]+'. Later unvisited samples remain unverified in this raw case.'
        ledger.append(dict(id=row['id'], status=status, category=category, reason=reason,
                           fallback_calls_or_attempts=calls, device=row.get('execution_device'),
                           details=row.get('details', [])))
    counts = dict(collections.Counter(r['category'] for r in ledger))
    unresolved = [r['id'] for r in ledger if r['category'] == 'unresolved']
    result = dict(summary, scope_classifications=counts, unresolved_cases=unresolved,
                  fallback_calls_or_attempts_by_raw_status=dict(fallback_by_status),
                  successful_case_fallback_calls_or_attempts=successful_fallbacks,
                  upstream_expected_failures_without_fallback=upstream_expected,
                  completion_gate_passed=not unresolved and not successful_fallbacks,
                  scope_note='Deferred precision and upstream API/RNG controls are not passes. No original outcome is changed.')
    (args.audit/'scope-ledger.json').write_text(json.dumps(ledger, indent=2)+'\n')
    (args.audit/'scope-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['raw_outcomes','scope_classifications','unresolved_cases','fallback_calls_or_attempts_by_raw_status','completion_gate_passed']}), flush=True)
    if not result['completion_gate_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
