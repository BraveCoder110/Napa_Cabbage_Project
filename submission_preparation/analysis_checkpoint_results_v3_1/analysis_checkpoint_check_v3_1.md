# Analysis-only ResNet18 Checkpoint Functional Check (V3.1)

**Status:** PASS

This is a read-only structural and CPU functional test, not independent validation.

Existing checkpoints were never modified.

## Checks

- checkpoint_safe_load_and_tensor_schema: PASS
- analysis_checkpoint_strict_load_into_project_architecture: PASS
- deterministic_cpu_forward_finite_output: PASS

## Tensor schema

```json
{
  "source_key_count": 182,
  "analysis_key_count": 182,
  "missing_in_analysis": [],
  "extra_in_analysis": [],
  "shape_mismatch": [],
  "exact_key_and_shape_match": true
}
```

## CPU inference

```json
{
  "shape": [
    1,
    1,
    128,
    128
  ],
  "raw_min": -1.5878890752792358,
  "raw_max": -0.5490654706954956,
  "raw_mean": -1.2883986234664917,
  "all_values_between_0_and_1": false,
  "parameter_count": 14328081
}
```
