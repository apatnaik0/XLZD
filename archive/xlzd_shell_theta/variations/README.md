# Shell Theta Variations

This folder contains five shell-theta variation experiments built on top of the base `xlzd_shell_theta` workflow.

Each variation has:

- a JSON preprocessing config under `config/`
- a training and validation CNP settings YAML under `settings/`
- a shared preparation driver:
  - `run_variation.py`
- one combined notebook that follows the same CNP and MF-GP plotting pattern as the base shell-theta workflows

Variations:

1. `method1_larger_delta`
   - larger `delta_r`, `delta_z`
2. `method2_smaller_grid`
   - smaller `r_shell_step`, `z_shell_step`
3. `method3_asymmetric_delta`
   - asymmetric `delta_r`, `delta_z`
4. `method4_higher_support`
   - larger `min_candidate_events`
5. `method5_soft_shell`
   - soft Gaussian shell target

Use the preparation driver from the repo root:

```bash
python3 xlzd_shell_theta/variations/run_variation.py --variation method1_larger_delta --stage prepare_convert
```

Then open the corresponding notebook and run the CNP + MF-GP cells.
