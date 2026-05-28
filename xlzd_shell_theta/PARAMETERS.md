# Recommended First-Pass Parameters

## Global bounds

- `R_max_global = 1500`
- `Z_max_global = 2000`

## Shell grid

- `r_shell_step = 100`
- `z_shell_step = 100`

## Shell widths

- `delta_r = 50`
- `delta_z = 50`

## Alternate width sweep

- narrow:
  - `delta_r = 40`
  - `delta_z = 40`

- wide:
  - `delta_r = 75`
  - `delta_z = 75`

## CNP definition

- `theta = (r_shell, z_shell)`
- `phi = (s_r, s_z_from_center)`
- `target = near_shell`

## Aggregated outputs

- `y_raw = mean(near_shell)`
- `y_cnp = mean(predicted probability of near_shell)`
- `y_cnp_err = aggregated event-level uncertainty`

## Initial pilot recommendation

Do a small pilot before a full run:

- use a reduced shell grid
- use one shell width pair first:
  - `delta_r = 50`
  - `delta_z = 50`

Check:

- fraction of shell theta files with nontrivial positive occupancy
- positive-fraction histogram
- whether the shell target is too sparse

Only then run the full grid.
