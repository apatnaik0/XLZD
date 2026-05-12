# Shell Theta Concrete Experiment Plan

## 1. Theta definition

Define:

- `theta = (r_shell, z_shell)`

where:

- `r_shell` is a final-position radial coordinate
- `z_shell` is a final-position centered axial coordinate

Both live in the same `(r, z_from_center)` space already used in XLZD.

## 2. Event label definition

For each event and each shell theta, define:

- `near_shell = 1` if:
  - `|r - r_shell| <= delta_r`
  - `|z_from_center - z_shell| <= delta_z`
- else `near_shell = 0`

This creates a local occupancy target rather than a cumulative inclusion target.

## 3. Why this should help

The old target has nested structure:

- larger `(R_max, Z_max)` always contain smaller ones

That makes nearby theta points heavily dependent.

The shell target is more local:

- each theta measures occupancy near one shell-centered region
- theta points are less trivially monotone

This should make:

- CNP learn a more local response surface
- MF-GP fit a more meaningful theta landscape

## 4. First-pass shell grid

Use a regular grid over the observed support:

- `r_shell` from `0` to `R_max_global`
- `z_shell` from `0` to `Z_max_global`

Recommended first-pass bounds:

- `R_max_global = 1500`
- `Z_max_global = 2000`

Recommended first-pass grid:

- `r_shell_grid = 0, 100, 200, ..., 1500`
- `z_shell_grid = 0, 100, 200, ..., 2000`

This should be adjusted later if the target becomes too sparse or too dense.

## 5. First-pass shell widths

Use fixed widths:

- `delta_r = 50`
- `delta_z = 50`

Reason:

- these are half the suggested grid spacing
- neighboring shell bins touch without becoming too cumulative

This is the most defensible first experiment.

## 6. Data-generation workflow

For each raw event pool block:

1. compute final-position `r`
2. compute final-position `z_from_center`
3. assign one shell theta `(r_shell, z_shell)`
4. compute `near_shell`
5. write one file per shell theta block

This mirrors the current theta-block setup, but with shell-local labels instead of volume inclusion labels.

## 7. CNP workflow

Keep:

- `phi = (s_r, s_z_from_center)`

Use:

- `theta = (r_shell, z_shell)`
- `target = near_shell`

Then the CNP learns:

- probability that an event starting at `(s_r, s_z_from_center)` ends near shell `(r_shell, z_shell)`

This is a local final-position occupancy model.

## 8. MF-GP workflow

Aggregate CNP outputs per shell theta:

- `y_raw = mean(near_shell)`
- `y_cnp = mean(predicted probability of near_shell)`
- `y_cnp_err = aggregated prediction uncertainty`

Then run MF-GP exactly as before, but over shell theta space:

- input theta = `(r_shell, z_shell)`

The target surface now becomes:

- local shell occupancy fraction

instead of cumulative inside-volume fraction.

## 9. Filtering step

Before committing to the grid, filter shell theta points that are too sparse.

Recommended rule:

- discard shell theta points where the mean positive fraction is effectively zero across almost all blocks

This avoids training on overwhelmingly empty shell bins.

## 10. First comparison experiments

Run three versions:

1. shell theta with:
   - `delta_r = 50`
   - `delta_z = 50`

2. shell theta with slightly wider bands:
   - `delta_r = 75`
   - `delta_z = 75`

3. shell theta with slightly narrower bands:
   - `delta_r = 40`
   - `delta_z = 40`

This will tell you whether the first-pass shell target is too sparse or too overlapping.

## 11. Success criteria

The shell-theta definition is worth keeping if:

- target distributions are not trivially all zero
- shell theta points are visibly less nested than the old cumulative target
- CNP can learn a meaningful shell-local response
- MF-GP surfaces vary in a nontrivial local way
- uncertainty becomes more interpretable across theta

## 12. Main risks

### Too sparse

If `delta_r`, `delta_z` are too small:

- most shell targets become nearly zero

### Too overlapping

If `delta_r`, `delta_z` are too large:

- shell bins start behaving like cumulative regions again

### Too many theta points

A dense grid may produce too many shell files.

So first pass should stay moderate.

## 13. Recommended implementation order

1. write shell-theta data constructor
2. create CSV/parquet shell-theta block files
3. convert them to H5
4. run one small CNP pilot
5. inspect positive-fraction distribution
6. tune `delta_r`, `delta_z`
7. run full CNP
8. run MF-GP

## 14. Recommendation

This is a good experiment.

It is more local, less nested, and more physically interpretable than the current cumulative-volume theta if the goal is to understand where events populate final-position space rather than how many are included by larger and larger detector regions.
