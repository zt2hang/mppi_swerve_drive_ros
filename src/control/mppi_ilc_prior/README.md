# mppi_ilc_prior

`mppi_ilc_prior` injects a learned ILC feedforward as a *control prior* into `mppi_hc`.

Key difference vs post-hoc bias addition:
- MPPI samples around a prior-centered mean control, improving sample efficiency.
- The returned command and trajectory use the combined control (optimal residual + prior).

## Run

```bash
roslaunch mppi_ilc_prior mppi_ilc_prior.launch
```

Parameters are in `config/mppi_ilc_prior.yaml`.
