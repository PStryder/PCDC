# Long-Session Analysis: long-session-v1-transcript

## Run Metadata
- **Started**: 2026-02-28T13:04:59.038691
- **Finished**: 2026-02-28T14:14:57.807646
- **Total turns**: 210
- **Errors**: 0
- **Server**: http://localhost:8000
- **Max tokens**: 80
- **Delay**: 2.0s
- **Scenarios**: complexity_ladder, general, gradual_drift, rapid_switching, repetition_probe, return_to_origin, sustained_domain

## Global Statistics

| Metric | Mean | Median | Std | Min | Max | N |
|--------|------|--------|-----|-----|-----|---|
| E_recon | 6155.12 | 6451.11 | 1367.15 | 2900.30 | 9686.28 | 210 |
| E_predict | 6087.53 | 6478.08 | 1470.45 | 2993.38 | 8888.95 | 210 |
| E_blend | 6121.32 | 6298.26 | 1147.15 | 3107.50 | 8081.29 | 210 |
| Temperature | 1.16 | 1.18 | 0.20 | 0.62 | 1.49 | 210 |
| Cosine Distance | 0.28 | 0.26 | 0.08 | 0.00 | 0.50 | 209 |
| Deviation Match | 0.82 | 0.82 | 0.08 | 0.22 | 1.00 | 210 |

## IQR Stability

| Checkpoint | Median | IQR | IQR/Median |
|------------|--------|-----|------------|
| Turn 25 | 3438.2 | 1698.8 | 0.494 |
| Turn 50 | 4909.9 | 3020.6 | 0.615 |
| Turn 100 | 5898.8 | 2240.4 | 0.380 |
| Turn 200 | 6452.4 | 1537.7 | 0.238 |

## Signal Crossings

- **Total crossings**: 105 / 209 transitions (50.2%)

| Scenario | Crossings |
|----------|-----------|
| complexity_ladder | 11 |
| general | 44 |
| gradual_drift | 8 |
| rapid_switching | 11 |
| repetition_probe | 4 |
| return_to_origin | 14 |
| sustained_domain | 13 |

## Per-Domain Summary

| Domain | N | E_recon | E_predict | Cos Dist | Dev Match | Temp |
|--------|---|---------|-----------|----------|-----------|------|
| agriculture | 5 | 7293.00 | 7586.69 | 0.260 | 0.835 | 1.335 |
| art | 5 | 6668.71 | 6631.08 | 0.238 | 0.823 | 1.143 |
| astronomy | 5 | 6239.96 | 7280.09 | 0.260 | 0.837 | 1.148 |
| biology | 12 | 5823.15 | 5962.28 | 0.273 | 0.801 | 1.132 |
| chemistry | 8 | 6876.64 | 7035.78 | 0.244 | 0.820 | 1.284 |
| cooking | 8 | 5966.46 | 5486.02 | 0.350 | 0.755 | 1.103 |
| cs | 54 | 5840.32 | 5609.09 | 0.272 | 0.822 | 1.145 |
| economics | 5 | 6033.89 | 7033.16 | 0.254 | 0.832 | 1.094 |
| edge_case | 3 | 6321.10 | 7461.42 | 0.470 | 0.694 | 1.160 |
| film | 5 | 6540.43 | 6694.74 | 0.259 | 0.842 | 1.123 |
| history | 9 | 6194.77 | 6376.56 | 0.267 | 0.851 | 1.180 |
| law | 5 | 7309.01 | 6611.64 | 0.249 | 0.819 | 1.229 |
| linguistics | 5 | 6808.12 | 7140.71 | 0.231 | 0.816 | 1.248 |
| literature | 5 | 6709.79 | 6652.32 | 0.270 | 0.804 | 1.133 |
| math | 8 | 5823.64 | 5249.98 | 0.244 | 0.854 | 0.966 |
| medicine | 5 | 6653.84 | 6819.98 | 0.273 | 0.822 | 1.180 |
| music | 5 | 6855.02 | 6336.79 | 0.239 | 0.807 | 1.132 |
| philosophy | 35 | 5568.43 | 5476.68 | 0.272 | 0.851 | 1.159 |
| physics | 12 | 6312.41 | 6154.12 | 0.310 | 0.851 | 1.230 |
| psychology | 6 | 7127.05 | 7096.48 | 0.315 | 0.828 | 1.263 |
| sports | 5 | 6860.88 | 5989.29 | 0.321 | 0.768 | 1.078 |

## Auto-Detected Observations

- E_recon rises on 104/209 transitions (50%) — NOT monotonically declining
- Deviation match stable (drift=0.007)
