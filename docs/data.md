# Data and method

## Public sources

| Role | Dataset | Content |
| --- | --- | --- |
| Tumour profiles | TCGA via the GDC portal (open access) | 450K arrays, 8 classes × 48 tumours, one tumour per patient |
| Blood background | Hannum et al. 2013, GEO GSE40279 | Whole blood from 164 adults, 450K |
| Real plasma | Moss et al. 2018, GEO GSE122126 | Cell-free DNA (EPIC): 12 healthy, 4 lung, 4 colorectal, 3 breast, 4 unknown primary |

Tissue-of-origin classes in the prototype: lung, colorectal, breast, pancreas, liver,
oesophagus-stomach, head and neck, prostate. The product targets 50 cancer types.

## Split

Healthy individuals and tumours are split into training (70%) and test (30%) sets **before** any
simulation and before the panel is chosen. No person or tumour from the test set is seen during
training. Real plasma from Moss et al. is used only for final validation.

## Targeted panel

1,681 CpG sites, kept because they are almost never methylated in blood (mean below 0.10, 95th
percentile below 0.20) and methylated in at least one tumour type. Background noise in plasma
is low at these sites, so a few tumour fragments are enough to make them stand out.

## Simulating a plasma sample

1. Draw a real healthy individual: this is the background, the DNA released by blood cells.
2. For a cancer case, mix in a real tumour of the chosen class at a tumour fraction between
   0.01% and 10%.
3. Simulate sequencing: read depth varies per CpG (log-normal around 500 reads) and methylated
   reads follow a binomial draw.

**Known limitation**: the simulation works CpG by CpG, without read-level methylation patterns.
Commercial tests use those patterns and are more sensitive at very low tumour fractions.

## Reproduce

```bash
make data reference demo-samples
```
