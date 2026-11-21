# Joint elastic analysis of ventricular contours and thickness profiles

Run code for *Joint Elastic Analysis of Ventricular Contours and Image-Derived Thickness Profiles*. This revision corrects the synthetic periodic grid and inverse SRVF scale restoration. Tangent FPCA reports eigenvalues with divisor $n$. The completed synthetic run supplies the results in the manuscript.

## Environment and data

Use Python 3.13 on macOS or Linux. The analysis runs used Python 3.13.9 and the package versions in `requirements.txt`. LightGBM requires a working OpenMP runtime on platforms where its wheel does not bundle one.

Download the NIfTI release from the [CAMUS public collection](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8) and place the extracted `database_nifti` directory at the repository root. It should contain 500 patient directories with the end-diastolic two-chamber images, masks and metadata in the original structure. See the [provider's terms](https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/item/66e2904d961576b1bad4ed75/download) and cite [Leclerc et al. (2019)](https://doi.org/10.1109/TMI.2019.2900516). The runner checks the order, EF values and 350/150 split against the frozen digests in `camus_split_signature.json`. It generates synthetic inputs from the fixed design and seeds.

## Setup and small execution checks

Clone the repository using the directory name shown below, or extract the
study archive into the same directory.

```sh
git clone https://github.com/byuzbasi/joint-srvf-runs.git srvf_joint_v23
```

Run the following commands from its parent directory.

```sh
cd srvf_joint_v23
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python tests/test_numerical_corrections.py
python code/run.py pilot synthetic synthetic-pilot
python code/run.py pilot camus camus-pilot
python code/run.py verify synthetic synthetic-pilot
python code/run.py verify camus camus-pilot
```

The pilots use a reduced grid. The CAMUS pilot checks the complete input cohort and fits only 12 subjects from the training split.

## Reproduce the study

```sh
cd srvf_joint_v23
. .venv/bin/activate
python code/run.py run synthetic synthetic-v19
python code/run.py run camus camus-study --workers 4
```

The synthetic run has 10 replicates of 50 curves and 700 lambda-by-fold evaluations. CAMUS uses the fixed 350/150 split with 450 lambda-by-fold evaluations, including the fold-local representations used for tuning. On the original four-worker setup, CAMUS took about 2.1 hours. The serial synthetic run took 3 hours 36 minutes. Runtime depends on the machine.

Run these commands in a second terminal to inspect progress.

```sh
cd srvf_joint_v23
. .venv/bin/activate
python code/run.py status synthetic synthetic-v19
python code/run.py status camus camus-study
```

To resume an interrupted run, keep its run ID and worker count and use the matching command below. Resume uses verified lambda-by-fold checkpoints. Work between checkpoints can be repeated.

```sh
cd srvf_joint_v23
. .venv/bin/activate
python code/run.py resume synthetic synthetic-v19
python code/run.py resume camus camus-study --workers 4
```

Verify the output and read the diagnostic record after each run.

```sh
cd srvf_joint_v23
. .venv/bin/activate
python code/run.py verify synthetic synthetic-v19
python code/run.py verify camus camus-study
```

Each run writes to `runs/<run-id>`. The run manifest identifies the
attempt and inventories figures, tables, diagnostics, and CV checkpoints.
A revised synthetic run needs a new run ID because the earlier generator
has a different signature.

| Manuscript table | Run | Output CSV |
|---|---|---|
| 2. Synthetic alignment | Synthetic | `synthetic_alignment_repeated_summary.csv` |
| 3. Synthetic tangent FPCA | Synthetic | `synthetic_srvf_tangent_fpca_error_results.csv` |
| 4. CAMUS cohort characteristics | CAMUS | `clinical_cohort_summary.csv` |
| 5. CAMUS alignment | CAMUS | `alignment_repeated_results.csv` |
| 6. CAMUS training tangent-mode variance | CAMUS | `srvf_tangent_fpca_variance_explained_first10.csv` |
| 7. CAMUS exploratory cluster summaries | CAMUS | `normal_EF_cluster_comparison_simplified.csv` |

`synthetic_fpca_comparison_results.csv` gives the component RMS
comparisons. The earlier CAMUS analysis provides the reported angular
alignment and cluster results. The first ten absolute CAMUS eigenvalues
in the manuscript are the historical archive values multiplied by
$349/350$. The runner in this package already uses divisor $n$. The archived CAMUS runner also computes exploratory EF
regression and classification. Treat generated patient-level outputs
according to the provider's reuse terms.

## CAMUS subgroup described alongside Table 7

From the parent directory of the extracted code or checkout, use the completed
CAMUS run and the original NIfTI release to reproduce the descriptive counts.

```sh
cd srvf_joint_v23
python3 analysis/describe_clusters.py runs/camus-study database_nifti
```

The first argument is the completed run directory containing `manifest.json`.
The script reads the split and test cluster assignments from the attempt named
in that manifest and verifies their file hashes. The second argument is the
CAMUS directory containing the original two-chamber metadata files
(`patient*/Info_2CH.cfg`). Substitute the paths to these directories if your
run has another name or your data are stored elsewhere.

The command uses only the Python standard library and prints aggregate JSON.
For the 52 test patients with EF of at least 50%, it reports cluster sizes,
mean EF, median age, sex and image-quality counts. It also counts distinct EF
values, lists the values observed in both clusters and counts patients with
any one of those shared values. Each patient has one cluster assignment.

## Manuscript figures

Figure 2 uses the first completed synthetic sample, which selected
$\lambda=0.005$. Figures 3 and 4 use the same sample and share a display
map for the gray observations, mean, and joint mode curves. The raw
comparison plots from the study runner use subject-specific restoration
for the gray observations. Use the accompanying figure materials
(`srvf_joint_fpca_figures.zip`) and their README to rebuild the manuscript
panels from the saved vectors. The figure materials also include the script
used to regenerate those vectors from the first sample with existing CV
checkpoints.
