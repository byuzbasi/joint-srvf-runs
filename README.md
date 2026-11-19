# Joint elastic analysis of ventricular contours and thickness profiles

Run code for the synthetic study and the CAMUS application in *Joint Elastic Analysis of Ventricular Contours and Image-Derived Thickness Profiles*.

## Environment and data

Use Python 3.13 on macOS or Linux. The analysis runs used Python 3.13.9 and the package versions in `requirements.txt`. LightGBM requires a working OpenMP runtime on platforms where its wheel does not bundle one.

Download the NIfTI release from the [CAMUS public collection](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8) and place the extracted `database_nifti` directory at the repository root. It should contain 500 patient directories with the end-diastolic two-chamber images, masks and metadata in the original structure. See the [provider's terms](https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/item/66e2904d961576b1bad4ed75/download) and cite [Leclerc et al. (2019)](https://doi.org/10.1109/TMI.2019.2900516). The runner checks the order, EF values and 350/150 split against the frozen digests in `camus_split_signature.json`. It generates synthetic inputs from the fixed design and seeds.

## Setup and small execution checks

After cloning, run these commands from the clone's parent directory.

```sh
cd joint-srvf-runs
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python code/run.py pilot synthetic synthetic-pilot
python code/run.py pilot camus camus-pilot
python code/run.py verify synthetic synthetic-pilot
python code/run.py verify camus camus-pilot
```

The pilots use a reduced grid. The CAMUS pilot checks the complete input cohort and fits only 12 subjects from the training split.

## Reproduce the study

```sh
cd joint-srvf-runs
. .venv/bin/activate
python code/run.py run synthetic synthetic-study
python code/run.py run camus camus-study --workers 4
```

The synthetic run has 10 replicates of 50 curves and 700 lambda-by-fold evaluations. CAMUS uses the fixed 350/150 split with 450 lambda-by-fold evaluations, including the fold-local representations used for tuning. On the original four-worker setup, CAMUS took about 2.1 hours. The serial synthetic run took about 3.8 hours. Runtime depends on the machine.

Run these commands in a second terminal to inspect progress.

```sh
cd joint-srvf-runs
. .venv/bin/activate
python code/run.py status synthetic synthetic-study
python code/run.py status camus camus-study
```

To resume an interrupted run, keep its run ID and worker count and use the matching command below. Resume uses verified lambda-by-fold checkpoints. Work between checkpoints can be repeated.

```sh
cd joint-srvf-runs
. .venv/bin/activate
python code/run.py resume synthetic synthetic-study
python code/run.py resume camus camus-study --workers 4
```

Verify the output and read the diagnostic record after each run.

```sh
cd joint-srvf-runs
. .venv/bin/activate
python code/run.py verify synthetic synthetic-study
python code/run.py verify camus camus-study
```

Each run writes to `runs/<run-id>`. Its manifest identifies the attempt and inventories the figures, tables, prediction files and diagnostics under that attempt. The synthetic alignment and tangent FPCA tables support manuscript Tables 2 and 3. CAMUS metadata, alignment, tangent modes, predictions and subgroup summaries support Tables 4 through 8. The `figures` directory contains the analysis figures that underlie the manuscript panels. `diagnostics.json` records sphere-mean tolerances and model convergence warnings. Treat CAMUS-generated patient-level outputs according to the provider's reuse terms.
