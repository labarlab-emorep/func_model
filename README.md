# func_model
This package contains workflows for modeling and extracting data from fMRI files. It contains two main divisions:
1. resources for FSL-based pipelines (detailed below), and
1. resources for AFNI-based pipelines (largely deprecated, not detailed here).

Sub-package/workflow navigation:
- [fsl_model](#fsl_model) : Conduct FSL-style first- and second-level regressions
- [fsl_extract](#fsl_extract) : Extract emotion betas from FSL first-level as a matrix
- [fsl_map](#fsl_map) : Make binary masks from classifier output
- [fsl_group](#fsl_group) : Generate required input for group-level analyses


## General Usage
- Install package into project environment (on DCC or labarserv2) via `$python setup.py install`.
- Trigger general package help and usage via entrypoint `$func_model`:

```
(emorep)[nmm51-dcc: ~]$func_model

Version : 4.3.0

The package func_model consists of sub-packages that can be accessed
from their respective entrypoints:

    fsl_model    : Conduct FSL-style first- and second-level regressions
    fsl_extract  : Extract emotion betas from FSL first-level as a matrix
    fsl_map      : Make binary masks from classifier output
    fsl_group    : Generate required input for group-level analyses

Sub-packages written for Duke Compute Cluster (DCC):

    - fsl_model

Sub-packages written for labarserv2:

    - fsl_extract
    - fsl_map
    - fsl_group

```
Note that [fsl_model](#fsl_model) is written for execution on the Duke Compute Cluster (DCC) while the remaining three sub-packages/workflows are written for execution on labarserv2.


## General Requirements
- The [fsl_model](#fsl_model) workflow requires the global variable `RSA_LS2` to contain a path to an RSA key for labarserv2.
- The workflows [fsl_extract](#fsl_extract) and [fsl_map](#fsl_map) require the global variable `SQL_PASS` to hold the user password for the MySQL database `db_emorep`.

Example:

```bash
$echo "export SQL_PASS=foobar" >> ~/.bashrc && source ~/.bashrc
```


## fsl_model
This sub-package is written to be executed on the DCC and conducts first- and second-level modeling using an FSL-based pipeline. It requires that preprocessed data already exists and is available on Keoki using the EmoRep derivatives structure (see [func_preprocess](https://github.com/labarlab-emorep/func_preprocess)).

A number of different models are supported:
* `sep`: Model the emotion stimulus and replay events separately
* `tog`: Model the emotion stimulus and replay events together
* `rest`: Model effects-of-no-interest in rsfMRI to produce cleaned residuals
* `lss`: Similar to `tog`, but each trial (emotion stimulus + replay) are modeled separately

Additionally, second-level modeling is possible for `sep` and `tog`, once first-level modeling has been conducted.


### Setup
* Generate an RSA key on the DCC for labarserv2 and set the global variable `RSA_LS2` to hold the path for the key
* Ensure the FSL is configured and executable in the environment


### Usage
The CLI `$fsl_model` supplies a number of options (as well as their corresponding defaults if optional) that allow the user to specify the subject, session, type, and level of the FSL model. Trigger sub-package help and usage via `$fsl_model`:

```
(emorep)[nmm51-dcc: ~]$fsl_model
usage: fsl_model [-h] [--model-level {first,second}] [--model-name {sep,tog,rest,lss}] [--preproc-type {scaled,smoothed}]
                 [--proj-dir PROJ_DIR] [--ses-list {ses-day2,ses-day3} [{ses-day2,ses-day3} ...]] -s SUB_LIST [SUB_LIST ...]

CLI for initiating FSL regressions.

Written for the remote Duke Compute Cluster (DCC) environment.

Setup and run first- and second-level models in FSL for task
and resting-state EPI data. Output are written to participant
derivatives:
    <proj-dir>/derivatives/model_fsl/<subj>/<sess>/func/run-*_<level>_<name>

Model names:
    - sep = emotion stimulus (scenarios, movies) and replay are
        modeled separately
    - tog = emotion stimulus and replay modeled together
    - rest = model resting-state data to remove nuissance regressors,
        first-level only
    - lss = similar to tog, but with each trial separate,
        first-level only

Level names:
    - first = first-level GLM
    - second = second-level GLM, for model-name=sep|tog only

Notes
-----
- Requires environmental variable 'RSA_LS2' to contain
    location of RSA key for labarserv2

Examples
--------
fsl_model -s sub-ER0009
fsl_model -s sub-ER0009 sub-ER0016 \
    --model-name tog \
    --ses-list ses-day2 \
    --preproc-type smoothed

optional arguments:
  -h, --help            show this help message and exit
  --model-level {first,second}
                        FSL model level, for triggering different workflows.
                        (default : first)
  --model-name {sep,tog,rest,lss}
                        FSL model name, for triggering different workflows
                        (default : sep)
  --preproc-type {scaled,smoothed}
                        Determine whether to use scaled or smoothed preprocessed EPIs
                        (default : scaled)
  --proj-dir PROJ_DIR   Path to BIDS-formatted project directory
                        (default : /hpc/group/labarlab/EmoRep/Exp2_Compute_Emotion/data_scanner_BIDS)
  --ses-list {ses-day2,ses-day3} [{ses-day2,ses-day3} ...]
                        List of subject IDs to submit for pre-processing
                        (default : ['ses-day2', 'ses-day3'])

Required Arguments:
  -s SUB_LIST [SUB_LIST ...], --sub-list SUB_LIST [SUB_LIST ...]
                        List of subject IDs to submit for pre-processing

```


### Functionality
`fsl_model` spawns a parent sbatch job for each subject x session specified, each of which runs the following workflow:

1. Download the required data from Keoki via labarserv2.
    1. First-level models: fMRIPrep and fsl_denoise derivatives
    1. Second-level models: first-level feat derivatives
2. (Second-level models) Bypass registration requirement
3. Generate required files for modeling:
    1. First-level models, task designs: Condition files for the specified model by mining BIDS rawdata func/*_events.tsv files.
    2. First-level models: Confound files by mining fMRIPrep *_desc-confounds_timeseries.tsv files.
    2. All-models: Design FSF files by populating pre-generated templates found at `func_model.reference_files.design_template_<model-level>_<model-name>_desc-*.fsf`. For first-level models, desc-full designs are for runs 1-3, 5-7 while desc-short designs are for runs 4, 8.
4. Schedule child jobs to parallelize executing each design file via FSL's `feat`.
5. Upload output data to Keoki via labarserv2
6. Clean up session directories on DCC.

Modeled output is organized in the derivatives sub-directory 'model_fsl':

```
derivatives/model_fsl/
└── sub-ER0009
    └── ses-day2
        └── func
            ├── condition_files
            │   ├── sub-ER0009_ses-day2_task-movies_run-01_desc-emoIntensity_events.txt
            │   └── many_other_events.txt
            ├── confounds_files
            │   ├── sub-ER0009_ses-day2_task-movies_run-01_desc-confounds_timeseries.txt
            ..  ..
            │   └── sub-ER0009_ses-day2_task-movies_run-08_desc-confounds_timeseries.txt
            ├── confounds_proportions
            │   ├── sub-ER0009_ses-day2_task-movies_run-01_desc-confounds_proportion.json
            ..  ..
            │   └── sub-ER0009_ses-day2_task-movies_run-08_desc-confounds_proportion.json
            ├── design_files
            │   ├── run-01_level-first_name-sep_design.fsf
            ..  ..
            │   └── run-08_level-first_name-sep_design.fsf
            ├── run-01_level-first_name-sep.feat
                └── various_fsl_directories
            ..
            ├── run-08_level-first_name-sep.feat
                └── various_fsl_directories
            └── level-second_name-sep.gfeat
                └── various_fsl_directories
```
Output is organized within the BIDS func directory, using a number of directories:
- **condition_files** contains task condition files generated from rawdata events files, where the description field becomes then name of the coefficient.
- **confounds_files** contain confound information derived from fMRIPrep *desc-confound_timeseries.tsv files
- **confound_proportions** contain a JSON for each run detailing the number and proportion of volumes that were censored
- **design_files** contain the generated design.fsf files
- **feat directories** are named according to the run, model name, and level


### Considerations
* The `lss` models spawn an enormous number of processes as parallelization of `feat` occurs for each trial, beware of DCC usage limits to avoid account restrictions!


## fsl_extract
This sub-package is written to be executed on labarserv2 and functions to extract first-level beta-coefficients from each voxel to populate the SQL table `db_emorep.tbl_betas_*`.


### Setup
* Set the global variable `SQL_PASS` to contain the user password for `db_emorep`.


### Usage
The CLI `$fsl_extract` allows the user to trigger beta-coefficient extraction of first-level models for specified subjects and fsl model names. Trigger sub-package help and usage via `$fsl_extract`:

```
(emorep)[nmm51-vm: ~]$fsl_extract
usage: fsl_extract [-h] [--model-name {lss,sep}] [--overwrite] [--proj-dir PROJ_DIR] [--sub-list SUB_LIST [SUB_LIST ...]]
                   [--sub-all]

Extract voxel beta weights from FSL FEAT files.

Written for the local labarserv2 environment.

Mine FSL GLM files for contrasts of interest and generate a
dataframe of voxel beta-coefficients. Dataframes may be masked by
identifying coordinates in a group-level mask. Extracted beta-values
are written sent to MySQL:
    db_emorep.tbl_betas_*

Model names (see fsl_model):
    - lss = conduct full GLMs for every single trial
    - sep = model stimulus and replay separately

Notes
-----
- Extraction of betas for model name 'tog' has been deprecated.
- Requires environmental variable 'SQL_PASS' to contain
    user password for db_emorep.

Examples
--------
fsl_extract --sub-all
fsl_extract --sub-list sub-ER0009 sub-ER0016 --overwrite
fsl_extract --sub-all --model-name lss

optional arguments:
  -h, --help            show this help message and exit
  --model-name {lss,sep}
                        FSL model name, for triggering different workflows.
                        (default : sep)
  --overwrite           Whether to overwrite existing records
  --proj-dir PROJ_DIR   Path to experiment-specific project directory
                        (default : /mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion)
  --sub-list SUB_LIST [SUB_LIST ...]
                        List of subject IDs to extract behavior beta-coefficients
  --sub-all             Extract beta-coefficients from all available subjects and
                        generate a master dataframe.

```


### Functionality
Triggering this sub-package will execute the following workflow:

1. Generate a mask and identify non-zero coordinates
1. Identify subjects with the desired [fsl_model](#fsl_model)  output
1. Iterate serially through subjects, spawn parallel processes for each session
    1. (Sessions for model `lss` are executed serially)
1. Check database `db_emorep.tbl_betas_*` for existing data, determine whether to overwrite from user options
1. Spawn parallel processes for each run
1. Align run cope IDs to task labels
1. Flatten cope NIfTIs into 1D arrays while maintaining coordinates
1. Apply masking to arrays
1. Concatenate transposed cope 1D arrays into dataframe
1. Concatenate dataframes across runs
1. Update database `db_emorep.tbl_betas_*` with beta-coefficients
1. Write dataframes to fsl_model subject's func directory (deprecated)


### Considerations
* Support for first-level model 'tog' has been deprecated
* This workflow is resource intensive over many hours due to the need to maintain coordinate information
* Utilizing a gray matter mask reduces the number of voxels from over 900K to 137K
* Runs with a proportion of censored volumes >20% are skipped, and beta-values >9999 or <-9999 are censored
* The FSL file design.con is used to identify successful run modeling, and the file strucutre of FSL's `feat` output is assumed. The design.con file is also read to identify the task label of each stats cope.


## fsl_map
This sub-package is written to be executed on labarserv2 and serves to generate a file in MNI coordinate space from an array of values. Specifically, the output of [fsl_extract](#fsl_extract)  is used as the input for classification, which in turn generates a binary table where 1 indicates the coordinate/voxel/feature contributed significantly to classification. This table is stored at `db_emorep.tbl_plsda_binary_*` and the selected values are used to reconstruct a binary cluster map in MNI space.

Maps can be made for different task names:
* `movies`: Generate a map from the classifier trained on the movie task
* `scenarios`: Same as `movies`, but for the scenario task
* `all`: Generate a map from the classifier trained on both the movies and scenarios tasks

Maps can also be made for different FSL models:
* `sep`: Generate a map from the classifier trained on data modeled with stimulus and replay loading on separate coefficients
* `tog`: Deprecated. Generate a map from the classifier trained on data modeled with stimulus and replay loading together on the same coefficient

Additionally, maps can be made from different task coefficients/contrasts:
* `stim`: Geneate a map from the classifier trained on only the stimulus (movie or task) portion of a trial
* `replay`: Same as `stim`, but from only the replay portion of the trial
* `tog`: Deprecated. Generate a map from from the classifier trained on the entire trial


### Setup
* Set the global variable `SQL_PASS` to contain the user password for the MySQL database `db_emorep`
* Verify that the global variable `SINGULARITY_TEMPLATEFLOW_HOME` exists and holds the path to a clone of the [templateflow repository](https://www.templateflow.org/)
* Verify that `tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz` exists within the templateflow repository
* Check for relevant classifier output in `db_emorep.tbl_plsda_binary_*`


### Usage
The CLI `$fsl_map` allows the user to trigger map building in MNI space for specific classifier output. Trigger sub-package help and usage via `$fsl_map`:

```
(emorep)[nmm51-vm: ~]$fsl_map
usage: fsl_map [-h] [--contrast-name {stim,replay,tog}] [--model-level {first}] [--model-name {sep,tog}] [--proj-dir PROJ_DIR] -t
               {movies,scenarios,all}

Generate NIfTI masks from classifier output.

Written for the local labarserv2 environment.

Convert data from db_emorep.tbl_plsda_binary_gm into
NIfTI files build in MNI template space. Then generate
conjunctive analysis maps.

Writes output to:
    proj_dir/analyses/classify_fMRI_plsda/voxel_importance_maps/name-*_task-*_maps

Examples
--------
fsl_map -t movies
fsl_map -t all

optional arguments:
  -h, --help            show this help message and exit
  --contrast-name {stim,replay,tog}
                        Desired contrast from which coefficients will be extracted,
                        substring of design.fsf EV Title.
                        (default : stim)
  --model-level {first}
                        FSL model level, for triggering different workflows
                        (default : first)
  --model-name {sep,tog}
                        FSL model name, for triggering different workflows
                        (default : sep)
  --proj-dir PROJ_DIR   Path to experiment-specific project directory
                        (default : /mnt/keoki/experiments2/EmoRep/Exp2_Compute_Emotion)

Required Arguments:
  -t {movies,scenarios,all}, --task-name {movies,scenarios,all}
                        Name of EmoRep stimulus type, corresponds to BIDS task field
                        (default : None)

```


### Functionality
Triggering this sub-package will execute the following workflow:

1. Extract header data from MNI template
1. Select requested data from `db_emorep.tbl_plsda_binary_*`
1. Generate 3D binary map for each emotion
1. Apply MNI metadata to 3D map
1. Generate conjunction maps
    1. High, low arousal maps
    1. High, low valence maps
    1. Cluster-thresholded maps and write survival table

Generated maps are written to the parent directory experiments2/EmoRep/Exp2_Compute_Emotion/analyses/classify_fMRI_plsda/voxel_importance_maps and organized by model name and task:

```
classify_fMRI_plsda/voxel_importance_maps/
├── name-sep_task-both_maps
├── name-sep_task-movies_maps
└── name-sep_task-scenarios_maps
```

Individual maps are found within their respective model-task directory. Unprocessed emotion maps are identifed by the 'emo-' field:

```
name-sep_task-movies_maps/
├── binary_model-sep_task-movies_con-stim_emo-amusement_map.nii.gz
├── binary_model-sep_task-movies_con-stim_emo-anger_map.nii.gz
├── binary_model-sep_task-movies_con-stim_emo-anxiety_map.nii.gz
..
└── binary_model-sep_task-movies_con-stim_emo-surprise_map.nii.gz
```

Similarly, conjuction maps are identified by the 'con-' field:

```
name-sep_task-movies_maps/
├── binary_model-sep_task-movies_con-stim_conj-aroHigh_map.nii.gz
├── binary_model-sep_task-movies_con-stim_conj-aroLow_map.nii.gz
├── binary_model-sep_task-movies_con-stim_conj-aroMed_map.nii.gz
├── binary_model-sep_task-movies_con-stim_conj-omni_map.nii.gz
├── binary_model-sep_task-movies_con-stim_conj-valNeg_map.nii.gz
├── binary_model-sep_task-movies_con-stim_conj-valNeu_map.nii.gz
└── binary_model-sep_task-movies_con-stim_conj-valPos_map.nii.gz
```

Finally, clustered maps have a file name starting with 'Clust_' and a corresponding table saved as a TXT file:
```
name-sep_task-movies_maps/
├── Clust_binary_model-sep_task-movies_con-stim_conj-aroHigh_map.nii.gz
├── Clust_binary_model-sep_task-movies_con-stim_conj-aroHigh_map.txt
..
├── Clust_binary_model-sep_task-movies_con-stim_emo-surprise_map.nii.gz
└── Clust_binary_model-sep_task-movies_con-stim_emo-surprise_map.txt
```

### Considerations
* Coordinates for 3D map construction are derived from [fsl_extract](#fsl_extract)
* The workflow also supports generating importance (and not just binary) masks, but this is not implemented at the CLI


## fsl_group
Deprecated.
This sub-package generates required files for third- and fourth-level analyses, where:

* first-level: collapse across trial generate coefficients
* second-level: collapse across runs
* third-level: collapse across tasks
* fourth-level: collapse across participants

Unfortunately, third- and particularly fourth-level are breaking the FSL GUI needed to generate the design and contrast files. The current plan is to utilize AFNI methods if group-level univariate analyses are needed.
