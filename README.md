# func_model
This package contains workflows for modeling and extracting data from fMRI files. It contains two main divisions:
1. FSL-based pipelines which support classification of EPI signal, including:
    1. Modeling data
    1. Beta-coefficient extraction
    1. Converting classifier output into MNI coordinates
    1. Group-level models
1. AFNI-based pipelines which support univariate group-level analyses, including:
    1. Extra preprocessing
    1. Modeling data
    1. T-testing
    1. Linear mixed effects modeling

FSL-based sub-package/workflow navigation:
- [fsl_model](#fsl_model) : Conduct FSL-style first- and second-level regressions
- [fsl_extract](#fsl_extract) : Extract emotion betas from FSL first-level as a matrix
- [fsl_map](#fsl_map) : Make binary masks from classifier output
- [fsl_group](#fsl_group) : Generate required input for group-level analyses

AFNI-based sub-package/workflow navigation:
- [afni_model](#afni_model) : Conduct AFNI-style deconvultions
- [afni_etac](#afni_etac) : Conduct student or paired T-testing using the ETAC approach
- [afni_lmer](#afni_lmer) : Conduct a linear mixed effects analysis

These various workflows are written for either the DCC or labarserv2. Specifically, all AFNI and the fsl_model workflows are written for the DCC, while the fsl_extract, fsl_map, and fsl_group are written for labarserv2.


## General Usage
- Install package into the appropriate project environment (on DCC or labarserv2) given desired workflow via `$python setup.py install`.
- Trigger general package help and usage via entrypoint `$func_model`:

```
(emorep)[nmm51-dcc: ~]$func_model

Version : 4.3.2

The package func_model consists of sub-packages that can be accessed
from their respective entrypoints:

    afni_model   : Conduct AFNI-style deconvolution
    afni_etac    : Conduct T-tests in AFNI via ETAC
    afni_lmer    : Conduct linear mixed effects in AFNI

    fsl_model    : Conduct FSL-style first- and second-level regressions
    fsl_extract  : Extract emotion betas from FSL first-level as a matrix
    fsl_map      : Make binary masks from classifier output
    fsl_group    : Generate required input for group-level analyses

Sub-packages written for Duke Compute Cluster (DCC):

    - afni_model
    - afni_etac
    - afni_lmer
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


## afni_model
This subpackage is written to be executed on the DCC. It serves to model the session EPI data via AFNI's `3dDeconvolve` and `3dREMLfit`, and requires that [preprocessed](https://github.com/labarlab-emorep/func_preprocess) exists and is available on Keoki using the EmoRep derivative structure. These models collapse across runs, modeling the entire session.

A number of different models are available:
- `task`: Model the task stimuli for each emotion, not including replay, to produce a coefficient reflective of 10 stimulus presentations
- `block`: Model the task block, after accounting for replay and orienting responses to produce a coefficient reflective of two blocks
- `mixed`: Use a mixed model to combine `task` and `block`, used to test if any variance is unexplained by the task coefficient and can load on the block coefficient. For instance, the `task` coefficient may be driven largely by the stimulus type (watching a video) and including the `block` may capture variance associated with an emergent emotion (anger).


### Setup
- Generate an RSA key on the DCC for labarserv2 and set the global variable `RSA_LS2` to hold the path for the key
- Set the global variable `SING_AFNI` to hold the path to an AFNI singularity image.
- Verify that `c3d` is executable in the shell


### Usage
A CLI supplies available options, and their defaults, which allow the user to specify the subject, session and model name. Trigger the CLI via `$afni_model`:

```
(emorep)[nmm51-dcc: freesurfer]$afni_model
usage: afni_model [-h] [--model-name {mixed,task,block,rest}] [--sess {ses-day2,ses-day3} [{ses-day2,ses-day3} ...]] -s SUBJ [SUBJ ...]

Conduct AFNI-based models of EPI run files.

Written for the remote Duke Compute Cluster (DCC) environment.

Utilizing output of fMRIPrep, construct needed files for deconvolution. Write
the 3dDeconvolve script, and use it to generate the matrices and 3dREMLfit
script. Execute 3dREMLfit, and save output files to group location.
A workflow is submitted for each session found in subject's fmriprep
directory.

Model names:
    - task = Model stimulus for each emotion
    - block = Model block for each emotion
    - mixed = Model stimulus + block for each emotion
    - rest = Deprecated. Conduct a resting-state analysis referencing
        example 11 of afni_proc.py.

Requires
--------
- Global variable 'RSA_LS2' which has path to RSA key for labarserv2
- Global variable 'SING_AFNI' which has path to AFNI singularity image
- c3d executable from PATH

Examples
--------
afni_model -s sub-ER0009
afni_model \
    -s sub-ER0009 sub-ER0016 \
    --sess ses-day2 \
    --model-name mixed

optional arguments:
  -h, --help            show this help message and exit
  --model-name {mixed,task,block,rest}
                        AFNI model name/type, for triggering different workflows
                        (default : task)
  --sess {ses-day2,ses-day3} [{ses-day2,ses-day3} ...]
                        List of session BIDS IDs
                        (default : ['ses-day2', 'ses-day3'])

Required Arguments:
  -s SUBJ [SUBJ ...], --subj SUBJ [SUBJ ...]
                        List of subject BIDS IDs

```


### Functionality
`afni_model` spawns a parent sbatch job for each subject x session specified, each of which runs the following workflow:

1. Download fMRIPrep output and rawdata events files from Keoki
1. Conduct extra preprocessing starting with fMRIPrep preorcessed files:
    1. Make a func-anat intersection mask
    1. Make eroded WM, CSF masks
    1. Make a minimum-signal mask
    1. Spatially smooth the preprocessed EPI
    1. Scale the smoothed EPI
1. Generate AFNI-style motion and censor files from fMRIPrep timeseries.tsv files:
1. Make AFNI-style timing files from rawdata events
1. Model data
    1. Build a `3dDeconvolve` command with `-x1D_stop` option
    1. Execute `3dDeconvolve` command to generate `3dREMLfit` command
    1. Build a noise estimation file from WM signal
    1. Execute `3dREMLfit`
1. Upload data to Keoki and clean up

Modeled output is organized in the derivatives sub-directory 'model_afni':

```
model_afni/sub-ER0009
└── ses-day2
    └── func
        ├── motion_files
        │   ├── info_censored_volumes.json
        │   ├── sub-ER0009_ses-day2_task-movies_desc-censor_timeseries.1D
        │   ├── sub-ER0009_ses-day2_task-movies_desc-deriv_timeseries.1D
        │   └── sub-ER0009_ses-day2_task-movies_desc-mean_timeseries.1D
        ├── sub-ER0009_ses-day2_task-movies_desc-decon_model-task_errts_REML+tlrc.BRIK
        ├── sub-ER0009_ses-day2_task-movies_desc-decon_model-task_errts_REML+tlrc.HEAD
        ├── sub-ER0009_ses-day2_task-movies_desc-decon_model-task_reml.sh
        ├── sub-ER0009_ses-day2_task-movies_desc-decon_model-task.sh
        ├── sub-ER0009_ses-day2_task-movies_desc-decon_model-task_stats.REML_cmd
        ├── sub-ER0009_ses-day2_task-movies_desc-decon_model-task_stats_REML+tlrc.BRIK
        ├── sub-ER0009_ses-day2_task-movies_desc-decon_model-task_stats_REML+tlrc.HEAD
        ├── sub-ER0009_ses-day2_task-movies_desc-intersect_mask.nii.gz
        ├── timing_files
        │   ├── sub-ER0009_ses-day2_task-movies_desc-comJud_events.1D
        │   ├── sub-ER0009_ses-day2_task-movies_desc-comRep_events.1D
        │   ├── sub-ER0009_ses-day2_task-movies_desc-comWas_events.1D
        │   ├── sub-ER0009_ses-day2_task-movies_desc-movAmu_events.1D
        ..  ..
        │   ├── sub-ER0009_ses-day2_task-movies_desc-movSur_events.1D
        │   ├── sub-ER0009_ses-day2_task-movies_desc-selEmo_events.1D
        │   └── sub-ER0009_ses-day2_task-movies_desc-selInt_events.1D
        ├── X.sub-ER0009_ses-day2_task-movies_desc-decon_model-task.jpg
        ├── X.sub-ER0009_ses-day2_task-movies_desc-decon_model-task.jpg.xmat.1D
        └── X.sub-ER0009_ses-day2_task-movies_desc-decon_model-task.xmat.1D

```
The modeled output found in the 'func' directory is the '\*_stats_REML+tlrc.[BRIK|HEAD]' files, with the accompanying residual '\*_errts_REML+tlrc.[BRIK|HEAD]' files. Also available are the 3dDeconvolve and 3dREMLfit commands used as shell scripts, and the design X-files. Separate models (task, block, mixed) are organized according to the 'model' key. Finally, the intersection mask used in modeling the data is provided.

The directory 'func/motion_files' contains the motion input 1D files as well as a JSON detailing the number and proportion of volumes excluded. Likewise, the directory 'func/timing_files' contain AFNI-style timing files.


### Considerations
- The modeled sub-brick will be named according to the description field of the timing file. For instance, the information for the amusement movie is held in the file '\*_desc-movAmu_events.1D' and the resulting sub-brick containig the beta-coefficient will be named 'movAmu'. These description values are necessarily short given the number of characters AFNI allows to name a sub-brick.


## afni_etac
This sub-package is written to be executed on the DCC and functions to conduct A vs B (paired) or A vs 0 (student) T-tests at the group level. Additionally, the [ETAC](https://afni.nimh.nih.gov/pub/dist/edu/data/CD.expanded/afni_handouts/afni07_ETAC.pdf) options are used, allowing for a denser sampling of significant space by using multiple cluster and threshold values.


### Setup
- Generate an RSA key on the DCC for labarserv2 and set the global variable `RSA_LS2` to hold the path for the key
- Set the global variable `SING_AFNI` to hold the path to an AFNI singularity image.


### Usage
A CLI is accessible at `$afni_etac` which provides a help, options, and examples. This workflow has three steps which are independently accessible through the CLI:
1. Download necessary data
1. Extract sub-brick labels
1. Conduct T-testing

These steps should be executed in order, and the user is able to specify the task, model name, statistic, and/or coefficient relevant for the step:


```
(emorep)[nmm51-dcc: func_model]$afni_etac
usage: afni_etac [-h] [--block-coef]
                 [--emo-name {amusement,anger,anxiety,awe,calmness,craving,disgust,excitement,fear,horror,joy,neutral,romance,sadness,surprise}]
                 [--get-subbricks] [--model-name {mixed,task,block}] [--run-etac] [--run-setup] [--stat {student,paired}]
                 [--task {movies,scenarios}]

Conduct T-Testing using AFNI's ETAC methods.

Conduct testing with 3dttest++ using the -etac_opts option. This
controls the number and value of thresholds and blurs, nearest
neighbor and alpha values, among other parameters.

Model names correspond to afni_model output:
    - task = Model stimulus for each emotion
    - block = Model block for each emotion
    - mixed = Model stimulus + block for each emotion

Stat names:
    - student = Student's T-test, compare each task emotion against zero
    - paired = Paired T-test, compare each task emotion against washout

Requires
--------
- Global variable 'RSA_LS2' which has path to RSA key for labarserv2
- Global variable 'SING_AFNI' which has path to AFNI singularity image

Notes
-----
When using --model-name=mixed, the default behavior is to
extract the task/stimulus subbricks. The block subbrick is
available by including the option --block-coef.

Example
-------
1. Get necessary data
    afni_etac \
        --run-setup \
        --task movies \
        --model-name mixed

2. Identify sub-brick labels
    afni_etac \
        --get-subbricks \
        --stat paired \
        --task movies \
        --model-name mixed \
        --block-coef

3. Conduct T-testing
    afni_etac \
        --run-etac \
        --stat paired \
        --task movies \
        --model-name mixed \
        --block-coef

optional arguments:
  -h, --help            show this help message and exit
  --block-coef          Test block (instead of event) coefficients when model-name=mixed
  --emo-name {amusement,anger,anxiety,awe,calmness,craving,disgust,excitement,fear,horror,joy,neutral,romance,sadness,surprise}
                        Use emotion (instead of all) for workflow
  --get-subbricks       Identify sub-brick labels for emotions and washout
  --model-name {mixed,task,block}
                        AFNI deconv name
                        (default : task)
  --run-etac            Conduct t-testing via AFNI's ETAC
  --run-setup           Download model_afni data and make template mask
  --stat {student,paired}
                        T-test type
  --task {movies,scenarios}
                        Task name

```


### Functionality
First, downloading data from Keoki is available with the `--run-setup` option, for which the user will also specify the desired task and model name (see, [afni_model](#afni_model)). Running setup will build the directory structure on the DCC and then download the requested data from Keoki. Downloaded subject will be found at '/work/user/EmoRep/model_afni', and a directory for group analyses will be found at '/work/user/EmoRep/model_afni_group'.

Second, the deconvolved sub-brick labels are extracted with the `--get-subbricks` option. A unique file for each sub-brick of interest will be written to the subject's 'func' directory, and a washout file will also be written if `--stat paired` is used. When using `mixed` models, the task sub-brick named [mov|sce]Emo for a movie or scenario emotion, respectively, while the block sub-brick is named blk[M|S]Emo.

Third, the T-test workflow will run with the following steps:
1. Find all deconvolved files
1. Make/find the template mask
1. Identify all extracted sub-bricks
1. For each emotion and task:
    1. Build the `3dttest++` command, including the ETAC options:
        1. Blur sizes = 0, 2, 4
        1. Nearest neighbor = 2
        1. H power = 0
        1. P threshold = 0.01, 0.005, 0.002, 0.001
    1. Execute `3dttest++`
1. Upload output to Keoki and clean up DCC

T-test output can be found an the model_afni_group directory of experiments2/EmoRep/Exp2_Compute_Emotion/analyses that corresponds to the type of statistic, task, model, and emotion:

```
analyses/model_afni_group/stat-paired_task-movies_model-task_emo-anger/
..
├── stat-paired_task-movies_model-task_emo-anger_clustsim.etac.ETACmaskALL.global.2sid.05perc.nii.gz
├── stat-paired_task-movies_model-task_emo-anger_clustsim.etac.ETACmask.global.2sid.05perc.nii.gz
├── stat-paired_task-movies_model-task_emo-anger.sh
├── stat-paired_task-movies_model-task_emo-anger+tlrc.BRIK
└── stat-paired_task-movies_model-task_emo-anger+tlrc.HEAD

```
While many files are written (see [here](https://afni.nimh.nih.gov/afni/community/board/read.php?1,164803,164803#msg-164803) for explanations), those illustrated above are the most relevant:
- stat-\*_cluststim.etac.ETACmask.global.\*, which contains the final output binary masks
- stat-\*_cluststim.etac.ETACmaskALL.global.\*, which contains binary masks for each blur and P threshold
- stat-\*.sh, the script used to execute `3dttest++`
- stat-\*+tlrc.[BRIK|HEAD], which contains the original Z-stat matrix


### Considerations


## afni_lmer
### Setup
### Usage
### Functionality
### Considerations