# BugSeverityPrediction

[![DOI](https://zenodo.org/badge/680951811.svg)](https://zenodo.org/badge/latestdoi/680951811)

### Paper
You can find the paper here: https://arxiv.org/abs/2309.03044

## Artifact Description
This artifact contains all data (including data gathering step), code, and scripts required to run the paper's experiment to reproduce the results. The structure of folders and files are:

### `experiments` folder 
This folder contains all scripts and code required (specific to this apper) to re-run the training and testing our models (including classic models, CodeBERT, ConcatInline, and ConcatCLS). The structure of this folder is:

```
+-- data (contains paper full dataset and preprocessing step script)
|   +-- preprocess.sh (splitting dataset and scaling values)
+-- dataset (contains small subset of dataset after preprocessing for getting started section)
+-- models
|   +-- code_metrics (contains code for training and testing our classic models)
|      +-- train_test.sh (training and testing the models)
|   +-- code_representation
|      +-- codebert
|          +-- CodeBertModel.py (code for CodeBERT model)
|          +-- ConcatInline.py (code ConcatInline model)
|          +-- ConcatCLS.py (code ConcatCLS model)
|          +-- train.sh (script for training the models)
|          +-- inference.sh (script for testing the models)
|   +-- evaluation
|      +-- evaluation.py (evaluation metrics)
+-- utils (constant file)
```

#### data
The `data` folder contains bugs from Defects4tJ and Bugs.jar datasets. This folder contains a preprocessing script that unify bug severity values, scale the source code metrics and create `train`, `val`, and `test` splits.

Running this script using ```bash preprocessing.sh``` command generates 6 files containing `train`, `val`, `tests` splits in `jsonl` (compatible with CodeBERT experiments) and `csv` (compatible with source code metrics experiments) formats. 

#### dataset
Files available in the `dataset` folder represent data for getting started section (small subset of data). For reproducing paper result the generated files in `data` folder should be copied to the `dataset` folder that is used by the model training scripts. 

#### models
This folder contains all code and scripts for all of the experiments including classic models, CodeBERT models, ConcatInline, and ConcatCLS.


### `data_gathering` folder (out of paper scope):
This folder contains all required code to gather the data including issue scraping, method extraction, and metric extraction. While this step is out of scope of this paper, however, the required step to reproduce the data is available in this instruction.
While there are many directories/files in this folder, the following tree shows the structure of 3 files that need to be run.
```
+-- issue_scraper
|   +-- main.py
+-- MetricsExtractor
|   +-- method_extractor
|      +-- MethodExtractorMain.java
|   +-- metric_extractor
|      +-- MetricCalculatorMain.java
```

##  Environment Setup:
For Getting Started:
- Operating System: The provided artifact is tested on Linux (20.04.6 LTS) and macOS (Ventura 13.5).
- GPU: It is better to have GPU for running experiments on GPU otherwise it may take long time.
- CPU/RAM: There is no strict minimum on these.
- Python: Python 3 is required.

## Getting Started:
This section is only set up the artifact and validate its general functionality based on a small example data (complete dataset for the classic models, but the first 50 rows for CodeBERT models).

1. Clone the repository
   - `git@github.com:EhsanMashhadi/ISSRE2023-BugSeverityPrediction.git`

2. Install dependencies (using `requirements.txt` file) or manually :
  - `pip install pandas==1.4.2`
  - `pip install jira`
  - `pip install beautifulsoup4`
  - `pip install lxml`
  - `pip install transformers==4.18.0`
  - `pip install torch==1.11.0` This should be enough for running on CPU, but install the next for running on GPU
  - `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html`
  - `pip install scikit-learn==1.1.1`
  - `pip install xgboost==1.6.1`
  - `pip install seaborn==0.11.2`
3. Adding the project root folder to the `PYTHONPATH`
  - `export PYTHONPATH=$PYTHONPATH:*/rootpath/you/clone/the/project*/experiments`
  - e.g., `export PYTHONPATH=$PYTHONPATH:/Users/ehsan/workspace/ISSRE2023-BugSeverityPrediction/experiments`
4. RQ1: 
     - `cd ISSRE2023-BugSeverityPrediction/experiments/models/code_metrics`
     - `bash train_test.sh`
     - Results are generated in the `log` folder
5. RQ2:
   - `cd ISSRE2023-BugSeverityPrediction/experiments/models/code_representation/codebert`
   - Set `CodeBERT` as the `model_arch` parameter's value in `train.sh` and `inference.sh` files.
   - `bash train.sh` for training the model
   - `bash inference.sh` for evaluating the model with the `test` split
   - Results are generated in the `log` folder
6. RQ3:
   - `cd ISSRE2023-BugSeverityPrediction/experiments/models/code_representation/codebert`
   - Set `ConcatInline` or `ConcatCLS` as the `model_arch` parameter's value in `train.sh` and `inference.sh` files.
   - `bash train.sh` for training the model
   - `bash inference.sh` for evaluating the model with the `test` split
   - Results are generated in the `log` folder
   
## Reproducibility Instructions:
1. Clone the repository
   - `git@github.com:EhsanMashhadi/ISSRE2023-BugSeverityPrediction.git` 
2. Install dependencies (You may need to change the torch version for running on your GPU/CPU)

- **Experiments**:
  -  It is better to install these dependencies on a virtual env (you can also use requirements.txt)
  - `pip install pandas==1.4.2`
  - `pip install jira`
  - `pip install beautifulsoup4`
  - `pip install lxml`
  - `pip install transformers==4.18.0`
  -  `pip install torch==1.11.0` This should be enough for running on CPU, but install the next for running on GPU
  - `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html`
  - `pip install scikit-learn==1.1.1`
  - `pip install xgboost==1.6.1`
  - `pip install seaborn==0.11.2`
  
3. Adding the project root folder to the `PYTHONPATH`
  - `export PYTHONPATH=$PYTHONPATH:*/rootpath/you/clone/the/project*/experiments`
  - e.g., `export PYTHONPATH=$PYTHONPATH:/Users/ehsan/workspace/ISSRE2023-BugSeverityPrediction/experiments`
4. Running data preprocessing
   - `cd ISSRE2023-BugSeverityPrediction/experiments/data`
   - `bash preprocessing.sh`
   - Copy generated `jsonl` and `csv` files into the dataset folder

### Running Source Code Metrics Models Experiments (RQ1)
1. `cd ISSRE2023-BugSeverityPrediction/experiments/models/code_metrics`
2. `bash train_test.sh`
3. Results are generated in the `log` folder

### Running CodeBERT Model Experiments (RQ2)
1. `cd ISSRE2023-BugSeverityPrediction/experiments/models/code_representation/codebert`
2. Set `CodeBERT` as the `model_arch` parameter's value in `train.sh` file
3. `bash train.sh` for training the model
4. `bash inference.sh` for evaluating the model with the `test` split
5. Results are generated in the `log` folder

### Running Source Code Metrics Integration with CodeBERT Model Experiments (RQ3)
1. `cd ISSRE2023-BugSeverityPrediction/experiments/models/code_representation/codebert`
2. Set `ConcatInline` or `ConcatCLS` as the `model_arch` parameter's value in `train.sh` file
3. `bash train.sh` for training the model
4. `bash inference.sh` for evaluating the model with the `test` split
5. Results are generated in the `log` folder

### How to run with different config/hyperparameters?
   - You can change/add different hyperparameters/configs in `train.sh` and `inference.sh` files.

### Have trouble running on GPU?
1. Check the `CUDA` and `PyTorch` compatibility
2. Assign the correct values for `CUDA_VISIBLE_DEVICES`, `gpu_rank`, and `world_size` based on your GPU numbers in all scripts.
3. Run on CPU by removing the `gpu_rank`, and `world_size` options in all scripts.
4. Refer to the [CodeBERT Repo](https://github.com/microsoft/CodeBERT) for finding common issue.


### How to re-run the data gathering step  (out of paper scope)?

The tools below should be installed and configured correctly, otherwise, this step won't work. It may take a long time to do this step and can be skipped (recommended).

  - Java: Java 18 is required (**only for running data gathering step**).
  - Git: (brew, apt, ... based on your OS)
  - SVN: (brew, apt, ... based on your OS)
  - [Defects4J](https://github.com/rjust/defects4j) (Follow all the steps in the provided installation guide).
  - [Bugs.jar](https://github.com/bugs-dot-jar/bugs-dot-jar) (You must install this in the `data_gathering` directory).

1. `cd ISSRE2023-BugSeverityPrediction/data_gathering/issue_scraper`
2. `python main.py`

For below steps it can easier to use `gradlew`or simply open by IntelliJ IDEA to run java files

3. `cd ISSRE2023-BugSeverityPrediction/data_gathering/MetricsExtractor/src/main/java/software/ehsan/severityprediction/method_extractor`

4. `run MethodExtractorMain.java`

5. `cd ISSRE2023-BugSeverityPrediction/data_gathering/MetricsExtractor/src/main/java/software/ehsan/severityprediction/metric_extractor`

6. `run MetricCalculatorMain.java`
