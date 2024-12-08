<div align="center">

# GenieRedux

This is the official repository of <b>"Learning Generative Interactive Environments by Trained Agent Exploration"</b>.

[![Website](docs/badges/badge_project_page.svg)](https://nsavov.github.io/GenieRedux/) 
[![Paper](docs/badges/badge_pdf.svg)](https://arxiv.org/pdf/2409.06445) 

Authors: [Naser Kazemi](https://naser-kazemi.github.io/)\*, [Nedko Savov](https://insait.ai/nedko-savov/)\*, [Danda Pani Paudel](https://insait.ai/dr-danda-paudel/), [Luc Van Gool](https://insait.ai/prof-luc-van-gool/)

![GenieRedux](docs/title.gif)

![GenieRedux](docs/models.png)
</div>
<b>GenieRedux</b> is a complete open-source Pytorch implementation of the <b>Genie</b> world model introduced by Google Research. Given a sequence of frames and actions from an environment, the model predicts the visual outcome of executing the actions, thus serving as environment simulators. The model has a Latent Action Model that predicts the actions in a self-supervised manner.

<b>GenieRedux-G</b> (Guided) is a version of GenieRedux, adapted for use with virtual environments and agents. In contrast to GenieRedux, this guided version takes its actions from an agent rather than predicting them from unnanotated data of human demonstrations (datasets which are costly to obtain and curate).

We train and evaluate the models on the CoinRun case study, as advised by the Genie paper. We provide an easy data generation utility and an efficient data handler for training. In addition, we provide evaluation scripts.

<b>Expect the release of our weights in the next days.</b>

## Installation
<b>Prerequisites:</b>
- Ensure you have Conda installed on your system. You can download and install Conda from the [official website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).


<b> Data Generation Environment Installation. </b>
This will install the `coinrun` environment which enables the use of the Coinrun environment:
```bash
bash data_generation/external/install_coinrun.sh
conda activate coinrun
```

<b>GenieRedux Environment  Installation:</b>
1. Clone the repository:

   ```shell
   git clone https://github.com/insait-institute/GenieRedux.git
   cd GenieRedux
   ```

2. Set up the Python environment:
   ```shell
    conda env create -f genie_redux_env.yaml
    conda activate genie_redux
   ``` 
   This script will create a conda environment named `genie_redux` and install all the required dependencies, inclduing `PyTorch-Cuda 12.1`, `Hydra`, and `Accelerate`.

Note: This implementation is tested on Linux-64 with Python 3.10 and Conda package manager.

## Quickstart



### Data Generation

Initial setup:
```
cd data_generation
conda activate coinrun
```

To generate a training dataset of 10k instances (episodes), with length at most 500 frames, and using a random agent, run:

```bash
python generate.py
```

This will generate a dataset in `data_generation/datasets/coinrun_v2.0.0`.

To generate a test dataset of 10k instances (episodes), with length at most 500 frames, and using a random agent, run:

```bash
python generate.py --config_name configs/data_gen_test_set.json
```

This will generate a dataset in `data_generation/datasets/coinrun_v2.0.0_test`.

<!-- 
To train an agent, follow Coinrun's instructions
To generate data with an agent change in `data_generation/configs/data_gen.json` set the following for `"connector_coinrun"` properties:
```json
"connector_coinrun":
    {
        "name": "coinrun",
        "version": "0.0.0",
        "agent_type": "ppo",
        "is_high_res": false,
        "is_high_difficulty": false,
        "should_paint_velocity": true,
        "image_size": [64,64],
        "generator_config": {
            "n_instances": 10000,
            "n_sessions": 1,
            "n_steps_max": 500,
            "n_workers": 46
        }
    }
}
```
It is expected that you will have your pretrained model weights at `data_generation/external/coinrun/coinrun/saved_models/sav_myrun_0`
Run generation with:
```bash
python generate.py
``` -->

### Training GenieRedux

Before we start, we set up the environment:
```bash
conda activate genie_redux
```

We trained GenieRedux and GenieRedux-G on 7 A100 GPUs, with batch size 84. We provide the settings as we used them in the following instructions, the parameters can be changed according to GPU and memory limitations.

#### Tokenizer
To train the tokenizer for on the generated dataset (for 150k iterations), run:
```bash
bash run.sh --config=tokenizer.yaml --num_processes=6 --train.batch_size=7 --train.grad_accum=2
```

#### GenieRedux (Dynamics+LAM)
Using the tokenizer that we just train, we can now train GenieRedux:
```bash
bash run.sh --config=genie_redux.yaml --num_processes=7 --train.batch_size=3 --train.grad_accum=4
```

#### GenieRedux-G (Dynamics Only)
Alternatively, to use the ground truth actions instead of LAM:
```bash
bash run.sh --config=genie_redux_guided.yaml --num_processes=7 --train.batch_size=4 --train.grad_accum=3
```

### Evaluation of GenieRedux

To get quantitative evaluation (ΔPSNR, FID, PSNR, SSIM):
```bash
bash run.sh --config=genie_redux_1.yaml --mode=eval --eval.action_to_take=-1 --eval.model_path=<path_to_model> --eval.inference_method=one_go
```

## Data Generation

Initial setup:
```bash
cd data_generation
conda activate coinrun
```

Data generation is ran in the following format:
```
python generate --config configs/<CONFIG_NAME>.json
```

### Data Generation With A Trained Agent

In Quickstart, we saw how to generate data with a random agent. Here we discuss using a PPO agent.

For training a PPO agent, follow the instructions in the [Coinrun repository](https://github.com/openai/coinrun).
It is expected that the pretrained model weights are at `data_generation/external/coinrun/coinrun/saved_models/sav_myrun_0`.

Run generation with:
```bash
python generate.py --config configs/data_gen_ppo.json
```

For a test set:

```bash
python generate.py --config /external/data_gen_ppo_test_set.json
```
### Data Generation Parameters

Additional customization is available in the configuration files in `data_generation/configs/`. Note the following important properties:
- `image_size` - defines the resolution of the images (default: 64)
- `n_instances` - number of episodes to generate
- `n_steps_max` - maximum number of steps before ending the episode
- `n_workers` - number of workers to generate episodes in parallel

## Training
### Training Configuration


This project leverages **Hydra** for flexible configuration management:

- Custom configurations should be added to the `configs/config/` directory.
- Modify `default.yaml` to set new defaults.
- Create new configuration files, with the predifined confifg `yaml` files as their base.

Control is given over model and training parameters, as well as dataset paths. 
For example, dataset arguments are provided in the following format:

```yaml
train:
  dataset_root_dpath: <path_to_dataset_root_directory>
  dataset_name: <dataset_name>
```

And if you want to train the `GenieRedux` or `GenieReduxGuided` model, an already trained tokenizer must be provided:

```yaml
tokenizer_fpath: <path_to_tokenizer>
```

### Running Training

`run.sh` is a wrapper script that we use for all our training and evaluation tasks. It streamlines specifying configurations and overriding parameters.

```bash
bash run.sh --config=<CONFIG_NAME>.yaml
```

Current available configurations are:   

- `tokenizer.yaml` (Default)
- `genie_redux.yaml`
- `genie_redux_guided.yaml`



### CLI Training Customization

You can override parameters directly from the command line. For example, to specify training steps:

```bash
bash run.sh --train.num_train_steps=10000
```

This overrides the `num_train_steps` parameter defined in `configs/config/default.yaml`.

Another example, where we train on 2 GPUs with batch size 2, using a specified tokenizer and dataset:

```bash

./run --num_processes=2 --config=genie_redux_1.yaml --tokenizer_fpath=<path_to_tokenizer> --train.dataset_root_dpath=<path_to_dataset_root_directory> --train.dataset_name=<dataset_name> --train.batch_size=2
```

We note two important parameters:

- `model`: This is the model you want to train. The training and evaluation scripts use this argument to determine which model to instantiate. The available options are:
  - `tokenizer`
  - `genie_redux`
  - `genie_redux_guided`

- `mode`: This determines if we want to train or evaluate a model:
  - `train`
  - `eval`


## Evaluation

With a single script, we provide qualitative and quantitative evaluation. We support two types of evaluation, specified by `eval.action_to_take` argument.

`model_path` parameter should be set to the path of the model you want to evaluate.

### Replication Evaluation

In this evaluation the model attempts to replicate a sequence of observations, given a single observation and a sequence of actions. Qualitative results are stored in `eval.save_root_dpath`. Quantitative results from the comparison with the ground truth are shown at the end of execution.

To enable this evaluation, set `eval.action_to_take=-1`.

### Control Evaluation

This evaluation is meant for qualitative results only. An action of choice, designated by an index (0 to 7 in the case of Coinrun), is given and it is executed by the model for the full sequence, given an input image.

To enable this evaluaiton, set `eval.action_to_take=<ACTION_ID>`, where `<ACTION_ID>` is the aciton index.


### Evaluation Modes

There are two evaluation modes:

- `one_go`: This mode evaluates the model by a one-go inference, using the ground truth actions or the specified action. This mode is faster than the autoregressive mode, and is suitable for testing the model's performance and debugging.

- `autoregressive`: This mode evaluates the model by an autoregressive inference, using the ground truth actions or the specified action. This mode is for better visual quality and controllability, but is computationally expensive.

### Example

```bash
./run.sh --config=genie_redux_1.yaml --mode=eval --eval.action_to_take=1 --eval.model_path=<path_to_model> --eval.inference_method=one_go
```

The above command will evaluate the model at the specified path using the action at index `1`, which corresponds to the `RIGHT` action in the `CoinRun` dataset. You can see the visualizations of the model's predictions in the `./outputs/evaluation/<model-name>/<dataset>/`, dirctory, which is default path for the evaluation results.

## Project Structure

### Data Generation Directory

- **`configs`** - a directory, containing the configuration files for data generation
- **`data/data.py`** - contains definition of the file structure of a generated dataset. This definition is also used by the data handler while training to read the datasets.
- **`generator/generator.py`** - An implementation of a dataset generator - it requests data from a connector that connects it with an environment and saves the data according to a dataset file structure.
- **`generator/connector_coinrun.py`** - A class to connect the generator with the Coinrun environment in order to obtain data.
- **`generate.py`** - a running script for the data generation

### Data Directory
- **`data.py`** - Contains data handlers to load the generated datasets for training. 
- **`data_cached.py`** - A version of `data.py` for use with very large datasets that can benmefit from caching.

### Models Directory

- **`genie_redux.py`**: Implements the GenieRedux model and the guided version.
- **`dynamic_model.py`**: Defines the `MaskGIT` and `Dynamics` models for both `GenieRedux` and `GenieRedux-G`.
- **`tokenizer.py`**: Handles tokenization for training and evaluation.
- **`lam.py`**: Implements the LAM (Learning Agent Module).
- **`construct_model.py`** and **`dynamics.py`**: Construct models and dynamics for Genie-based applications.
- **`components/`**: Modular utilities like attention mechanisms, vector quantization, and more.

### Training Directory

- **`trainer.py`**: Core training functionality, combining dataset preparation and model optimization.
- **`evaluation.py`**: Evaluation logic for trained models.
  - Quantitative evaluation:
    - Fidelity metrics: FID, PSNR, and SSIM
    - Controllability metrics: DeltaPSNR
  - Qualitative evaluation:
    - Visualizing the model's predictions with ground truth or custom input actions.
- **`optimizer.py`**: Custom optimization routines.


## Citation

```bibtex
@inproceedings{kazemi2024learning,
  title={Learning Generative Interactive Environments By Trained Agent Exploration},
  author={Kazemi, Naser and Savov, Nedko and Paudel, Danda Pani and Van Gool, Luc},
  booktitle={NeurIPS 2024 Workshop on Data-driven and Differentiable Simulations, Surrogates, and Solvers}
}
```
