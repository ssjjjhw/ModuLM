# ModuLM

## Requirements

See `modulm.yml`. Run the following command to create a new anaconda environment `modulm`: 

```bash
conda env create -f modulm.yml
```

# Dataset and Model Usage Instructions

## Dataset

The datasets used in this project can be downloaded from the [MolTC project](https://github.com/MangoKiller/MolTC/).  
All data should be placed in the `/data` folder.

## LLM Backbone Models

The backbones of different Large Language Models (LLMs) can be downloaded from Hugging Face.  
Please make sure the downloaded LLMs are stored in the `backbone` folder.

## DDI Tasks

For Drug–Drug Interaction (DDI) tasks, we provide code for training separately on all DDI datasets.  
We plan to release code for joint training across all DDI datasets in the future.

## Solvation Gibbs Free Energy Prediction

For solvation Gibbs free energy prediction tasks:

- You can run the pretraining stage using the provided pretraining data.
- Fine-tuning may converge in just a few epochs, so early stopping or truncation is recommended.

