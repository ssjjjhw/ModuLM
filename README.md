# ModuLM

# Requirements

See `modulm.yml`. Run the following command to create a new anaconda environment `modulm`: 

```bash
conda env create -f modulm.yml
```

# Dataset and Backbone

* **Dataset**

* The datasets used in this project can be downloaded from the [MolTC](https://github.com/MangoKiller/MolTC/).  
All data should be placed in the `/data` folder.

* **LLM Backbone Models**

* The backbones of different Large Language Models (LLMs) can be downloaded from [Hugging Face](https://huggingface.co/).  
Please make sure the downloaded LLMs are stored in the `backbone` folder.

* It is worth noting that in ModuLM, to ensure a fair comparison, we adopted a configuration similar to Galactica. This means that if you download other series of LLMs for extension, you will need to make some modifications to the pretokenizer in the tokenizer vocabulary. For specific changes, you can refer to the tokenizer used in Galactica.

# Usage of ModuLM
* In this section, we will explain how to specifically use our ModuLM framework for training.

## Data Process

* We provide dataset processing methods in the dataproces folder, including 2D molecular graph processing and 3D molecular conformation processing. You can choose different processing approaches based on your specific needs e.g..

```bash
python ZhangDDI.py
python ChChMiner.py
python ZhangDDI_3d.py
python ChChMiner_3d.py
python CombiSolv.py
python CombiSolv_3d.py
```

## ModuLM Config
* We provide the configuration of the best-performing model from our paper, and you can run it directly with `bash python demo.py`.

* You can specify the dataset to use by the config below and the rest of the dataset configuration can follow the default settings. If you wish to make modifications, you can edit the configuration file yourself.
```json
{
  "root": "data/DDI/DeepDDI/"
}
```

* You can specify the molecular data format by setting `use_3d` or `use_2d` to `true`. Correspondingly, you can choose the encoder under the selected data format, as shown in the following example:

```json
{
  "use_3d": true,
  "graph3d": "unimol"
}
```
* You can select the backbone and choose between fine-tuning or pretraining by specifying the configuration as shown below. Note that different methods correspond to different datasets. To select the backbone and set fine-tuning or pretraining mode, use the following configuration:

```json
{
  "mode": "ft",
  "backbone": "DeepSeek-1.5B",
  "min_len": 10,
  "max_len": 40
}
```

* You can use the default configuration for LoRA that we provide.

* The specific methods for calculating the model performance evaluation metrics are provided in the `test_result` folder.






