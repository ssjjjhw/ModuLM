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
## Data Process

* We provide dataset processing methods in the dataproces folder, including 2D molecular graph processing and 3D molecular conformation processing. You can choose different processing approaches based on your specific needs e.g..

```bash
python ZhangDDI.py
python ChChMiner.py
python ZhangDDI_3d.py
python ChChMiner_3d.py
```




