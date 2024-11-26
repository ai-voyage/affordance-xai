# Towards Explainability of Affordance Learning in Robot Vision

The official repository for "Towards Explainability of Affordance Learning in Robot Vision", published at DICTA2024, Perth


## Introduction
This repository supports the paper **"Towards Explainability of Affordance Learning in Robot Vision"**, presented at **DICTA, 2024**. The paper introduces a novel post-hoc multimodal explainability framework that capitalizes on the emerging synergy between visual and language models. Our proposed multimodal system provides human-like explanations to answer "what" the autonomous machines look at and "what" they think of the functional affordances offered by the objects they are looking at. This repository includes a sample of the dataset, code, and models used in our experiments. The contributions of this paper are summarised below: 

- We propose a post-hoc multimodal deep learning-based and multi-label object affordance recognition framework for the interpretability of intelligent system predictions.
- In addition to leveraging the post-hoc visual attribution heatmaps as a visual modality, the state-of-the-art LLM models are leveraged as a language modality that provides an intuitive description of the visual explanations.
- We evaluate our proposed approach on a comprehensive benchmark dataset using an alignment XAI evaluation metric for the affordance learning task against the ground truth. Our experimental results demonstrate the proposed framework's superior performance.
- The proposed framework can be applied to any autonomous systems powered by DNN architectures, including Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs).

## Release
- [26/11/2024] Training code for Swin-T released. Training codes for other models will be released soon.

## Installation
To set up the repository on your local machine:
1. Clone the repository:
```
   git clone https://github.com/ai-voyage/affordance-xai.git
   cd affordance-xai
```
3. Create a virtual environment and install dependencies
```
  conda create -n your_env_name python=3.9
  conda activate your_env_name
  pip install -r requirements.txt
```


## Usage
1. Data Preparation:
   - Download the dataset available at:  [A large scale multi-view RGBD visual affordance learning dataset](https://sites.google.com/view/afaqshah/datasets?authuser=0)
   - Extract "Json Files.zip" and "RGB images.zip" into data folder. We have only provided a small sample of the dataset in this repo.
   - Ensure the dataset is available at the correct paths.
2. Training:
   - Run the training script:
```bash
python data_train_pipeline.py
 ```

3. CAM visualization
   - Generate the CAM based affordance heatmaps:
```bash
python swin_gradcam_generator.py
```
4. Generate Textual Explanations of Affordances:
   - Obtain an OpenAI api key.
   - Past it in GPT component/gpt_affordance.py. Run the textual affordance explainability generator:
```bash
python GPT component/gpt_affordance.py
```

