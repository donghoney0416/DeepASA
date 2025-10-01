# DeepASA: An object-oriented multi-purpose network for auditory scene analysis


[![PWC](https://img.shields.io/badge/NeurIPS-paper-red)](https://arxiv.org/pdf/2509.17247)
[![PWC](https://img.shields.io/badge/ASA2-dataset-yellow)](https://huggingface.co/datasets/donghoney22/ASA2_dataset)
[![PWC](https://img.shields.io/badge/HuggingFace-demo-blue)](https://huggingface.co/spaces/donghoney22/DeepASA)

Official implementation of **"[DeepASA: An object-oriented multi-purpose network for auditory scene analysis](https://arxiv.org/pdf/2509.17247) (NeurIPS 2025)"**.

*We propose DeepASA, a multi-purpose model for auditory scene analysis that performs multi-input multi-output (MIMO) source separation, dereverberation, sound event detection (SED), audio classification, and direction-of-arrival estimation (DoAE) within a unified framework. DeepASA is designed for complex auditory scenes where multiple, often similar, sound sources overlap in time and move dynamically in space. To achieve robust and consistent inference across tasks, we introduce an object-oriented processing (OOP) strategy. This approach encapsulates diverse auditory features into object-centric representations and refines them through a chain-of-inference (CoI) mechanism. The pipeline comprises a dynamic temporal kernel-based feature extractor, a transformer-based aggregator, and an object separator that yields per-object features. These features feed into multiple task-specific decoders. Our object-centric representations naturally resolve the parameter association ambiguity inherent in traditional track-wise processing. However, early-stage object separation can lead to failure in downstream ASA tasks. To address this, we implement temporal coherence matching (TCM) within the chain-of-inference, enabling multi-task fusion and iterative refinement of object features using estimated auditory parameters. We evaluate DeepASA on representative spatial audio benchmark datasets, including ASA2, MC-FUSS, and STARSS23. Experimental results show that our model achieves state-of-the-art performance across all evaluated tasks, demonstrating its effectiveness in both source separation and auditory parameter estimation under diverse spatial auditory scenes.*

![DeepASA figure](figure/DeepASA.png)

## 1. Setup
1. Clone repository
```
git clone https://github.com/donghoney0416/DeepASA.git
cd DeepASA
```

2. Install requirements
```
pip install -r requirements_.txt
```

## 2. Details
### Dataset
We constructed a new dataset, Auditory Scene Analysis V2 (ASA2) dataset for multichannel USS and polyphonic audio classification tasks. The proposed dataset is designed to reflect various conditions, including moving sources with temporal onsets and offsets. For foreground sound sources, signals from 13 audio classes were selected from open-source databases (Pixabay¹, FSD50K, Librispeech, MUSDB18, Vocalsound). Specific information and how to download the dataset can be found at the hugging face link below.

[ASA2 dataset link](https://huggingface.co/datasets/donghoney22/ASA2_dataset)

### Training
Training DeepASA from scratch
```
python SharedTrainer.py fit --config=configs/DeepASA.yaml --configs/dataset/auditory_scene_analysis.yaml --data.batch_size=[2,2] --trainer.devices=[0,1,2,3] --trainer.max_epochs=100
```

### Inference
You can evaluate the model you trained by appropriately modifying the code below
```
python SharedTrainer.py test --config=configs/logs/DeepASA/version_0/config.yaml --checkpoints=configs/logs/DeepASA/version_0/checkpoints/last.ckpt --data.batch_size=[2,2] --trainer.devices=[0,1,2,3]
```

## Citations
```
@article{lee2025deepasa,
  title={DeepASA: An object-oriented multi-purpose network for auditory scene analysis},
  author={Lee, Dongheon, Kwon Younghoo and Choi, Jung-Woo},
  journal={in Proc. Conference on Neural Information Processing Systems},
  year={2025}
}
```
