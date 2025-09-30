# DeepASA: An object-oriented multi-purpose network for auditory scene analysis


[![PWC](https://img.shields.io/badge/NeurIPS-paper-red)](https://arxiv.org/pdf/2509.17247)
[![PWC](https://img.shields.io/badge/ASA2-dataset-yellow)](https://huggingface.co/datasets/donghoney22/ASA2_dataset)
[![PWC](https://img.shields.io/badge/HuggingFace-demo-blue)](https://huggingface.co/spaces/donghoney22/DeepASA)

Official implementation of Neural Information Processing Systems (NeurIPS) 2025 paper **"[DeepASA: An object-oriented multi-purpose network for auditory scene analysis](https://arxiv.org/pdf/2509.17247) (accepted)"**.

*We propose DeepASA, a multi-purpose model for auditory scene analysis that performs multi-input multi-output (MIMO) source separation, dereverberation, sound event detection (SED), audio classification, and direction-of-arrival estimation (DoAE) within a unified framework. DeepASA is designed for complex auditory scenes where multiple, often similar, sound sources overlap in time and move dynamically in space. To achieve robust and consistent inference across tasks, we introduce an object-oriented processing (OOP) strategy. This approach encapsulates diverse auditory features into object-centric representations and refines them through a chain-of-inference (CoI) mechanism. The pipeline comprises a dynamic temporal kernel-based feature extractor, a transformer-based aggregator, and an object separator that yields per-object features. These features feed into multiple task-specific decoders. Our object-centric representations naturally resolve the parameter association ambiguity inherent in traditional track-wise processing. However, early-stage object separation can lead to failure in downstream ASA tasks. To address this, we implement temporal coherence matching (TCM) within the chain-of-inference, enabling multi-task fusion and iterative refinement of object features using estimated auditory parameters. We evaluate DeepASA on representative spatial audio benchmark datasets, including ASA2, MC-FUSS, and STARSS23. Experimental results show that our model achieves state-of-the-art performance across all evaluated tasks, demonstrating its effectiveness in both source separation and auditory parameter estimation under diverse spatial auditory scenes.*

![DeFTAN-II figure](fig/Fig_overall_architecture.png)

## 1. Setup
1. Clone repository
```
git clone https://github.com/donghoney0416/DeFTAN-II.git
cd DeFTAN-II
```

2. Install requirements
```
pip install -r requirements.txt
```

## 2. Details
### Dataset
The dataset was simulated using pyroomacoustics. See `generate_rir/gen_rir.py` for an example of the simulation code, and `generate_rir/pyroom_rir.cfg` for the configuration file.

### Model
We released the code so that the model could be trained from scratch, and we uploaded a pre-trained model, trained on the spatialized DNS Challenge dataset, to Hugging Face. 
See `DeFTAN2.py` to adjust the parameters or change modules for custom training.

### Loss
The model was trained using PCM loss and SI-SDR loss; PCM loss was uploaded as the primary loss. See `pcm_loss.py` for details, and feel free to modify it as needed.

### Using pre-traind model [![PWC](https://img.shields.io/badge/HuggingFace-pre_trained_model-yellow)](https://huggingface.co/donghoney0416/DeFTAN-II)
We have uploaded the pre-trained model and instructions for use on Hugging Face. Thank you for exploring and using DeFTAN-II.

## 3. Results and Demos [![PWC](https://img.shields.io/badge/Demo-webpage-blue)](https://donghoney0416.github.io/demos-DeFTAN-II/demo-page.html)
We have uploaded more audio clips and spectrogram examples to our demo page. Results from five datasets are provided: the spatialized WSJCAM0 dataset, the spatialized DNS Challenge dataset, the spatialized WSJ0-2mix dataset, the CHiME-3 real dataset, and the EasyCom dataset. This includes sound source separation, real-world speech enhancement, and more. Spectrograms and audio clips can be downloaded directly from the `fig` and `audio` directories, respectively.

![result](fig/results.PNG)

## Citations
```
@article{lee2025deepasa,
  title={DeepASA: An object-oriented multi-purpose network for auditory scene analysis},
  author={Lee, Dongheon, Kwon Younghoo and Choi, Jung-Woo},
  journal={in Proc. Conference on Neural Information Processing Systems},
  year={2025}
}
```
