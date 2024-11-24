# Beyond Imperfections: A Conditional Inpainting Approach for End-to-End Artifact Removal in VTON and Pose Transfer

This repository contains the official code for the paper:

**[Beyond Imperfections: A Conditional Inpainting Approach for End-to-End Artifact Removal in VTON and Pose Transfer](https://arxiv.org/abs/2410.04052)**

## Abstract
In this paper, we propose a novel conditional inpainting method designed to tackle the artifact removal problem in virtual try-on (VTON) and pose transfer tasks. Our approach effectively restores image quality and continuity by generating context-aware image completions, offering significant improvements over existing methods.

## Key Features
- Implementation of the proposed artifact removal pipeline.
- Pretrained models for artifact removal on VTON and pose transfer tasks.
- Scripts for dataset preparation, model training, and inference.

## Inference
You can test it in [Demo](https://colab.research.google.com/drive/1sea7gad2rED0nKJn0s7D7Z_o1aZd6L7X?usp=sharing)

## Results
Examples of artifact removal using our method:
![Result](results/result.png)

## Citation
If you find this code helpful in your research, please consider citing our paper:

```bash
@article{tabatabaei2024beyond,
  title={Beyond Imperfections: A Conditional Inpainting Approach for End-to-End Artifact Removal in VTON and Pose Transfer},
  author={Tabatabaei, Aref and Dehghanian, Zahra and Amirmazlaghani, Maryam},
  journal={arXiv preprint arXiv:2410.04052},
  year={2024}
}
