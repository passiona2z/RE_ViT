# Reproducibility Project
- our experiments using PyTorch and Hugging Face frameworks

## Replication experiments (Resolution adjustment, Data sampling, Domain shifting)

### ViT
- (DDP) CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch vit_ddp.py
- (DDP & Data sampling) CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch vit_ddp_sampling.py

### BiT(ResNet)
- (DDP) CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch bit_ddp.py
- (DDP & Data sampling) CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch bit_ddp_sampling.py

## Ablation and Parameter Study
- study/vit_ablation.ipynb 
- study/vit_parameter_stduy.ipynb
- study/visualization.ipynb
  - visualize : linear projection, Positional embedding, Tranformer attention matrix