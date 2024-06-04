# Reproducibility Project
- our experiments using PyTorch and Hugging Face frameworks

## Replication experiments   
### Replication, Resolution adjustment, Data sampling, Domain shifting

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
- study/vit.py
  - main file which selects vit model and train the model.
  - vit2_1.py : SGD optimizer, cosine learning rate decay (Replication original model)
  - vit4.py, vit_4_1.py : adam optimizer
  - vit5.py: SGD optimizer, linear learning rate decay
  - vit6.py: SGD optimizer, exponential learning rate decay
  - vit7.py: SGD optimizer, ReduceLROnPlateau learning rate decay
- study/model_with_positional_embedding.py, study/model_without_positional_embedding.py
  - model files are separated with two kinds, one is the model with adding positional embedding and the other is the model without adding positional embedding.
- study/patchdata.py 
  - patchdata file make patches according to size of image and patch.
- study/test.py
  - test file derives accuracy and loss from test data.
