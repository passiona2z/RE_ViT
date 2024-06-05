# Reproducibility Project
- Our experiments using PyTorch and Hugging Face frameworks

## Replication experiments   
### Replication, Resolution adjustment, Data sampling, Domain shifting

### ViT
- Replication, Resolution adjustment
  - accelerate launch vit_ddp.py
    - (parser setting) resolution : 512~64
- Data sampling 
  - accelerate launch vit_ddp_sampling.py
    - (parser setting) dataset_ex : True / dataset_ratio : 1.0~0.1
- Domain shifting
  - accelerate launch vit_ddp_domain.py
    - (parser setting) dataset_name 

### BiT(ResNet)
- Replication, Resolution adjustment
  - accelerate launch bit_ddp.py
    - (parser setting) resolution : 512~64
- Data sampling 
  - accelerate launch bit_ddp_sampling.py
    - (parser setting) dataset_ex : True / dataset_ratio : 1.0~0.1
- Domain shifting
  - accelerate launch bit_ddp_domain.py
    - (parser setting) dataset_name 

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
