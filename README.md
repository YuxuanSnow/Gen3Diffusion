# Gen-3Diffusion: Realistic Image-to-3D Generation via 2D & 3D Diffusion Synergy 
#### [Project Page](https://yuxuan-xue.com/gen-3diffusion) | [Paper](https://yuxuan-xue.com/gen-3diffusion/paper/Gen_3Diffusion.pdf)

[Yuxuan Xue](https://yuxuan-xue.com/)<sup>1 </sup>, [Xianghui Xie](https://virtualhumans.mpi-inf.mpg.de/people/Xie.html)<sup>1, 2</sup>, [Riccardo Marin](https://ricma.netlify.app/)<sup>1</sup>, [Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html)<sup>1, 2</sup>


<sup>1</sup>Real Virtual Human Group @ University of Tübingen & Tübingen AI Center \
<sup>2</sup>Max Planck Institute for Informatics, Saarland Informatics Campus

![](https://github.com/YuxuanSnow/Gen3Diffusion/blob/main/assets/teaser_video.gif)

## News :triangular_flag_on_post:
- [2024/12/9] Inference Code release. 
- [2024/12/9] Gen-3Diffusion paper is available on Arxiv.

## Key Insight :raised_hands:
- 2D foundation models are powerful but output lacks 3D consistency!
- 3D generative models can reconstruct 3D representation but is poor in generalization!
- How to combine 2D foundation models with 3D generative models?:
  - they are both diffusion-based generative models => **Can be synchronized at each diffusion step**
  - 2D foundation model helps 3D generation => **provides strong prior informations about 3D shape**
  - 3D representation guides 2D diffusion sampling => **use rendered output from 3D reconstruction for reverse sampling, where 3D consistency is guaranteed**

## Difference to Human-3Diffusion
- We extend the joint 2D-3D diffusion idea on daily objects reconstruction
- We adopt relative camera system in Gen-3Diffusion, because the front-view of objects has ambiguity. Human have clear front-view, and we used absolute camera system in Human-3Diffusion.

## Install
Same Conda environment to Human-3Diffusion. Please skip if you already installed it.
```
# Conda environment
conda create -n gen3diffusion python=3.10
conda activate gen3diffusion
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu121

# Gaussian Opacity Fields
git clone https://github.com/YuxuanSnow/gaussian-opacity-fields.git
cd gaussian-opacity-fields && pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/ && cd ..
export CPATH=/usr/local/cuda-12.1/targets/x86_64-linux/include:$CPATH

# Dependencies
pip install -r requirements.txt

# TSDF Fusion (Mesh extraction) Dependencies
pip install --user numpy opencv-python scikit-image numba
pip install --user pycuda
pip install scipy==1.11
```

## Pretrained Weights
Our pretrained weight can be downloaded from huggingface.
```
mkdir checkpoints_obj && cd checkpoints_obj
wget https://huggingface.co/yuxuanx/gen3diffusion/resolve/main/model.safetensors
wget https://huggingface.co/yuxuanx/gen3diffusion/resolve/main/model_1.safetensors
wget https://huggingface.co/yuxuanx/gen3diffusion/resolve/main/pifuhd.pt
cd ..
```
The avatar reconstruction module is same to Human-3Diffusion. Please skip if you already installed Human-3Diffusion.
```
mkdir checkpoints_avatar && cd checkpoints_avatar
wget https://huggingface.co/yuxuanx/human3diffusion/resolve/main/model.safetensors
wget https://huggingface.co/yuxuanx/human3diffusion/resolve/main/model_1.safetensors
wget https://huggingface.co/yuxuanx/human3diffusion/resolve/main/pifuhd.pt
cd ..
```

## Inference
```
# given one image of object, generate 3D-GS object
# subject should be centered in a square image, please crop properly 
# recenter plays a huge role in object reconstruction. Please adjust the recentering if the reconstruction doesn't work well
python infer.py --test_imgs test_imgs_obj --output output_obj --checkpoints checkpoints_obj

# given generated 3D-GS, perform TSDF mesh extraction
python infer_mesh.py --test_imgs test_imgs_obj --output output_obj --checkpoints checkpoints_obj --mesh_quality high
```

``` 
# given one image of human, generate 3D-GS avatar
# subject should be centered in a square image, please crop properly
python infer.py --test_imgs test_imgs_avatar --output output_avatar --checkpoints checkpoints_avatar

# given generated 3D-GS, perform TSDF mesh extraction
python infer_mesh.py --test_imgs test_imgs_avatar --output output_avatar --checkpoints checkpoints_avatar --mesh_quality high
```

## Citation :writing_hand:

```bibtex
@inproceedings{xue2024gen3diffusion,
  title     = {{Gen-3Diffusion: Realistic Image-to-3D Generation via 2D & 3D Diffusion Synergy }},
  author    = {Xue, Yuxuan and Xie, Xianghui and Marin, Riccardo and Pons-Moll, Gerard.},
  journal   = {Arxiv},
  year      = {2024},
}

@inproceedings{xue2024human3diffusion,
  title     = {{Human 3Diffusion: Realistic Avatar Creation via Explicit 3D Consistent Diffusion Models}},
  author    = {Xue, Yuxuan and Xie, Xianghui and Marin, Riccardo and Pons-Moll, Gerard.},
  journal   = {NeurIPS 2024},
  year      = {2024},
}
