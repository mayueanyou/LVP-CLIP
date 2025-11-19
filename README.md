# LVP-CLIP: Revisiting CLIP for Continual Learning with Label Vector Pool 🚀

[[Paper (CVF Open Access)](https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/html/Ma_LVP-CLIP_Revisiting_CLIP_for_Continual_Learning_with_Label_Vector_Pool_CVPRW_2025_paper.html)] [[PDF]](https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/papers/Ma_LVP-CLIP_Revisiting_CLIP_for_Continual_Learning_with_Label_Vector_Pool_CVPRW_2025_paper.pdf)

**Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2025.**

## Authors

Yue Ma, Huantao Ren, Boyu Wang, Jingang Jin, Senem Velipasalar, Qinru Qiu

-----

We use the offcial CLIP model from (https://github.com/openai/CLIP) and include it in the local folder for convernience.

## Abstract

Continual learning (CL) aims to update a model so that it can sequentially learn new tasks without forgetting previously acquired knowledge. Recent CL approaches often leverage the vision-language model **CLIP** for its high-dimensional feature space and cross-modality feature matching capabilities.

However, traditional CLIP-based classification is sensitive to the quality of text phrases and less effective for classes lacking meaningful text labels. In this work, we **rethink CLIP-based continual learning** and introduce the concept of **Label Vector Pool (LVP)**. LVP replaces text labels with **training images** as similarity references, eliminating the need for ideal text descriptions. We present three variations of LVP and demonstrate their performance on class and domain incremental learning tasks.

-----

## Key Contributions

  * **Introduction of Label Vector Pool (LVP):** A novel approach that uses the high-dimensional features of training images as similarity references, moving away from text-based labels in CLIP-based continual learning.
  * **Enhanced Stability and Efficiency:** LVP learning algorithms are **task-order invariant** and result in **minimum catastrophic forgetting** because new knowledge does not modify the old knowledge. Tasks can be learned independently and in parallel with low computational and memory demands.
  * **State-of-the-Art Performance:** Our proposed LVP-based methods significantly **outperform the current state-of-the-art baseline by a margin of 40.7%** on Cross-Task Incremental Learning(CTIL) tasks.

-----

![Screenshot of the main app window](images/method.png)

![Screenshot of the main app window](images/gusian.png)

## Proposed Method: Label Vector Pool (LVP)

The core idea of **LVP** is to leverage the rich feature space of CLIP by directly using **visual embeddings** from training samples as class prototypes, rather than relying on potentially noisy or inadequate text embeddings.

  * LVP eliminates the dependency on the quality of hand-crafted or auto-generated text prompts/labels.
  * We explore **three variations** of the LVP strategy to maximize performance and efficiency across different continual learning scenarios.
  * -----

## Setup and Installation

### Prerequisites

  * Python 3.x
  * PyTorch
  * [Insert other dependencies like torchvision, CLIP library, etc.]

### Installation

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)[Your_Username]/LVP-CLIP.git
cd LVP-CLIP

# Create a conda environment (optional but recommended)
conda create -n lvp_clip python=3.x
conda activate lvp_clip

# Install dependencies
pip install -r requirements.txt
```


## Citation

If you find this work useful for your research, please cite our paper:

```
@InProceedings{Ma_2025_CVPR,
    author    = {Ma, Yue and Ren, Huantao and Wang, Boyu and Jin, Jingang and Velipasalar, Senem and Qiu, Qinru},
    title     = {LVP-CLIP: Revisiting CLIP for Continual Learning with Label Vector Pool},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {231-240}
}
```
