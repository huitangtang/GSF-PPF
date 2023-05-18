# GSF&PPF
Code release for ``Towards Discovering the Effectiveness of Moderately Confident Samples for Semi-Supervised Learning'' published in CVPR 2022.

[Project Page](https://huitangtang.github.io/GSF-PPF/) $\cdot$ [PDF Download](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Towards_Discovering_the_Effectiveness_of_Moderately_Confident_Samples_for_Semi-Supervised_CVPR_2022_paper.pdf)

## Requirements
- python 3.6.4
- pytorch 1.4.0
- torchvision 0.5.0

## Data preparation
The references of the used datasets are included in the paper.

## Model training
1. Install necessary python packages.
2. Replace root and dataset in run.sh with those in one's own system. 
3. Run command `sh run.sh`.

The results are saved in the folder `./results/`.

## Paper citation
```
@InProceedings{tang2022towards,
    author    = {Tang, Hui and Jia, Kui},
    title     = {Towards Discovering the Effectiveness of Moderately Confident Samples for Semi-Supervised Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14658-14667}
}
```
