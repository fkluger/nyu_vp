# NYU-VP: Vanishing Point Labels for the NYU Depth v2 Dataset

![example](example.jpg)

If you use this dataset and code, please cite [our paper](https://arxiv.org/abs/2001.02643):
```
@inproceedings{kluger2020consac,
  title={CONSAC: Robust Multi-Model Fitting by Conditional Sample Consensus},
  author={Kluger, Florian and Brachmann, Eric and Ackermann, Hanno and Rother, Carsten and Yang, Michael Ying and Rosenhahn, Bodo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
Please also cite the corresponding [NYU Depth v2 paper](https://cs.nyu.edu/~silberman/bib/indoor_seg_support.bib)! 


## Prerequisites
In order to use the original RGB images as well, you need to obtain the original 
[dataset MAT-file](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) and convert it to a 
*version 7* MAT-file in MATLAB so that we can load it via scipy:
```
load('nyu_depth_v2_labeled.mat')
save('nyu_depth_v2_labeled.v7.mat','-v7')
```

## Installation
Get the code:
```
git clone --recurse-submodules git@github.com:fkluger/nyu_vp.git
cd nyu_vp
```

Set up the Python environment using [Anaconda](https://www.anaconda.com/): 
```
conda env create -f environment.yml
source activate nyu_vp
```

Build the [LSD line segment detector](https://www.ipol.im/pub/art/2012/gjmr-lsd/) module:
```
cd lsd
python setup.py build_ext --inplace
```

## Usage
In order to visualise the dataset, run:
```
python nyu.vp --mat_file nyu_depth_v2_labeled.v7.mat
```

For usage within your own project, refer to the ```NYUVP``` class:
```
from nyu import NYUVP

dataset = NYUVP(
            data_dir_path="./data",     # Path where the CSV files containing VP labels etc. are stored
            split='all',                # train, val, test, trainval or all
            keep_data_in_memory=True,   # whether data shall be cached in memory
            mat_file_path=None,         # path to the MAT file containing the original NYUv2 dataset
            normalise_coordinates=False,# normalise all point coordinates to a range of (-1,1)
            remove_borders=True,        # ignore the white borders around the NYU images
            extract_lines=False         # do not use the pre-extracted line segments
          )
          
idx = 0
sample = dataset[idx] # get a single sample from the dataset
VPs = sample['VPs'] # array Mx3 with vanishing points in homogeneous coordinates
image = sample['image'] # RGB image
lines = sample['line_segments'] # array Nx12 containing all extracted line segments
p1 = lines[:, 0:3] # line segment start points in hom. coordinates
p2 = lines[:, 3:6] # line segment end points in hom. coordinates
hom_lines = lines[:, 6:9] # parametrised line [a,b,c] s.t. ax+by+c=0
centroids = lines[:, 9:12] # centroid = (p1+p2)/2.
```
