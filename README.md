
conda create -n sfr_env python=2.7
source activate sfr_env
conda list -e

# Start Follow Read

This repository is the implementation of the methods described in our paper [Start Follow Read: Full Page End-to-end Handwriting Recognition](http://example.com).
All steps to reproduce our results on the ICDAR 2017 Handwriting Recognition Competition can be found in this repo.
We will also include pretrained models.


## Dependencies

This is a non complete list of dependencies you need

- pytorch  
- opencv 3  
- numpy
- scipy
- pyclipper
- warpctc  (https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings)
- editdistance

## Prepare Data

Download Train-A and Train-B from the competition website. You need `Train-A.tbz2`, `Train-B_batch1.tbz2`, `Train-B_batch2.tbz2`. Put them in the data folder

#### Extract Files

```console
mkdir data
cd data
tar jxf Train-A.tbz2
tar jxf Train-B_batch1.tbz2
tar jxf Train-B_batch2.tbz2
cd ..
```

#### Prepare Train-A

This process can be a bit slow because the normalization code is inefficient.
This extracts start of lines positions, line follower targets, and normalized HW lines.

```console
python preprocessing/prep_train_a.py data/Train-A/page data/Train-A data/train_a data/train_a_training_set.json data/train_a_validation_set.json  
```

#### Prepare Train-B

This extracts only the GT lines from the XML.

```console
python preprocessing/prep_train_b.py data/Train-B data/Train-B data/train_b data/train_b_training_set.json data/train_b_validation_set.json
```

#### Prepare Test data

Currently we only support running the tests for the most difficult Test-B data. When we compute the results for the Test-B while fully exploiting the competition provided regions-of-interest (ROI) we have to do a preprocessing step. This process masks out parts of the image that are not contained in the ROI.

```
python preprocessing/prep_test_b_with_regions.py data/Test-B data/Test-B data/train_b_roi
```

#### Generate Character Settings

This will generate a character set based on the lines in both Train-A and Train-B.
There should 196 unique characters.
This means the network will output 197 characters to include the CTC blank character.

```console
python utils/character_set.py data/train_a_training_set.json data/train_a_validation_set.json data/train_b_training_set.json data/train_b_validation_set.json data/char_set.json
```


## Pretraining

Note that this training is performed using a 32 pixel tall images.
I would recommend training on 32 pixel tall images.
Then training the HWR network after independently at a larger resolution.
The 32 pixel tall images should be accurate enough for a good alignment.

All of these can be run at the same time. You can probably fit all three on a single 8 GB GPU.
Currently there is no stopping criteria except for after a 1000 epochs which is much longer than you should need.
Sorry, no graphs of the training and validation loss at this time.

A sample SLURM file to pretrain can be found in `slurm_examples/pretrain.sh`.

#### Start of Line

You should expect to be done when the validation loss is around 50-60.

```console
python sol_pretraining.py sample_config.yaml  
```

#### Line Follower

You should expect to be done when the validation loss is around 40-50.

```console
python lf_pretraining.py sample_config.yaml  
```

#### Handwriting Recognition

You should expect to be done when the validation CER is around 0.50 to 0.55.

```console
python hw_pretraining.py sample_config.yaml  
```

#### Copy Weights

After pretraining you need to copy the initial weights into the best_overall, best_validation, and current folders.

```console
cp -r data/snapshots/init data/snapshots/best_overall
cp -r data/snapshots/init data/snapshots/best_validation
cp -r data/snapshots/init data/snapshots/current
```

## Training

Training of each component and alignment can be performed independently.
I have run using 4 GPUs.
You could do it on a single GPU but you would have to adapt the code to do that.

For BYU's super computer I run the following SLURM files for 4 GPUs.
You can run the python files independent of the SLURM scripts.

#### Initial Alignment

Before you can train, you have to first run the alignment so there are targets for the validation and the training set.
It will perform alignment over the validation set and the first training group (2000 images total)

```sh
#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2666M

module load cuda/8.0
module load cudnn/6.0_cuda-8.0
module load python
module load python-pytorch
module load opencv/3/0

python -u continuous_validation.py sample_config.yaml init >validation.out 2>&1

```

#### Training

```sh
#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks=24
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2666M

module load cuda/8.0
module load cudnn/6.0_cuda-8.0
module load python
module load python-pytorch
module load opencv/3/0

CUDA_VISIBLE_DEVICES=0 python -u continuous_validation.py sample_config.yaml >validation.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u continuous_sol_training.py sample_config.yaml >sol_training.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u continuous_lf_training.py sample_config.yaml >lf_training.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u continuous_hw_training.py sample_config.yaml >hw_training.out 2>&1 &

wait
```
