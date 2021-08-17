--- 
layout: post
title: Installation of RHA (Red Hen Anonymizer)
---

In this post, I will summarize the deliverables and the installation procedure. 

### Deliverables

The RHA (Red Hen Anonymizer) is used to anonymize the faces in videos. You can either hide the face or replace with some other person's face. 

### Dependencies

*RHA strictly uses python3. There is no preferred version of python3, but any latest version will work.*

Audio anonymizer requires sox

```bash
pip install sox # do not use conda install
```

The face hider requires packages of opencv, ffmpeg-python and MTCNN. So one can follow these commands

```bash
conda install -c conda-forge opencv
conda install -c conda-forge mtcnn
pip install ffmpeg-python
```

These are a few installations for swapper

```bash
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia # actually any latest version will work
conda install -c conda-forge yacs
conda install -c conda-forge tqdm
conda install -c conda-forge matplotlib
pip install --upgrade tensorflow
pip install tensorboardX
pip install ffmpeg-python
```

Clone the repo 
```bash
git clone https://github.com/yashkhasbage25/AnonymizingAudioVisualData.git --depth 1 
# remove --depth 1 if you want to check how the code was gradually developed
``` 

**If your machine has GPUs, do use them or the behaviour is unpexpected** . Correcting this will unnecesarrily increase the amount of manual changes needed. 


### Not everyone is allowed to run RHA

The FSGAN present in RHA can be used for defaming or creating DeepFakes. Hence, its usage it not open to general public. The pretrained weights for FSGAN are not provided in this public repository for the same purpose. 

If you want to get access to pretrained weights of FSGAN directly from FSGAN team, see the page https://github.com/YuvalNirkin/fsgan/wiki/Paper-Models-Inference and fill out their form. Upon knowing your purpose of using FSGAN,  they will share you a script download_fsgan_models.py. You need to place it at AnonymizingAudioVisualData/fsgan/download_fsgan_models.py. Change its line 
```python 
from fsgan.utils.utils import download_from_url
```
to 
```python
from utils.utils import download_from_url
``` 
The file download_fsgan_models.py can also be requested from the RedHen mentors (specifically, Mark Turner, Francis Steen, Peter Uhrig, Karan Singla, Daniel Alcaraz). Then the same instructions as mentioned above, can be followed. 

Then, run the script
```bash
python download_fsgan_models.py -m v2 
```

This will download the pretrained models at correct places.

### Downloading Face Detector Weights

Install gdown using 
```bash
pip install gdown
```

and download the weights 

```bash
gdown https://drive.google.com/u/0/uc?id=1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-
```

Place the weights at face_detection_dsfd/weights/WIDERFace_DSFD_RES152.pth

Unlike the FSGAN weights, these weights are publicly available. 

*** RHA face-swapper cannot be used at all without this step ***

### Running RHA

#### For CWRU HPC Users

There are some kernel incompatibilities for K40 GPUs. Hence, the following constraint `-C 'gpu2v100|gpu4v100'` has to be added for gpu types.

```bash
srun -p gpu --gpus 1 --mem 8000 -C 'gpu2v100|gpu4v100' --pty bash
```

Enter into the cloned repo
```bash
cd AnonymizeAudioVisualData
```
There you can find rha.py. It is a single file for running hider and swapper. 

Swapper:
```bash
python rha.py --input <input_video_path> --facepath <path_to_face image> --outpath <path_for_output video> --pitch <pitch change value>
```

For hider, do not use the --facepath option of the above command.


facepath is the imaginary face or a target face that should be present in the anonymized video. It should be visually visible. A 256x256 size photo is usually recommended, but you can try other sizes also. Rectangular photos are not allowed. However, input video can have any size and frame.  


Pitch option is used for anonymizing audio. The value provided will change the pitch by that amount. It has to be an integer (both positive and negative integers). Usually, values near zero, hardly make any changes. Zero value actually, leaves the sound unchanged. Hence if you do  not want to change sound, use --pitch 0. 

It is known that female voice has high pitch and male voice has low pitch. Hence use a positive value like 3,4,5, etc to make it female-like. Use negative values likes -3, -4, -5 etc to make it more male-like. 

For running on cpu, you need to use --cpu_only flag. This will only work for swapper. For hider, the use of gpu/cpu will depend on the tensorflow-gpu/cpu installed. 

Additionally, we recommend the use of our facebank to get random target faces. (https://drive.google.com/drive/folders/1EGiVI3fMLwNiYG-Es-Sy5qqie9Co2eZI?usp=sharing)

There are some more installations mentioned in https://github.com/YuvalNirkin/fsgan/wiki/Ubuntu-Installation-Guide . However, these are mostly present in every modern linux distribution. I don't think anybody will ever need to do the apt-get mentioned in this page.

If your machine has 4 gpus, and you want to use only gpus 0,3 (indexing starts at 0) then do

```bash
CUDA_VISIBLE_DEVICES=0,3 python rha.py <remaining_options>
```

### Demo 

Download the video covid.mp4 and face-image random_face.jpg from https://drive.google.com/drive/folders/1y3kytBHZULQ2gLDL0xKBmq2oR4uPx0D4?usp=sharing .
Place them along with rha.py and run

Face Hider:
```bash
python rha.py --inpath covid.mp4 --outpath hider_output.mp4 --pitch -4
```

Face Swapper:
```bash
python rha.py --inpath covid.mp4 --facepath random_face.jpg --outpath swapper_output.mp4 --pitch -4
```

You can set pitch according to your choice. But since, the target face is male, we prefer a low pitched voice. We encourage you to try out several target faces, by downloading the facebank. (https://drive.google.com/drive/folders/1EGiVI3fMLwNiYG-Es-Sy5qqie9Co2eZI?usp=sharing)