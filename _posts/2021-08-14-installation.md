--- 
layout: post
title: Installation of RHA (Red Hen Anonymizer)
---

In this post, I will summarize the deliverables and the installation procedure. 

### Deliverables

The RHA (Red Hen Anonymizer) is used to anonymize the faces in videos. You can either hide the face or replace with some other person's face. 

### Dependencies

The face hider requires only a package of Opencv, ffmpeg-python and MTCNN. So one can follow these commands

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

and follow these instructions for completing installations:

For understanding the face-swapper/replacer, we need to get some idea of the execution of modules. Here, we will analyze only for swapper, as hider is very simple without any complications. 

* The metadata of the input video is extracted. This involves getting start times. 
* Swap the faces:
    - call the script swap.py whose location is fsgan/inference/swap.py
    - this further calls face_detector.py which is located at face_detection_dsfd/face_detector.py 
    - there are some intermediate files and folders formed which store some temporary data. we need not worry about them though. 
- the audio is extracted from video using ffmpeg
- audio is anonymized using audio.py, which is located at ./audio.py
- audio and video are combined using ffmpeg

In the directory containing rha.py clone these two repos
```bash
git clone https://github.com/YuvalNirkin/face_detection_dsfd
git clone https://github.com/YuvalNirkin/fsgan
```

**If your machine has GPUs, do use them or the behaviour is unpexpected** . Correcting this will unnecesarrily increase the amount of manual changes needed. 


(Ignore this if you will anyways use GPUs)
Make these changes in face_detection_dsdf/face_detector.py :
1. line 33
self.net.load_state_dict(torch.load(detection_model_path))
to 
self.net.load_state_dict(torch.load(detection_model_path, map_location=self.device))

2. line 72
torch.set_default_tensor_type('torch.cuda.FloatTensor')
to 
```python
if self.gpus:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
```

(Ignore this if you will anyways use GPUs)
Changes needed in fsgan/criterions/vgg_loss.py
1. line 22:
checkpoint = torch.load(model_path)
to 
```bash
if torch.cuda.is_available():
    checkpoint = torch.load(model_path)
else:
    checkpoint = torch.load(model_path, map_location="cpu")
```

Changes needed in fsgan/inference/swap.py
1. add these lines before all import statements: 

```import sys
sys.path.append('<path-to-directory containing rha.py>')
```

This will make face detection and fsgan modules visible to rha.py . Note that the path to directory has to be absolute path, i.e., the path beginning from root. For example: "/home/yck5/projects" where rha.py will be located as /home/yck5/projects/rha.py

(Ignore this if you will anyways use GPUs)
Changed needed in fsgan/utils/utils.py
1. checkpoint = torch.load(model_path)
to 
checkpoint = torch.load(model_path, map_location=device)


With these changes, we can run rha.py

### Running RHA

rha.py is a single file for running hider and swapper. 

Swapper:
```bash
python rha.py --input <input_video_path> --facepath <path_to_face image> --outpath <path_for_output video> --pitch <pitch change value>
```

For hider, do not use the --facepath option of the above command.
facepath is the imaginary face or a target face that should be present in the anonymized video. It should be clear enough. A 256x256 size photo is usually recommended, but you can try other sizes also. Recabgular photos are not allowed. However, input video can have any size and frame.  


Pitch option is used for anonymizing audio. The value provided will change the pitch by that amount. It has to be an integer. Usually, values near zero, hardly make any changes. Zero value actually, leaves the sound unchanged. Hence if you do  not want to change sound, use --pitch 0. It is known that female voice has high pitch and male voice has low pitch. Hence use a positive value like 3,4,5, etc to make it female-like. Use negative values likes -3, -4, -5 etc to make it more male-like. 

For running of cpu, you need to use --cpu_only flag. This will only work for swapper. For hider, the use of gpu/cpu will depend on the tensorflow-gpu/cpu installed. 

Additionally, we recommend the use of our facebank to get random target faces. 

There are some more installations mentioned in https://github.com/YuvalNirkin/fsgan/wiki/Ubuntu-Installation-Guide . However, these are mostly present in every modern linux distribution. I don't think anybody will ever need to do the apt-get mentioned in this page.