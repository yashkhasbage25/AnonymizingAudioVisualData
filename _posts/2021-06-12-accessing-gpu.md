---
layout: post
title: Accessing GPUs on CWRU HPC
---

## Refernces

* https://sites.google.com/case.edu/techne-public-site/singularity
* https://sites.google.com/a/case.edu/hpcc/
* https://sites.google.com/a/case.edu/hpcc/guides-and-training/faqs
* https://sites.google.com/a/case.edu/hpcc/hpc-cluster/quick-start/jobs
* https://sites.google.com/a/case.edu/hpcc/hpc-cluster/quick-start/jobs/job-control
* https://sites.google.com/a/case.edu/hpcc/hpc-cluster/quick-start/jobs/examples
* https://sites.google.com/a/case.edu/hpcc/hpc-cluster/important-notes-for-new-users/slurm-command-overview

## Accessing without creating container


```bash
module load singularity # is important to invoke singularity and srun command 
module load cuda
srun -p gpu --gpus 1 --mem 4000 --cpus-per-gpu 2 --pty bash # do -h to know more options
# you will be allocated gpu
nvidia-smi # to see gpu
pestat -p gpu -w <hostname> # this will show memory usage, has to be run on another terminal
```

One can also use SLURM as mentioned in example here: https://sites.google.com/a/case.edu/hpcc/hpc-cluster/quick-start/jobs/batch and https://sites.google.com/a/case.edu/hpcc/hpc-cluster/important-notes-for-new-users/slurm-command-overview


## Using a container

Refernce: https://github.com/singularityhub/singularityhub.github.io/wiki/Build-A-Container

First, you can pull a singularity container from shub of RedHenLab: https://sites.google.com/case.edu/techne-public-site/singularity
You have to create a singularity.tag file that is supposed to install all software that your program depends on. Then, while running the container, add the --nv flag (https://sylabs.io/guides/3.8/user-guide/gpu.html#nvidia-gpus-cuda)

### Comments from Prof Mark

Singularity is a container for your final product:
the purpose of Singularity (like Docker) is to create a functionality that can be moved across all platforms without any adjustment.
You place inside the Singularity container everything that is needed.  Then, we could take your Singularity container from CWRU HPC
and move it to Germany or China or Brazil or UCLA, and it would work just the same.  At the end of the day, we need your final 
product inside a Singularity container on CWRU HPC.  Then we could move it to other HPCs around the world, and it would run.  
You don’t need to develop that product inside Singularity. You just need to put it inside a Singularity container at the end.
