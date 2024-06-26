Fork Notes:
1. Test images will retain their original file names (probably).



# Noise2Noise

#### Title
[Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189)

#### Abstract
We apply basic statistical reasoning to signal reconstruction by machine learning -- learning to map corrupted observations to clean signals -- with a simple and powerful conclusion: it is possible to learn to restore images by only looking at corrupted examples, at performance at and sometimes exceeding training using clean data, without explicit image priors or likelihood models of the corruption. In practice, we show that a single model learns photographic noise removal, denoising synthetic Monte Carlo images, and reconstruction of undersampled MRI scans -- all corrupted by different processes -- based on noisy data only.

## Train
    $ python main.py --mode train \
                     --scope [scope name] \
                     --name_data [data name] \
                     --dir_data [data directory] \
                     --dir_log [log directory] \
                     --dir_checkpoint [checkpoint directory]
                     --gpu_ids [gpu id; '-1': no gpu, '0, 1, ..., N-1': gpus]
---
    $ python main.py --mode train \
                     --scope resnet \
                     --name_data bsd500 \
                     --dir_data ./datasets \
                     --dir_log ./log \
                     --dir_checkpoint ./checkpoint
                     --gpu_ids 0

* Set **[scope name]** uniquely.
* To understand hierarchy of directories based on their arguments, see **directories structure** below. 
* Hyperparameters were written to **arg.txt** under the **[log directory]**.


## Test
    $ python main.py --mode test \
                     --scope [scope name] \
                     --name_data [data name] \
                     --dir_data [data directory] \
                     --dir_log [log directory] \
                     --dir_checkpoint [checkpoint directory] \
                     --dir_result [result directory]
                     --gpu_ids [gpu id; '-1': no gpu, '0, 1, ..., N-1': gpus]
---
    $ python main.py --mode test \
                     --scope resnet \
                     --name_data bsd500 \
                     --dir_data ./datasets \
                     --dir_log ./log \
                     --dir_checkpoint ./checkpoints \
                     --dir_result ./results
                     --gpu_ids 0

* To test using trained network, set **[scope name]** defined in the **train** phase.
* Generated images are saved in the **images** subfolder along with **[result directory]** folder.
* **index.html** is also generated to display the generated images.  


## Tensorboard
    $ tensorboard --logdir [log directory]/[scope name]/[data name] \
                  --port [(optional) 4 digit port number]
---
    $ tensorboard --logdir ./log/resnet/bsd500 \
                  --port 6006
                  
After the above comment executes, go **http://localhost:6006**

* You can change **[(optional) 4 digit port number]**.
* Default 4 digit port number is **6006**.

<!--
## Results
![alt text](./img/results.png "Segmentation Images by unet")

    1st row: input; serial section Transmission Electron Microscopy (ssTEM) 
    2nd row: label; sementation map
    3rd row: output; predection map by unet

* The results were generated by a network trained with **em** dataset during **300 epochs**.
* After running the Test phase, execute **display_result.py** to display the figure.
--!>
