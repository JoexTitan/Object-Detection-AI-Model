
![Logo](https://raw.githubusercontent.com/JoexTitan/Object-Detection-Model/main/utils/ObjectDetection.jpg)


# Object-Detection-Model

The following machine learning model aims to identify objects in the video and predict the trajecory that they are moving in. The image processing algorithm will detect instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos.

## Deployment Instructions

Ensure that you have the following for both GPU and CPU Installations: 

- PyCharm Community Edition or any other IDE of your choice
- Anaconda (https://www.anaconda.com/products/distribution)
- NVIDIA Drivers (https://www.nvidia.com/Download/index.aspx)
- CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)

## Model Performance on Mall CCTV Camera

<p>
  <img alight='center' alt='gif' src='https://github.com/JoexTitan/Object-Detection-AI-Model/blob/main/yolor/mall_detection.gif?raw=true' width='850' height='500'/></p>


## Model Performance on Figure Skaters

<p>
  <img alight='center' alt='gif' src='https://github.com/JoexTitan/Object-Detection-AI-Model/blob/main/yolor/skating_detection.gif?raw=true' width='850' height='500'/></p>


## Model Performance on Highway Traffic

<p>
  <img alight='center' alt='gif' src='https://github.com/JoexTitan/Object-Detection-AI-Model/blob/main/yolor/highway_traffic.gif?raw=true' width='850' height='500'/></p>

## Model Performance on Soccer Players

<p>
  <img alight='center' alt='gif' src='https://github.com/JoexTitan/Object-Detection-AI-Model/blob/main/yolor/soccer_detection.gif?raw=true' width='650' height='800'/></p>



## Model Performance on a Tennis Player

<p>
  <img alight='center' alt='gif' src='https://github.com/JoexTitan/Object-Detection-AI-Model/blob/main/yolor/tennis_detection.gif?raw=true' width='650' height='800'/></p>


GPU Installation, set-up a virtual environment with the following command

```bash
  conda env create -f environment.yml
```
Then activate your environment

```bash
  conda activate <Name_of_the_Project>
```

Check which drivers you are using

```bash
  nvcc --version
```

You should be seeing the following

```bash
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2019 NVIDIA Corporation
  Built on Fri_Feb__8_19:08:26_Pacific_Standard_Time_2019
  Cuda compilation tools, release 10.1, V10.1.105
```

Make sure conda is saved in your Environment Variables (PATH)

```bash
   C:\Users\Daniil_Zhilyayev\Anaconda3\Scripts

   C:\Users\Daniil_Zhilyayev\Anaconda3

   C:\Users\Daniil_Zhilyayev\Anaconda3\Library\bin
```

You are all set, here is a few commands for you to get started:

```bash
  python motioned_detection.py --source videos/yolor/F1_CARS_DETECTION.mp4 --cfg cfg/yolor_p6.cfg --weights yolor_p6.pt --conf 0.25 --img-size 1280 --device 0 --view-img

  python motioned_detection.py --source videos/yolor/F2_MALL_DETECTION.mp4 --cfg cfg/yolor_p6.cfg --weights yolor_p6.pt --conf 0.50 --img-size 1280 --device 0 --view-img

  python motioned_detection.py --source videos/yolor/F5_SKATING_DETECTION.mp4 --cfg cfg/yolor_p6.cfg --weights yolor_p6.pt --conf 0.75 --img-size 1280 --device 0 --view-img
```



