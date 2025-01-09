# MOF-KAN: Kolmogorov-Arnold Networks for Digital Discovery of Metal-Organic Frameworks
![image](https://github.com/xiaoyu961031/MOF-KAN/blob/main/toc.jpg)

## Overview
Welcome to the official repository for **MOF-KAN**, a framework leveraging **Kolmogorov-Arnold Networks (KAN)** for the digital discovery of Metal-Organic Frameworks (MOFs). 
**MOF-KAN** represents the first application of **KAN** in MOF research, offering comparable accuracy and efficiency than tradition Multilayer Perceptron (MLP) in navigating the diverse chemical and structural landscapes of MOFs. 

## Prerequsite
- tfkan
- keras<=2.15.1
- tensorflow>=2.10.0,<=2.15.1

## Datasets
To reproduce each model under Fine-tuned-models/, please download the accompanied data available at https://zenodo.org/records/14619370

## Build your own MOF-KAN
To set up your **MOF-KAN** framework, navigate to the Experiments/ directory. This directory contains all the essential modules for customizing and building your **MOF-KAN** model. Below is an explanation of the subdirectories and the corresponding steps:

**Step 1**. 0_layer/
This folder is designed for tuning the layer structures of your **KAN** model. It allows you to experiment with:
- The number of neurons in each layer.
- The shape of layers.
- The number of layers in the network.

**Step 2**. 1_embedding/
This folder is dedicated to fine-tuning the embedding layers of your **KAN** model. Embedding layers are critical for representing categorical or structural data in a way that the model can interpret using:
- Transformer.
- attention.
- BiGRU.
- BiLSTM.
- cnn.
- GRU.
- LSTM.
- ResNet.
- TCN.

**Step 3**. 2_baseline/
Once the optimal layer structure (Step 1) and embedding configurations (Step 2) have been established, proceed to compare your KAN model with traditional architectures. This folder contains baseline models using Multilayer Perceptron (MLP). Ensure the same layer structure optimized in Step 1 is employed for a fair comparison.

## References
Our paper is under publication process: Xiaoyu Wu, Xianyu Song, Yifei Yue, Rui Zheng, and Jianwen Jiang. MOF-KAN: Kolmogorov-Arnold Networks for Digital Discovery of Metal-Organic Frameworks.
