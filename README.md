# PLCJ
This is the code for paper "Partition and Learned Clustering with Joined-training:Active Learning of GNNs on Large-scale Graph". 
![image](https://github.com/jianjianGJ/PLCJ/blob/main/frame.png)
Running the code requires the device to have a GPU and install CUDA.

## Run
run `python main.py --dataset=arxiv --need=500 --n-run=1 --mode=PLCJ --gpu=0`

## Requirement
* ogb       1.3.1
* torch     1.8.0+cu111
* dgl       0.7.1
