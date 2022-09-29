# PLCJ
This is the code for PLCJ (Partition and Learned Clustering with Joined-training:Active Learning of GNNs on Large-scale Graph). Running the code requires the device to have a GPU and install CUDA. PLCJ focuses on the active learning of attributed undirected graphs. In order to train a strong GNN model under limited labels, PLCJ searches for representative nodes to form the training set. PLCJ partitions the large-scale graph into multiple subgraphs, and then clusters and selects the central node within each subgraph.
<div align=center>
<img src="https://github.com/jianjianGJ/PLCJ/blob/main/frame.png" width="350" height="350" />
</div>
## Datasets
We used datasets Reddit, ogbn-arxiv, ogbn-mag, ogbn-products in the experiments. When run `main.py`, the specified dataset will be automatically downloaded, here is the info table of the dataset. 

|  Dataset |#Classes|#Features|#Nodes| #Edges|
|   :----: | :----: | :----: | :----: | :----: |
| Arxiv    | 40 | 128 |169343| 1166243|
| Mag      | 349 |128 |736389| 5396336|
| Products | 47 |100 | 2449029| 61859140|
| Reddit   | 41 |602 |232965| 11606919|

## Run
run `python main.py --dataset=arxiv --need=500 --n-run=1 --mode=PLCJ --gpu=0`

## Requirement
* ogb       1.3.1
* torch     1.8.0+cu111
* dgl       0.7.1
