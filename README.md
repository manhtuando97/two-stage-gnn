# Two-stage Training of Graph Neural Networks for Graph Classification
Source code and datasets for training several Graph Neural Network (GNN) architectures for the task of graph classification in the 3 settings described in the main paper: original, 2stg, and 2stg+

In this work, we compare the performance of each GNN architecture on the graph classification task when the architecture is trained in the following 3 settings
*  Original
*  2stg
*  2stg+
Details of these 3 settings are described in the main paper.



## Requirements

The packages and libraries that were used to run the code are:
```setup
torch 1.7.0
networkx 2.5
sklearn 0.23.2
numpy 1.16.4
torch-geometric 1.16.3
```



## Code
```
Code
  |__eigengcn
  |__sag
  |__sage+gat+diffpool
```
The commands to run to train each GNN architecture on the 3 settings (original, 2stg, 2stg+) are described in the 'run_examples.txt' file in the respective folder.



## Datasets
All datasets used in this work are available in this anonymous directory: https://bit.ly/36zCH9b 


