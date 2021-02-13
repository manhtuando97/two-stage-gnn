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

## Datasets
** The datasets used in this work are available in the 'Datasets' folder as follows:
```
Datasets
  |__benchmark
  |__taxi
```

Refer to the Supplementary document for more details on the datasets.


| Name           | #Graphs    | Average #Nodes    | Average #Edges    | # Classes |
|----------------|------------|-------------------|-------------------|-----------|
| DD             |    1,168   |        268.71     |       676.21      |      2    |
| MUTAG          |     188    |        17.93      |       19.79       |      2    |
| Mutagenicity   |    4,337   |        29.76      |       30.76       |      2    |
| PTC-FM         |     349    |        14.11      |       14.48       |      2    |
| PROTEINS       |    1,113   |        39.05      |       72.81       |      2    |
| IMDB-BINARY    |    1,000   |        19.77      |       96.53       |      2    |
| Jan. G.        |     744    |        174.25     |       497.35      |      2    |
| Feb. G.        |     648    |        175.28     |       502.96      |      2    |
| Mar. G.        |     744    |        174.43     |       480.33      |      2    |  
| Jan. Y.        |     744    |        203.04     |       1865.66     |      2    |
| Feb. Y.        |     648    |        199.28     |       1868.28     |      2    |
| Mar. Y.        |     744    |        207.24     |       1967.59     |      2    |

## Code
** The code related to each GNN architecture is available in each corresponding folder as follows:
```
Code
  |__eigengcn
  |__sag
  |__sage+gat+diffpool
```
The commands to run to train each GNN architecture on the 3 settings (original, 2stg, 2stg+) are described in the 'run_examples.txt' file in the respective folder.




