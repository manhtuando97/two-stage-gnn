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
| Dataset  | #Graphs     | Avg. #Nodes| Avg. #Edges |    Download    |
| ---------| ------------| ---------- | ----------- | -------------- |
| DD       |    1,168    |     269    |     676     |                |
| MUTAG    |     188     |     18     |     20      |                |
| MUTAG2   |    4,337    |     30     |     31      |                |
| PTC-FM   |     349     |     14     |     14      |                |
| PROTEINS |    1,113    |     39     |     73      |                |
| IMDB-B   |    1,000    |     20     |     97      |                |
| JAN. G.  |     744     |     174    |     497     |                |
| FEB. G.  |     648     |     175    |     503     |                |
| MAR. G.  |     744     |     174    |     480     |                |
| JAN. Y.  |     744     |     203    |     1866    |                |
| FEB. Y.  |     648     |     199    |     1868    |                |
| MAR. Y.  |     744     |     207    |     1968    |                |


## License
The source code in this directory is licensed under the MIT license, which can be found in the LICENSE file.
