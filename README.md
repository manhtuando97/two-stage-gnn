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
| :-------:| :----------:| :--------: | :---------: | :------------: |
| DD       |    1,168    |     269    |     676     |  [Link](https://www.dropbox.com/s/91xe1ixpqdm36c5/DD.zip?dl=0)              |
| MUTAG    |     188     |     18     |     20      |  [Link](https://www.dropbox.com/s/jgpolw5m0yrpbus/MUTAG.zip?dl=0)           |
| MUTAG2   |    4,337    |     30     |     31      |  [Link](https://www.dropbox.com/s/6mqvuktfwma4azu/MUTAG2.zip?dl=0)          |
| PTC-FM   |     349     |     14     |     14      |  [Link](https://www.dropbox.com/s/3qtkwstgk5859pz/PTC_FM.zip?dl=0)          |
| PROTEINS |    1,113    |     39     |     73      |  [Link](https://www.dropbox.com/s/rh3iuqpup9u9two/PROTEINS.zip?dl=0)        |
| IMDB-B   |    1,000    |     20     |     97      |  [Link](https://www.dropbox.com/s/wc9bvyfhroiv17c/IMDB_BINARY.zip?dl=0)     |
| JAN. G.  |     744     |     174    |     497     |  [Link](https://www.dropbox.com/s/ail6tkguc7zu1v4/Jan%20G.zip?dl=0)         |
| FEB. G.  |     648     |     175    |     503     |  [Link](https://www.dropbox.com/s/l7p99f8jlfybff4/Feb%20G.zip?dl=0)         |
| MAR. G.  |     744     |     174    |     480     |  [Link](https://www.dropbox.com/s/q9ely9ntvs7b01t/Mar%20G.zip?dl=0)         |
| JAN. Y.  |     744     |     203    |     1,866    | [Link](https://www.dropbox.com/s/0yr2ri9pcwnle1q/Jan%20Y.zip?dl=0)         |
| FEB. Y.  |     648     |     199    |     1,868    | [Link](https://www.dropbox.com/s/2ncfvcdbfct0shf/Feb%20Y.zip?dl=0)         |
| MAR. Y.  |     744     |     207    |     1,968    | [Link](https://www.dropbox.com/s/cxxr5ajkiiwhbnr/Mar%20Y.zip?dl=0)         |


## License
The source code in this directory is licensed under the MIT license, which can be found in the LICENSE file.
