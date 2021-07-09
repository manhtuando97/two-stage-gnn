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
All datasets used in this work are available in this directory: https://bit.ly/36zCH9b 

## License
The source code in this directory is licensed under the MIT license.

Copyright (c) Manh Tuan Do et al.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



