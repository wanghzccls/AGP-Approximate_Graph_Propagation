#  Approximate Graph Propagation

Codes Contributors: Hanzhi Wang, Mingguo He
<br/>

## Citation
Please cite our paper when using the codes: 

```
@inproceedings{10.1145/3447548.3467243,
author = {Wang, Hanzhi and He, Mingguo and Wei, Zhewei and Wang, Sibo and Yuan, Ye and Du, Xiaoyong and Wen, Ji-Rong},
title = {Approximate Graph Propagation},
year = {2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
pages = {1686–1696},
location = {Virtual Event, Singapore},
series = {KDD '21}
}
```

<br/>

## I. Local Clustering with HKPR
### Tested Environment:
- Ubuntu 16.04.10
- C++ 11
- GCC 5.4.0


### Data:
* All of the datasets used in the paper are publicly available at: 
    * YouTube: https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz
    * Orkut: https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
    * Friendster: https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
    * Twitter: http://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010.graph

    Additionally, we provide a toy dataset at *AGP-master/clustering-HKPR/dataset/*. 

* Each dataset has a unique name called filelabel (e.g. *youtube*, *orkut*, *friendster* and *twitter*) . 
* Please rename the raw datasets as *${filelabel}.txt* (e.g. *youtube.txt*, *orkut.txt*, *friendster.txt*, and *twitter.txt*) and put them in the directory: *AGP-master/clustering-HKPR/dataset/*. 
* We assume that all raw datasets follow a consistent format: 
    * The number of nodes is explicitly specified in the first line of the data file. 
    * Each line following the second line indicates a **directed** edge in the graph. 
* We assume that all undirected graphs have been converted to directed graphs that each undirected edge appears twice in the data file. 
* We assume that the node index starts from $0$. The number of nodes is larger than the largest node index. 
* The code converts the raw data to binary file in CSR format when reading raw graphs for the first time. The converted binary data is stored in the directory: *AGP-master/clustering-HKPR/dataset/${filelabel}/*. 


### Query nodes:
* When the code is invoked for the first time, it will automatically construct a query file containing 100 query nodes.
* We name the query file as ${filelabel}.query and put it in the directory: *AGP-master/clustering-HKPR/query/*. 


### Execution:
We include the fundamental commands in the script file: *AGP-master/clustering-HKPR/run_script.sh*. To automatically execute our codes, please use the following bash commands: 
```
bash run_script.sh
```

Alternatively, our codes can be executed mannually. Specifically, to compile the codes: 
```
cd AGP-master/clustering-HKPR
rm HKPR
make
```
To run powermethod: 
```
./HKPR -d ./ -f youtube -algo powermethod -qn 10 -t 5
```
To run AGP: 
```
./HKPR -d ./ -f youtube -algo AGP -e 1e-07 -qn 10 -t 5
```

### Parameters:
- -d \<path of the "AGP-master" directory\> (default "./")
- -f \<filelabel\> (default youtube)
- -algo \<algorithm\> (default "QUERY")
- [-e \<epsilon\> (default 0.001)]
- [-qn \<querynum\> (default 10)]
- [-t \<the heat kernel parameter\> (default 5)]


### Remark:
* *AGP/clustering-HKPR/datatset/*: containing the datasets 
* *AGP/clustering-HKPR/query/*: containing the query files
* *AGP/clustering-HKPR/result/*: containing the approximation results. 

<br/>

## II. Node Classficiation with GNN
### Requirements
- CUDA  10.1
- python 3.8.5
- pytorch 1.7.1
- GCC 5.4.0
- cython 0.29.21
- eigency 1.77
- numpy 1.18.1
- torch-geometric 1.6.3 
- tqdm 4.56.0
- ogb 1.2.4
- [eigen 3.3.9] (https://gitlab.com/libeigen/eigen.git)

### Data
All of the datasets used in the paper are publicly available at:
* Reddit: https://github.com/GraphSAINT/GraphSAINT
* Yelp: https://github.com/GraphSAINT/GraphSAINT
* Amazon2M: https://github.com/google-research/google-research/tree/master/cluster_gcn
* Papers100M: https://ogb.stanford.edu



### Compilation
To compile the code, please run cython first, following  the commands shown as below. 
```
python setup.py build_ext --inplace
```


### Execution
* On Reddit dataset: 
```
sh reddit.sh
```
* On Yelp dataset: 
```
sh yelp.sh
```
* On Amazon2M dataset: 
```
sh amazon2M.sh
```
* On Papers100M dataset: 
```
sh papers100M.sh
```


