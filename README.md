# MaxMinDP_GA

## Description

This code is aimed at calculating the k subgraph that maximises the minimum sum of distances between nodes in the subgraph.

## How to use

The code is written in the Julia language and as such you need to install Julia to run it.

The dependencies are listed in the `install.jl` file, so in order to be able to run the code you have to install them, by simply running the following command:

```bash
julia install.jl
```

After installing the dependencies, your main entry point is the `alg_tester_cli.jl` file. This program has no parameters, all configuration is done by placing the graphs in a folder next to it called `mmdp_graphs`, and configuring the wide array of parameters within the file, in the marked span. 

Each configuration can have multiple values and in the end the algorithm will run for each version resulting from the Cartesian product of those.

After configuring the file you can run it with the command:

```bash
julia alg_tester_cli.jl
```

For each configuration it will create a folder in the `results` folder combining the different parameters into the folder name with the current date, with one file for each graph containing the nodes of the subgraph for each run, and one other file that compiles the results.

Different logs may be emmited into the `logs` folder.


## Graphs we used

The code only accepts complete undirected graphs in the weighted edge list format. In this format only the edges are present in the file, one edge per line in the following format:

```
node1 node2 weight
```

All files have to be transformed into this format in order to proceed. Files in this format have to have the extension `.dat` in order to be discovered by the application.

Please note that these graphs have to be transformed in order to be useable with the code. 

* https://github.com/optsicom-hub/MDPLIB-2.0/tree/main
* http://www.di.unito.it/~aringhie/benchmarks.html