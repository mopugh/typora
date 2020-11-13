# Graph Representation Learning

## Chapter 1: Introduction

* General view: a graph is a collection of objects (i.e nodes) along with a set of interaction (i.e. edges) between pairs of objects.
  * Example: social network where nodes are people and edges are between friends
* Power of graph formalism lies in its focus on 
  * relationship between points (rather than properties of individual points)
  * generality

### What is a graph?

* A graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is defined by a set of node $\mathcal{V}$ and a set of edges $\mathcal{E}$ between these nodes. 
  * An edge going from node $u \in \mathcal{V}$ to node $v \in \mathcal{V}$ is denoted $(u,v) \in \mathcal{E}$ 
* **Simple graphs**
  * at most one edge between each pair of nodes
  * no edges between a node and itself
  * all edges are undirected, i.e. $(u,v) \in \mathcal{E} \Leftrightarrow (v,u) \in \mathcal{E}$
* Adjacency matrix $\mathbf{A} \in \mathbb{R}^{\vert \mathcal{V} \vert \times \vert \mathcal{V} \vert}$ encodes connections
  * number all nodes and $\mathbf{A}[i,j] = 1$ if $(i,j) \in \mathcal{E}$ and 0 otherwise
  * if the edges are undirected, then $\mathbf{A}$ is symmetric
  * if the edges are directed, then $\mathbf{A}$ need not be symmetric
  * if edges are weighted, then the entires of $\mathbf{A}$ need not be $\{0, 1\}$

#### Multi-relational Graphs

* Edges, in addition to be directed, undirected or weighted, can also have various types
  * e.g. types of interactions in drug-drug interactions
* Extend edge notation to include an edge of relation type $\tau$ : $(u, \tau, v) \in \mathcal{E}$
  * Can then define one adjacency matrix $A_{\tau}$ per edge type 
* These graphs are **multi-relational** 
  * Can encode with an adjacency tensor $\mathcal{A} \in \mathbb{R}^{\vert \mathcal{V} \vert \times \vert \mathcal{R} \vert \times \vert \mathcal{V} \vert}$ 
    * $\mathcal{R}$ is the set of relations.
  * Two types of multi-relational graphs:
    * heterogeneous graphs
    * multiplex graphs

##### Heterogeneous graphs

* nodes in heterogeneous graphs also have types
  * $\mathcal{V} = \mathcal{V}_{1} \cup \mathcal{V}_{2} \cup \ldots \cup \mathcal{V}_{k}$ where $\mathcal{V}_{i} \cap \mathcal{V}_{j} = \empty, \forall i \neq j$
    * i.e. the node types form a partition
* edges generally satisfy certain constraints acoocrding to the node types
  * most commonly certain edges only connect nodes of certain types
    * e.g. $(u, \tau_{i}, v) \in \mathcal{E} \rightarrow u \in \mathcal{V}_{j}, v \in \mathcal{V}_{k}$
* **multipartite graphs** are a type of heterogeneous graph where edges can only connect nodes that have different types
  * i.e. $(u, \tau_{i}, v) \in \mathcal{E} \rightarrow u \in \mathcal{V}_{j}, v \in \mathcal{V}_{k} \and j \neq k$ 

##### Multiplex graphs

* In multiplex graphs we assume that the graph can be decomposed into k layers. 
  * every node is assumed to belong to every layer
  * each layer corresponds to a unique relation representing the **intra-layer** edge type for that layer
* **inter-layer** edge types exist which connect the same node across layers.
* Example: multiplex transportation network
  * each city represents a city
  * each layer represents a different mode of transportation (e.g. air travel, train travel)
  * intra-layer edges represented cities connected by different modes of transportation
  * inter-layer edges represent the possibility of switching modes of transportation within a particular city

#### Feature Information

* **attribute** or **feature** information associated with a graph
  
  * E.g. a profile picture associated with a user in a social network

* Represent node-level attributes as a matrix $\mathbf{X} \in \mathbb{R}^{\vert \mathcal{V} \vert\times m}$ where the node ordering is the same as in the adjacency matrix.

### Machine learning on graphs

* Categorize machine learning model on the type of task they seek to solve
  
  * supervised: predict target output given input data point
  
  * unsupervised: intfer patterns, such as clusters of points, in the data

* This categorization isn't the best for machine learning with graphs

#### Node Classification

* Example: given a social network, identify bots

* Goal: predict the label $y_{u}$ (which could be a type, category, attribute) associated with nodes $u \in \mathcal{V}$ when given the true labels on a **training set** of nodes $\mathcal{V}_{train} \subset \mathcal{V}$ 

* Note: nodes in a graph are **not** i.i.d. as opposed to standard supervised learning problems

#### Relation prediction

**Relation prediction**: )a.k.a. relational inference, graph completion, or link prediction) has the goal of infering the edges between nodes in a graph.

* Standard setup: given all the nodes $\mathcal{V}$ and an incomplete set of edges $\mathcal{E}_{train} \subset \mathcal{E}$. 
  
  * Want to infer the missing edges $\mathcal{E} \setminus \mathcal{E}_{train}$

#### Clustering and community detection

* Node classification and relation prediction infer **missing** information and are analogs of supervised learning

* Community detection is the graph analog of unsupervised clustering.

* Given an input graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, infer latent community structures

#### Graph classification, regression, and clustering

* Perform classification, regression and clustering over entire graphs
  
  * E.g. Given a graph representing the structure of a molecule, build a regression model to predict the molecule's toxicity or solubility.

* Note: as opposed to making predictions over individual components of a single graph, for **graph regression** or **graph classification**, we are given a dataset of multiple different graphs and the goal is to make independent predictions specific to each graph.

* In **graph clustering**, the goal is to learn an unsupervised measure of similarity between pairs of graphs.

* Graph regression and classification are the closest analog of supervised learning because each graph is i.i.d. and we map from a graph data point to labels.

* Graph classification is the closest analog of unsupervised learning.

## Chapter 2: Background and Traditional Approaches




