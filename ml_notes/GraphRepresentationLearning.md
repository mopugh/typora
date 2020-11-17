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

### Graph Statistics and Kernel Methods

#### Node-level statistics and features

Motivating example: 

![](/home/mopugh/Documents/typora/ml_notes/figures/2020-11-16-20-46-25-image.png)

Motivating question: what features or statistics could a machine learning model use to predict the Medici's rise?

* What are useful properties and statistics that can be used to characterize the nodes in the graph?

##### Node degree

* Simplest node feature is the **degree**: the number of edges incident to a node
  
  $$
  d_{u} = \sum_{v \in V} A[u,v]
  $$

* For directed and weighted graphs, can differentiate between incoming and outgoing edges

* For the motivating example, the Medici family has the largest degree

##### Node centrality

* Node degree isn't sufficient to measure the *importance* of a node

* **Eigenvector centrality**: take into account how important the neighbors are and not just the number of neighbors.
  
  * Define as a recurrence relation: proportional to the average centrality of its neighbors
    
    $$
    e_{u} = \frac{1}{\lambda}\sum_{u\ in V}A[u,v]e_{v} \forall u \in \mathcal{V}
    $$
    
    where $\lambda$ is a constant.
  
  * Rewriting in vector notation:
    
    $$
    \lambda \mathbf{e} = A\mathbf{e}
    $$
    
    i.e. eigvenvector of the adjacency matrix
    
    * By the Perron-Frobenius Theorem, $\mathbf{e}$ is given by the eigenvector corresponding to the largest eigenvalue of $\mathbf{A}$ 
    
    * Can interpret as the likelihood that a node is visited on a random walk of infinite length on the graph.
  
  * In the motivating example, the Medici family has the highest eigenvector centrality

* Other notions of node centrality
  
  * **betweenness centrality**: how often a node lies on the shortest path between two other nodes
  
  * **closeness centrality**: the average shortest path length between a node and all other nodes

##### The clustering coefficient

Note in the motivating example, the Peruzzi and Guadagni families have similar degrees and eigenvector centralities, but their roles in the graph appear different.

* The **clustering coefficient** measures the proportion of closed triangles in a node's local neighborhood.

* The **local variant** of the clustering coefficient:
  
  $$
  c_{u} = \frac{\vert (v_{1}, v_{2}) \in \mathcal{E} : v_{1},v_{2} \in \mathcal{N}(u)}{{d_{u} \choose 2}}
  $$
  
  * The numerator calculates the number of edges between neighbors of $u$.
  
  * The denominator computes the number of pairs of nodes in $u$'s neighborhood.

* Measures how tightly clustered a node's neighborhood is
  
  *  A cluster coefficient of 1 means that all of $u$'s neighors are also neighbors of each other.

##### Closed triangles, ego graphs, and motifs

Can interpret the clustering coefficient as counting the number of closed triangles within each node's local neighborhood:

* the ratio between the number of triangles and the total possible number of triangles within a node's **ego graph**
  
  * ego graph: the subgraph containing the node, its neighbors and all the edges between nodes in its neighborhood.

* Can consider general **motifs** or **graphlets**, e.g. cycles of a particular lengh, and then count how often these motifs occur in a node's ego graph.

* Ego graphs transform computing node-level statistics and features to a graph-level task.

#### Graph-level features and graph kernels


