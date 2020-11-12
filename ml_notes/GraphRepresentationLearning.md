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

