# Notes on Graph Neural Networks

## Change Log

* 9/30/20: Initial notes. Completed upto and including Section 1.2

## Graph Representation Learning

### Chapter 1: Introduction

* In the most general view, graphs are simply a collection of objects (nodes) along with a set of interactions (edges) between pairs of objects
  * E.g. Individuals (nodes) and friendships (edges) in a social network
* Power of graph formalism is
  * focus on relationships between points (rather than properties of individual points)
  * generality

#### What is a graph?

* A graph $\mathcal{G} = (\mathcal{V, E})$ is a set of nodes $\mathcal{V}$ and a set of edges $\mathcal{E}$ between the nodes
  * Denote an edge going from node $u \in \mathcal{V}$ to node $v \in \mathcal{V}$ as $(u, v) \in \mathcal{E}$. 
  * A *simple graph* is a graph where
    * there is at most one edge between any pair of nodes
    * no edges between a node and itself
    * all edges are undirected, i.e. $(u, v) \in \mathcal{E} \leftrightarrow (v, u) \in \mathcal{E}$ 
* *Adjacency matrix* $A \in \mathbb{R}^{\vert \mathcal{V} \vert \times \vert \mathcal{V} \vert}$ such that
  * $A[u,v] = 1$ if $(u,v) \in \mathcal{V}$ and $A[u,v]=0$ otherwise
  * Requires ordering and labeling of nodes
  * If edges are undirected, then $A$ is symmetric
  * Generally, edges can have weights in which case elements of $A$ need not be restricted to $\left \{0,1 \right \}$. 

##### Multi-relational Graphs

* Can consider undirected, directed, weighted edges and *types* of edges. 
  * E.g. in a graph representing drug-drug interactions, might want edges to correspond to different side effects that can occur when you take a pair of drugs at the same time.
* For types of edges, extend notion of an edge to include the *type* $\tau$: $(u, \tau, v) \in \mathcal{E}$ 
  * Can create an adjacency matrix per type: $A_{\tau}$ 
  * Can "stack" these adjacency matrices per type to get an adjacency tensor $\mathcal{A} \in \mathbb{R}^{\vert \mathcal{V} \vert \times \vert \mathcal{R} \vert \times \vert \mathcal{V} \vert}$ where $\mathcal{R}$ is the set of relations
    * such a graph is called a *multi-relational* graph
* Two important subsets of multi-relational graphs are *heterogeneous* graphs and *multiplex* graphs
  * **Heterogeneous Graphs**: Nodes have types
    * Can partition nodes into disjoint sets: $\mathcal{V} = \mathcal{V}_{1} \cup \ldots \cup \mathcal{V}_{k}$ where $\mathcal{V}_{i} \cap \mathcal{V}_{j} = \emptyset, \forall i \neq j$ 
    * Edges generally satisfy constraints according to the node types, e.g. certain edges only connect nodes of certain types, i.e. $(u, \tau_{i}, v) \in \mathcal{E} \rightarrow u \in \mathcal{V}_{j}, v \in \mathcal{V}_{k}$ 
    * *Multipartitite graphs* have eges that can only connect nodes that have different types, i.e. $(u, \tau_{i}, v) \in \mathcal{E} \rightarrow u \in \mathcal{V}_{j}, v \in \mathcal{V}_{k} \and j \neq k$
  * **Multiplex Graphs**: Graph can be decomposed into $k$ layers.
    * Every node belongs to every layer
    * Each layer corresponds to a unique relation representing the *interlayer* edge type for that layer. 
    * *inter-layer* edges connect the same node across layers. 
      * E.g. transportation network where nodes are cities
        * Intra-layer edges represent cities that are connected by different modes of transportation
        * Inter-layer edges represent switching modes of transportation within a city.

##### Feature Information 

* *Attribute* or *feature* information associated with the graph. 
* Most often node-level features
  * represent with a matrix $X \in \mathbb{R}^{\vert \mathcal{V} \vert \times m}$ where there are $m$ attributes and assumed to use same node indexing as the adjacency matrix.
  * In heterogeneous graphs generally assume each type of node has its own distinct type of attributes.