import networkx as nx
import numpy as np
import tensorflow as tf
import math

class DBLPDataLoader:
    def __init__(self, graph_file):
        self.g = nx.read_edgelist(graph_file, create_using=nx.Graph(), nodetype=None, data=[('weight', int)])
        self.num_of_nodes = self.g.number_of_nodes()
        self.num_of_edges = self.g.number_of_edges()
        self.edges_raw = self.g.edges()
        self.nodes_raw = self.g.nodes()

        self.embedding = []
        self.neg_nodeset = []
        self.node_index = {}
        self.node_index_reversed = {}
        for index, node in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v in self.edges_raw]

    def deepwalk_walk(self, walk_length):
        start_node = np.random.choice(self.nodes_raw)
        walk = [start_node]
        walk_index = [self.node_index[start_node]]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.g.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk_node = np.random.choice(cur_nbrs)
                walk.append(walk_node)
                walk_index.append(self.node_index[walk_node])
            else:
                break
        return walk_index

    def fetch_batch(self, embedding, lu=0.1, batch_size=8, K=3, window_size=2, walk_length=8):
        self.embedding = embedding
        self.lu = lu
        u_i = []
        u_j = []
        label = []
        embedding_dim = embedding.shape[1]
        for i in range(batch_size):
            self.walk_index = self.deepwalk_walk(walk_length)
            for index, node in enumerate(self.walk_index):
                for n in range(max(index-window_size, 0), min(index+window_size+1, walk_length)):
                    if n != index:
                        u_i.append(node)
                        u_j.append(self.walk_index[n])
                        label.append(1.)

                self.neg_nodeset = []
                u_one_hot = np.zeros(self.num_of_nodes)
                u_one_hot[node] = 1
                u_i_embedding = np.matmul(u_one_hot, self.embedding)
                for node_neg in self.node_index.values():
                    if node_neg not in self.walk_index:
                        self.neg_nodeset.append(node_neg)

                neg_one_hot = np.zeros((len(self.neg_nodeset), self.num_of_nodes))
                for b in range(len(self.neg_nodeset)):
                    neg_one_hot[b][self.neg_nodeset[b]] = 1

                negnode_embedding = np.matmul(neg_one_hot, self.embedding)
                node_negative_distribution = np.exp(np.sum(u_i_embedding * negnode_embedding, axis=1)/embedding_dim)

                node_negative_distribution /= np.sum(node_negative_distribution)

                node_negative_distribution[node_negative_distribution > lu] = 0
                node_negative_distribution /= np.sum(node_negative_distribution)

                for c in range(K):
                    negative_node = np.random.choice(self.neg_nodeset, p=node_negative_distribution)
                    u_i.append(node)
                    u_j.append(negative_node)
                    label.append(-1.)
        
        return u_i, u_j, label

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node in self.nodes_raw}


if __name__ == '__main__':
    graph_file = './data/cora/cora_edgelist.txt'
    data_loader = DBLPDataLoader(graph_file=graph_file)
    a = np.random.rand(data_loader.num_of_nodes, 100)
    u_i, u_j, label = data_loader.fetch_batch(a,)
    print(u_i)
    print('\n---------------\n')
    print(u_j)
    print('\n---------------\n')
    print(label)





