from __future__ import division
from __future__ import print_function

import numpy as np
from random import shuffle, choice


class TripletSampler(object):
    """
    This sampler sample one triplet (anchor, pos, neg) for each epoch, each graph in Graphs be the anchor once.
    pos is randomly selected from other graphs of the same attacker with the anchor
    neg is randomly selected from graphs of another attacker.

    Graphs: a dict of graphs with the attacker name as the keys.

    """

    def __init__(self, Graphs, **kwargs):
        self.graphs = dict()
        self.graphs[0] = []
        self.graphs[1] = []
        for g in Graphs.G_list:
            if(int(g.graph['label']) == 0):
                self.graphs[0].append(g)
            else:
                self.graphs[1].append(g)
        self.attackers = [0, 1]  # a list of unique attackers
        # a list of attackers which is repeated by the number of graphs of this attacker
        self.attacker_list = [0, 1]

        # iter for iterating anchor over all attacker
        self.attacker_iter = 0      # iterator for all attackers
        self.graph_iter = 0         # iterator for all graphs
        self.neg_iter = 0           # iterator for all negative attackers, with respect to the positive attacker

    def sampler(self, embed=None, hardness=0):
        sampled_Gs = {}
        pos_attacker = self.attacker_list[self.attacker_iter]
        neg_attacker = self.sample_neg_attacker(pos_attacker)
        # print('positive attacker: {}, negative attacker: {}'.format(pos_attacker, neg_attacker))

        num_G_pos = len(self.graphs[pos_attacker])
        num_G_neg = len(self.graphs[neg_attacker])
        anchor = self.graphs[pos_attacker][self.graph_iter]
        sampled_Gs['anchor'] = anchor

        pos_idx = self.graph_iter
        while pos_idx == self.graph_iter:
            # pos graph should be different from the anchor graph
            pos_idx = np.random.choice(num_G_pos, 1).item()
            # print('sampled pos index ', pos_idx)
        sampled_Gs['pos'] = self.graphs[pos_attacker][pos_idx]

        neg_idx = np.random.choice(num_G_neg, 1).item()
        sampled_Gs['neg'] = self.graphs[neg_attacker][neg_idx]
        sampled_Gs['label'] = pos_attacker

        ## sample the negative graph semi-hard or hardest
        if hardness > 0: 
            anchor_feat = embed[pos_attacker][self.graph_iter]
            # sample the semi-hard negative graph
            if hardness == 0.5:
                pos_feat = embed[pos_attacker][pos_idx]
                pos_distance = np.linalg.norm(anchor_feat - pos_feat)
                for neg in range(len(self.graphs[neg_attacker])):
                    neg_feat = embed[neg_attacker][neg]
                    neg_distance = np.linalg.norm(anchor_feat - neg_feat)
                    if neg_distance < pos_distance:
                        sampled_Gs['neg'] = self.graphs[neg_attacker][neg]
                        break

            # sample the hardest negative graph
            else:
                
                neg_feat = embed[neg_attacker][neg_idx]
                
                distance = np.linalg.norm(anchor_feat - neg_feat)
                for neg in range(len(self.graphs[neg_attacker])):
                    neg_feat = embed[neg_attacker][neg]
                    neg_distance = np.linalg.norm(anchor_feat - neg_feat)
                    
                    if neg_distance < distance:
                        distance = neg_distance
                        sampled_Gs['neg'] = self.graphs[neg_attacker][neg]


        self.attacker_iter += 1
        if self.graph_iter == len(self.graphs[pos_attacker]) - 1:
            # finished the iteration of one attacker, starting the iteration of the next attacker
            self.graph_iter = 0
        else:
            self.graph_iter += 1
        return sampled_Gs


    def exhaust_sample(self, embed=None, hardness=0):
        sampled_Gs = {}

        pos_attacker = self.attacker_list[self.attacker_iter]
        neg_attacker = self.sample_neg_attacker(pos_attacker)
        # print('positive attacker: {}, negative attacker: {}'.format(pos_attacker, neg_attacker))

        self.neg_iter = (self.neg_iter + 1) % 2
        
        num_G_pos = len(self.graphs[pos_attacker])
        num_G_neg = len(self.graphs[neg_attacker])
        anchor = self.graphs[pos_attacker][self.graph_iter]
        sampled_Gs['anchor'] = anchor

        pos_idx = self.graph_iter
        while pos_idx == self.graph_iter:
            pos_idx = np.random.choice(num_G_pos, 1).item()            
        sampled_Gs['pos'] = self.graphs[pos_attacker][pos_idx]

        neg_idx = np.random.choice(num_G_neg, 1).item()
        
        sampled_Gs['label'] = pos_attacker

        ## sample the negative graph semi-hard or hardest
        if hardness > 0: 
            anchor_feat = embed[pos_attacker][self.graph_iter]
            # sample the semi-hard negative graph
            if hardness == 0.5:
                pos_feat = embed[pos_attacker][pos_idx]
                pos_distance = np.linalg.norm(anchor_feat - pos_feat)
                for neg in range(len(self.graphs[neg_attacker])):
                    neg_feat = embed[neg_attacker][neg]
                    neg_distance = np.linalg.norm(anchor_feat - neg_feat)
                    if neg_distance < pos_distance:
                        sampled_Gs['neg'] = self.graphs[neg_attacker][neg]
                        break

            # sample the hardest negative graph
            else:
                
                neg_feat = embed[neg_attacker][neg_idx]
                
                distance = np.linalg.norm(anchor_feat - neg_feat)
                for neg in range(len(self.graphs[neg_attacker])):
                    neg_feat = embed[neg_attacker][neg]
                    neg_distance = np.linalg.norm(anchor_feat - neg_feat)
                    
                    if neg_distance < distance:
                        distance = neg_distance
                        sampled_Gs['neg'] = self.graphs[neg_attacker][neg]
            
        
        if self.neg_graph_iter == len(self.graphs[neg_attacker]) - 1:
            self.neg_graph_iter = 0
            self.graph_iter += 1
        else:
            self.neg_graph_iter += 1
                            
        self.attacker_iter += 1                

        if self.graph_iter == len(self.graphs[pos_attacker]):
            # finished the iteration of one attacker, starting the iteration of the next attacker
            self.graph_iter = 0
                    
        return sampled_Gs

    # sample a triplet of 3 equal graphs, for save embedding in test phase of leave-one-out method (in which test set only contains 1 graph)
    def sampler_leaveOneOut(self):

        sampled_Gs = {}
        pos_attacker = self.attacker_list[0]
        neg_attacker = self.attacker_list[0]


        sampled_Gs['anchor'] = self.graphs[pos_attacker]
        sampled_Gs['pos'] = self.graphs[pos_attacker]
        sampled_Gs['neg'] = self.graphs[neg_attacker]


        sampled_Gs['label'] = pos_attacker
        return sampled_Gs


    def sample_neg_attacker_exhaust(self, pos_attacker):
        neg_attackers = list(self.graphs.keys())
        neg_attackers.remove(pos_attacker)

        return neg_attackers[self.neg_iter]

    def sample_neg_attacker(self, pos_attacker):
        neg_attackers = list(self.graphs.keys())
        neg_attackers.remove(pos_attacker)

        # sample one negative attacker
        return choice(neg_attackers)

    def end(self):
        """
        Check if one time sample ends
        """
        return self.attacker_iter >= len(self.attacker_list)

    def shuffle(self):
        """
        Re-shuffle the graph list.
        Also reset iter for sampling the anchor graph.
        """
        for attacker in self.attackers:
            shuffle(self.graphs[attacker])
        self.attacker_iter = 0
        self.graph_iter = 0