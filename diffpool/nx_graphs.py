import networkx as nx
import pickle
import numpy as np

# directory of dataset here
#f = open("darpa/darpa_hourly.txt", 'r')
#dataset = 'darpa'

dataset = "yellow_1904"
f = open("taxi/" + dataset + ".txt", 'r')
time = ''

# IMPORTANT: SPECIFY THE TYPE OF GRAPH HERE
# 1 for undirected-unweighted, 2 for directed-unweighted, 3 for directed-weighted
category = 3

# MAX NODE ID
max_id = 10000

graphs = []

if category == 1:  # undirected graph
    g = nx.Graph()
else:  # directed graph
    g = nx.DiGraph()

zeros = 0
ones = 0

# this is for the darpa dataset only
attackers = dict()

for line in f:
    lines = line.split(',')
    src, dst, timestamp, type = int(lines[0]), int(lines[1]), lines[2], int(lines[3])

    # new timestamp means new graph
    if timestamp != time:
        if time != '':
            if g.graph['label'] > 0:
                g.graph['label'] = 1
            graphs.append(g)

            if g.graph['label'] == 0:
                zeros += 1
            else:
                ones += 1

        # reset the time and the networkx
        time = timestamp

        if category == 1:
            g = nx.Graph()
        else:
            g = nx.DiGraph()

        g.graph['label'] = 0
        g.graph['type'] = 0

    
    if src < max_id and dst < max_id:
        if category == 1 or category == 2:   # not consider weights
            g.add_edge(src, dst)

        else: # consider weights
            if g.has_edge(src, dst):
                g[src][dst]['weight'] += 1
            else:
                g.add_edge(src, dst, weight=1)


        if type == 1 or type == 2 or type == 3 or type == 4:
            g.graph['label'] += 0
        else:
            g.graph['label'] += 1
            if not type in attackers.keys():
                attacker_id = len(attackers) + 1
                attackers[type] = attacker_id
            else:
                attacker_id = attackers[type]
           
        
            g.graph['type'] = attacker_id

# append the last graph
if g.graph['label'] > 0:
    g.graph['label'] = 1
    ones += 1
else:
    zeros += 1
graphs.append(g)

print(len(graphs))

if category == 1:
    with open("taxi/" + dataset + "_hourly_undir_unweight.pkl", 'wb') as h:
        pickle.dump(graphs, h)
elif category == 2:
    with open("taxi/" + dataset + "_hourly_dir_unweight.pkl", 'wb') as h:
        pickle.dump(graphs, h)
else:
    with open("taxi/" + dataset + "_hourly_dir_weight.pkl", 'wb') as h:
        pickle.dump(graphs, h)

print("normal: " + str(zeros))
print("abnormal: " + str(ones))



