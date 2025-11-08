import pickle

with open('subgraph1.pickle','rb') as file:
    dict_get=pickle.load(file)

besttestacc=dict_get['besttestacc']
bestevaltacc=dict_get['bestevaltacc']
