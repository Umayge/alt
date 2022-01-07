import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import scipy as sp
import scipy.optimize
import scipy.sparse
from networkx.utils import random_state

def fruchterman_reingold(G):
    center = np.zeros(2)
    pos_arr = None
    dom_size = 1
    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}
    A = nx.to_scipy_sparse_matrix(G, dtype="f")
    pos = la(A)
    pos = dict(zip(G, pos))
    return pos

@random_state(1)
def layout_fr_re(A, seed=None):

    nnodes = A.shape[0]
   
    # make sure we have a LIst of Lists representation

    A = A.tolil()

    # рандомная инициализация позиций
    pos = np.asarray(seed.rand(nnodes, 2), dtype=A.dtype)

    fixed = []

    # оптимальное расстояние между вершинами
    l = np.sqrt(1.0 / nnodes)
    # определяем максимальную "температуру".
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    #вычисляем значение на которое будет уменьшаться температура на каждом шаге итерации
    dt = t / float(100 + 1)

    displacement = np.zeros((2, nnodes))
    for iteration in range(100):
        displacement *= 0
        # цикл по строкам
        for i in range(A.shape[0]):
            if i in fixed:
                continue
            # разница между положением взятого узла и всеми остальными
            delta = (pos[i] - pos).T
            # находим растояния между вершинами
            distance = np.sqrt((delta ** 2).sum(axis=0))
            # если дистанция больше минимальной заменяем её на минимальную
            distance = np.where(distance < 0.01, 0.01, distance)
            # строка матрици смежности
            Ai = np.asarray(A.getrowview(i).toarray())
            # сила пружины
            displacement[:, i] +=(delta * (-Ai*(distance ** 2 /l)+ l**2/ distance)).sum(axis=1)
        # изменяем позиции вершин
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.01, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        # уменьшаем температуру
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < 1e-4:
            break
    return pos
  
  
#раскраска вершин
def create_colors(G):
    ids = list(G.nodes)
    
    colors = []

    r=nx.get_node_attributes(G,"Pop")
    b=nx.get_node_attributes(G,"Dance")
    g=nx.get_node_attributes(G,"Rap/Hip Hop")
    N=len(ids)
    z=0
    for r1,b1,g1 in zip(r,b,g):
        z+=1
        if int(r[r1]) and int(g[g1]) and int(b[b1]):
            color = "white"
        elif int(r[r1]) and int(b[b1]):
            color = "purple"
        elif int(r[r1]) and int(g[g1]):
            color = "yellow"
        elif int(b[b1]) and int(g[g1]):
            color = "cyan"
        elif int(r[r1]):
            color="red"
        elif int(b[b1]):
            color="blue"
        elif int(g[g1]):
            color ="green"
        else:
            color = "black"
        colors.append(color)
    
    carac = pd.DataFrame({ 'ID':ids, 'colors':colors })
    
    carac = carac.set_index('ID')
    
    carac = carac.reindex(G.nodes())
   
    carac['colors'] = pd.Categorical(carac['colors'])
 
    return carac['colors']

DataPathNodes="HR_genres.json"
DataPathEdges="HR_edges.csv"
file1=open(DataPathNodes)
Data=json.load(file1)
sorted(Data,key=lambda x:x[0])
file1.close()
G = nx.Graph()


DictGenre=["Acoustic Blues", "African Music", "Alternative", "Alternative Country", "Asian Music", "Baroque", "Bluegrass", 'Blues', 'Bolero', 'Bollywood', 'Brazilian Music', 'Chicago Blues', 'Chill Out/Trip-Hop/Lounge', 'Classic Blues', 'Classical', 'Classical Period', 'Comedy', 'Contemporary R&B', 'Contemporary Soul', 'Country', 'Country Blues', 'Dance', 'Dancefloor', 'Dancehall/Ragga', 'Delta Blues', 'Dirty South', 'Disco', 'Dub', 'Dubstep', 'Early Music', 'East Coast', 'Electric Blues', 'Electro', 'Electro Hip Hop', 'Electro Pop/Electro Rock', 'Film Scores', 'Films/Games', 'Folk', 'Game Scores', 'Grime', 'Hard Rock', 'Indian Music', 'Indie Pop', 'Indie Pop/Folk', 'Indie Rock', 'Indie Rock/Rock pop', 'Instrumental jazz', 'International Pop', 'Jazz', 'Jazz Hip Hop', 'Kids', 'Kids & Family', 'Latin Music', 'Metal', 'Modern', 'Musicals', 'Nursery Rhymes', 'Old School', 'Old school soul', 'Oldschool R&B', 'Opera', 'Pop', 'R&B', 'Ranchera', 'Rap/Hip Hop', 'Reggae', 'Rock', 'Rock & Roll/Rockabilly', 'Romantic', 'Singer & Songwriter', 'Ska', 'Soul & Funk', 'Soundtracks', 'Spirituality & Religion', 'Sports', 'TV Soundtracks', 'TV shows & movies', 'Techno/House', 'Traditional Country', 'Trance', 'Tropical', 'Urban Cowboy', 'Vocal jazz', 'West Coast']
d = {a: "0" for a in DictGenre}


for a,b1 in Data.items():
	buf=d.copy()
	for z in b1:
		buf[z]="1"
	G.add_nodes_from([(int(a),buf)])

df=pd.read_csv(DataPathEdges)

for i in range(df.shape[0]):
	ar=tuple(df.iloc[i].tolist())
	G.add_edge(*ar)

colors=create_colors(G)

fig=plt.figure(figsize=(240,220))
ax=fig.add_subplot(111)
ax.patch.set_facecolor('lightblue')

pos=spring_layout(G)

nx.draw_networkx_edges(G,
       pos,
    edge_color="black")
nx.draw_networkx_nodes(G,
      pos,
      node_size=550,
     node_color=colors)

plt.savefig('graph.png')
