import networkx as nx
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import scipy as sp
import scipy.optimize

def F_cost(pos_vec, np, invdist, meanweight):
    
    nNodes = invdist.shape[0]
    #приводим значения переданные в виде массива в прежний вид
    pos_arr = pos_vec.reshape((nNodes,2))
    #получаем расстояния от каждой вершыны до каждой из врешин в проекциях на оси x,y(думаю их можно рассматривать как векктора в пространстве)
    p1=pos_arr[:, np.newaxis, :]
    p2=pos_arr[np.newaxis, :, :]
    delta = p1 - p2
    #находим модули наших векторов, получаем матрицу расстояний между вершнами 
    nodesep = np.linalg.norm(delta, axis=-1)

    direction = np.einsum("ijk,ij->ijk", delta, 1 / (nodesep + np.eye(nNodes)))

    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0
    #расчитываем сумарную энергию всей системы
    cost = 0.5 * np.sum(offset ** 2)

    grad = np.einsum("ij,ij,ijk->ik", invdist, offset, direction) - np.einsum("ij,ij,ijk->jk", invdist, offset, direction)

    # прибавляем усреднённое значение,чтоб положение вершин было близко к началу координат
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos ** 2)
    grad += meanweight * sumpos
    #print(cost)
    print(grad)
    return (cost, grad.ravel())

def layout_k_k(dist_mtx, pos_arr):
    
    meanwt = 1e-3
    inv_dist=1 / (dist_mtx + np.eye(dist_mtx.shape[0])*1e-3)
    costargs = (np, inv_dist, meanwt)
    optresult = sp.optimize.minimize(
        F_cost,
        pos_arr.ravel(),
        method="L-BFGS-B",
        args=costargs,
        jac=True,
    )
     
    return optresult.x.reshape((-1, 2))

def kamada_kawai_right_version(G):
    
    center = np.zeros(2)
    nNodes = len(G)
    #проверка на наличае вершин в графе
    if nNodes == 0:
        return {}
    #поиск кратчайших путей между вершинами
    dist = dict(nx.shortest_path_length(G))
   
    dist_mtx = np.ones((nNodes, nNodes))
    #создаём матрицу в клетках которой записаны кратчайшие пути между вершинами
    for row, nr in enumerate(G):
        if nr not in dist:
            continue
        rdist = dist[nr]
        for col, nc in enumerate(G):
            if nc not in rdist:
                continue
            dist_mtx[row][col] = rdist[nc]
    
   
    #зхадаем изначальные позиции вершин
    pos = nx.circular_layout(G)
    pos_arr = np.array([pos[n] for n in G])
    #расчет раскладки
    pos = layout_k_k(dist_mtx, pos_arr)
    return dict(zip(G, pos))
#раскраска
def create_colors(G):
    ids = list(G.nodes)
    print(ids)
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



DataPathNodes="HU_genres.json"
DataPathEdges="HU_edges.csv"


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

pos=kamada_kawai(G)
print(pos)
print(5)
nx.draw_networkx_edges(G,
       pos,
    edge_color="grey")
nx.draw_networkx_nodes(G,
      pos,
      node_size=500,
     node_color=colors)

plt.savefig('graph.png')
