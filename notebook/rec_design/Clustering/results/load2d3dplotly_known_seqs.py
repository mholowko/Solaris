import plotly
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
import argparse
from HSN_util import *
from plotly.validators.scatter.marker import SymbolValidator

marker_symbols = SymbolValidator()

parser = argparse.ArgumentParser(description='Show 3d scatter plot')
parser.add_argument('npz_path', help = "the path to the npz file which contains both label and coordinates")

args = parser.parse_args()

data = np.load(args.npz_path, allow_pickle = True)

embed = data['coord']
# y = data['label']
text = data['text']
y_km = data['ykm']
known_seq = data['known_seq']
ucb_rec = data['ucb_rec']

print(len(y_km))
# docs = data['docs']

tir_labels = []
text_labels = []

for i in text:
    tir_labels.append(float(i[-1]))
    text_labels.append(str(i))


print(args.npz_path[:-4])

# l = np.array(['Harm', 'Harmless', 'Impurity','InternationalConspiracy', 'LiteralHigh','LiteralLow','MetaphoricalHigh', 'MetaphoricalLow', 'NonHuman', 'Threat', 'Threat1', 'Threat2'])

idxes = np.asarray(range(len(y_km)))

trace_list = []

'''
for i in range(len(set(y_km))):
    known_seq_k = []
    for k in idxes[y_km==i]:
        known_seq_k.append(k)
    known_seq_k = np.asarray(known_seq_k)
    print(np.asarray(text_labels)[known_seq_k])
    trace_list.append(go.Scatter(x = embed[known_seq_k,0], y = embed[known_seq_k,1], mode = 'markers', marker = dict(size = 12, symbol = i, line = dict(width = 0), color =  np.asarray(tir_labels)[known_seq_k], 
               colorbar=dict(title="Colorbar"),colorscale = "Viridis", opacity = 0.9), text = np.asarray(text_labels)[known_seq_k], name = str(i), hoverinfo='text'))
'''       
        
# trace_list.append(go.Scatter(x = embed[:,0], y = embed[:,1], mode = 'markers', marker = dict(size = 12, symbol = y_km[:], line = dict(width = 0), color =  tir_labels, 
#                colorbar=dict(title="Colorbar"),colorscale = "Viridis", opacity = 0.9), text = text_labels[:], hoverinfo='text'))

group_indicator = np.asarray([text[:,1] == 'bandit2']) + [0] # 1 indicates in group bandit2
group_indicator = group_indicator[0]
print(group_indicator)

def sigmoid(z_list):
    func_z_list = []
    for z in z_list:
        func_z_list.append(20.0/(1 + np.exp(-z)))
    return func_z_list
marker_size = sigmoid(tir_labels) 

trace_list.append(go.Scatter(x = embed[:,0], y = embed[:,1], mode = 'markers', marker = dict(size = marker_size, symbol = y_km[:], line = dict(width = 0), color = text[:,2],
                opacity = 0.6), text = text_labels[:], hoverinfo='text'))
    
    # trace_list.append(px.scatter(x = embed[known_seq,0], y = embed[known_seq,1],  color =  tir_labels, 
    #             color_continuous_scale = plotly.colors.sequential.Viridis, opacity = 0.9, text = text[known_seq]))

    # trace_pred = go.Scatter(x = embed[:,0], y = embed[:,1], mode = 'markers', marker = dict(size = 14, line = dict(width = 0), color = y_pred, colorscale = 'Viridis', opacity = 0.6), text = text, name = "Predict Label") 

layout = go.Layout(
    hovermode= 'closest',
    title = go.layout.Title(text = args.npz_path.split('/')[-1].split('.')[0].replace('_', ' ')),
    # showLegend = True
    # margin=dict(
    #     l=0,
    #     r=0,
    #     b=0,
    #     t=0
    # ),
    # legend=dict(x = -0.1, y = 1.2)
)
    
fig = go.Figure(data = trace_list, layout = layout)
# fig = go.Figure(data = [trace_0, trace_1, trace_2, trace_3, trace_4, trace_5, trace_6, trace_7, trace_8, trace_9], layout = layout)
fig.update_layout(title = args.npz_path)

plotly.offline.plot(fig, filename = args.npz_path[:-4] + "plot")


# dim = embed.shape[1]

# assert (dim == 2 or dim == 3), "invalid dim"

# print(args.npz_path[:-4])

# doc = np.unique(docs)
# doc_dict = {}


# for i in range(len(doc)):
#     doc_dict[doc[i]] = i

# doc_index = []

# for d in docs:
#     doc_index.append(doc_dict[d])
    
# y = np.array(doc_index)

# l = np.array(doc)


# if dim == 2:
#     trace_true = go.Scatter(x = embed[:,0], y = embed[:,1], mode = 'markers', marker = dict(size = 14, line = dict(width = 0), color = y, colorscale = 'Viridis', opacity = 0.6), text = text, name = "True Label", hoverinfo='text') 
#     # trace_pred = go.Scatter(x = embed[:,0], y = embed[:,1], mode = 'markers', marker = dict(size = 14, line = dict(width = 0), color = y_pred, colorscale = 'Viridis', opacity = 0.6), text = text, name = "Predict Label") 
# else:
#     # trace_true = go.Scatter3d(x = embed[:,0], y = embed[:,1], z = embed[:, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = y, colorscale = 'Viridis', opacity = 0.6), text = text, name = "True Label", hoverinfo='text') 
#     trace_0 = go.Scatter3d(x = embed[y==0,0], y = embed[y==0,1], z = embed[y==0, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 0], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==0], name = l[0], hoverinfo='text') 
#     trace_1 = go.Scatter3d(x = embed[y==1,0], y = embed[y==1,1], z = embed[y==1, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 1], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==1], name = l[1], hoverinfo='text') 
#     trace_2 = go.Scatter3d(x = embed[y==2,0], y = embed[y==2,1], z = embed[y==2, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 2], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==2], name = l[2], hoverinfo='text') 
#     trace_3 = go.Scatter3d(x = embed[y==3,0], y = embed[y==3,1], z = embed[y==3, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 3], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==3], name = l[3], hoverinfo='text') 
#     trace_4 = go.Scatter3d(x = embed[y==4,0], y = embed[y==4,1], z = embed[y==4, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 4], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==4], name = l[4], hoverinfo='text')
#     trace_5 = go.Scatter3d(x = embed[y==5,0], y = embed[y==5,1], z = embed[y==5, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 5], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==5], name = l[5], hoverinfo='text') 
#     trace_6 = go.Scatter3d(x = embed[y==6,0], y = embed[y==6,1], z = embed[y==6, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 6], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==6], name = l[6], hoverinfo='text')  
#     trace_7 = go.Scatter3d(x = embed[y==7,0], y = embed[y==7,1], z = embed[y==7, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 7], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==7], name = l[7], hoverinfo='text')  
#     trace_8 = go.Scatter3d(x = embed[y==8,0], y = embed[y==8,1], z = embed[y==8, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 8], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==8], name = l[8], hoverinfo='text')  
#     trace_9 = go.Scatter3d(x = embed[y==9,0], y = embed[y==9,1], z = embed[y==9, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 9], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==9], name = l[9], hoverinfo='text')  
#     trace_10 = go.Scatter3d(x = embed[y==10,0], y = embed[y==10,1], z = embed[y==10, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 10], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==10], name = l[10], hoverinfo='text')  
#     trace_11 = go.Scatter3d(x = embed[y==11,0], y = embed[y==11,1], z = embed[y==11, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 11], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==11], name = l[11], hoverinfo='text')  
#     trace_12 = go.Scatter3d(x = embed[y==12,0], y = embed[y==12,1], z = embed[y==12, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 12], np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==12], name = l[12], hoverinfo='text')  

# layout = go.Layout(
#     hovermode= 'closest',
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0
#     ),
#     legend=dict(x = -0.1, y = 1.2)
# )
    
# # fig = go.Figure(data = [trace_true, trace_pred], layout = layout)
# fig = go.Figure(data = [trace_0, trace_1, trace_2, trace_3, trace_4, trace_5, trace_6, trace_7, trace_8, trace_9, trace_10, trace_11, trace_12], layout = layout)

# plotly.offline.plot(fig, filename = args.npz_path[:-4] + "plot")