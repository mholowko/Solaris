import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import argparse
from HSN_util import *

parser = argparse.ArgumentParser(description='Show 3d scatter plot')
parser.add_argument('npz_path', help = "the path to the npz file which contains both label and coordinates")

args = parser.parse_args()

data = np.load(args.npz_path)

embed = data['coord']
# y = data['label']
text = data['text']
y_km = data['ykm']
print(y_km)
# docs = data['docs']


dim = embed.shape[1]

print(args.npz_path[:-4])

# l = np.array(['Harm', 'Harmless', 'Impurity','InternationalConspiracy', 'LiteralHigh','LiteralLow','MetaphoricalHigh', 'MetaphoricalLow', 'NonHuman', 'Threat', 'Threat1', 'Threat2'])

assert (dim == 2 or dim == 3), "invalid dim"

if dim == 2:
    trace_list = []
    for i in range(len(set(y_km))):
        trace_list.append(go.Scatter(x = embed[y_km==i,0], y = embed[y_km==i,1], mode = 'markers', marker = dict(size = 14, line = dict(width = 0), color =  np.concatenate((y_km[y_km == i], np.array([j for j in range(len(set(y_km)))]))), colorscale = 'Viridis', opacity = 0.6), text = text, name = str(i), hoverinfo='text'))
    
    # trace_pred = go.Scatter(x = embed[:,0], y = embed[:,1], mode = 'markers', marker = dict(size = 14, line = dict(width = 0), color = y_pred, colorscale = 'Viridis', opacity = 0.6), text = text, name = "Predict Label") 
else:
    # trace_true = go.Scatter3d(x = embed[:,0], y = embed[:,1], z = embed[:, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = y, colorscale = 'Viridis', opacity = 0.6), text = text, name = "True Label", hoverinfo='text') 
    trace_0 = go.Scatter3d(x = embed[y==0,0], y = embed[y==0,1], z = embed[y==0, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 0], np.array([0,1,2,3,4,5,6,7,8,9]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==0], name = l[0], hoverinfo='text') 
    trace_1 = go.Scatter3d(x = embed[y==1,0], y = embed[y==1,1], z = embed[y==1, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 1], np.array([0,1,2,3,4,5,6,7,8,9]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==1], name = l[1], hoverinfo='text') 
    trace_2 = go.Scatter3d(x = embed[y==2,0], y = embed[y==2,1], z = embed[y==2, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 2], np.array([0,1,2,3,4,5,6,7,8,9]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==2], name = l[2], hoverinfo='text') 
    trace_3 = go.Scatter3d(x = embed[y==3,0], y = embed[y==3,1], z = embed[y==3, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 3], np.array([0,1,2,3,4,5,6,7,8,9]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==3], name = l[3], hoverinfo='text') 
    trace_4 = go.Scatter3d(x = embed[y==4,0], y = embed[y==4,1], z = embed[y==4, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 4], np.array([0,1,2,3,4,5,6,7,8,9]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==4], name = l[4], hoverinfo='text')
    trace_5 = go.Scatter3d(x = embed[y==5,0], y = embed[y==5,1], z = embed[y==5, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 5], np.array([0,1,2,3,4,5,6,7,8,9]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==5], name = l[5], hoverinfo='text') 
    trace_6 = go.Scatter3d(x = embed[y==6,0], y = embed[y==6,1], z = embed[y==6, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 6], np.array([0,1,2,3,4,5,6,7,8,9]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==6], name = l[6], hoverinfo='text')  
    trace_7 = go.Scatter3d(x = embed[y==7,0], y = embed[y==7,1], z = embed[y==7, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 7], np.array([0,1,2,3,4,5,6,7,8,9]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==7], name = l[7], hoverinfo='text')  
    trace_8 = go.Scatter3d(x = embed[y==8,0], y = embed[y==8,1], z = embed[y==8, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 8], np.array([0,1,2,3,4,5,6,7,8,9]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==8], name = l[8], hoverinfo='text')  
    trace_9 = go.Scatter3d(x = embed[y==9,0], y = embed[y==9,1], z = embed[y==9, 2], mode = 'markers', marker = dict(size = 4, line = dict(width = 0), color = np.concatenate((y[y == 9], np.array([0,1,2,3,4,5,6,7,8,9]))), colorscale = 'Viridis', opacity = 0.6), text = text[y==9], name = l[9], hoverinfo='text')  

layout = go.Layout(
    hovermode= 'closest',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    legend=dict(x = -0.1, y = 1.2)
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