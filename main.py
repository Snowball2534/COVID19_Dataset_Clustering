import numpy as np
import function

# load data from dataset
with open('time_series_covid19_deaths_US.csv') as f:
    data = list(f)[1:]
d_dict = {}
for d in data:
    l = d.strip('\n').split(',')
    stat = np.array(list(map(int, l[13:])))
    c = l[6]
    if c in d_dict:
        d_dict[c] += stat
    else:
        d_dict[c] = stat
print("Loading data done")

# note that only cluster 50 states of USA
# manually delete the states which do not belong to USA
del d_dict['American Samoa']
del d_dict['Diamond Princess']
del d_dict['District of Columbia']
del d_dict['Grand Princess']
del d_dict['Guam']
del d_dict['Northern Mariana Islands']
del d_dict['Virgin Islands']
del d_dict['Puerto Rico']

# feature engineering
# five features:
# mean of increase per day
# variance of increase per day
# linear regression coefficients for cubed time, squared time, and time
i = 0
q3 = np.zeros((50, 5))
for d in d_dict.values():
    # normalize the data by dividing the corresponding population in the state
    q3_stat = d[1:]
    q3_stat = q3_stat/d[0]

    # feature extracting
    coff = np.polyfit(np.arange(q3_stat.shape[0]), q3_stat, 3)
    q3[i,:3] = coff[:3]
    q3_diff = np.zeros(q3_stat.shape[0]-1)  # generate the differenced time series
    for j in range(q3_stat.shape[0]-1):
        q3_diff[j] = q3_stat[j+1] - q3_stat[j]
    q3[i,3] = np.mean(q3_diff)
    q3[i,4] = np.var(q3_diff)
    i+=1

# rescale to [0,100]
min_ = q3.min(axis=0)
max_ = q3.max(axis=0)
rag = max_-min_
q3 = ((q3 - min_)/rag)*100

# replace the data by five features
i=0
for key in d_dict.keys():
    d_dict[key] = q3[i,:]
    i+=1

# hierarchical clustering with single linkage
# k = 5
clusters = function.hier_cluster(d_dict, 5, False)
print(clusters)

# hierarchical clustering with complete linkage
# k = 5
clusters = function.hier_cluster(d_dict, 5, True)
print(clusters)

# k-means clustering
# k = 5
clusters = function.kmeans(d_dict, 5)
print(clusters)

# print the center of each cluster and the total distortion
function.kmeans_analy(d_dict, clusters)




