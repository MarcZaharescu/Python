'''
Created on 10 Nov 2016
@author: Marc Zaharescu

Before running the part1.py file make sure that the file intern_casestudy_data.csv is in the same directory
'''

import csv
import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import itertools
 
''' read the csv file '''
df=pd.read_csv('intern_casestudy_data.csv')

''' create a set with aff unique values '''
L1=  list(set(df['affiliate_id']))

''' get the top 10 markets based on the total sum ''' 
df2=df.groupby(['affiliate_id', 'mkt'])['bkgs'].sum()
df2 = df2.groupby(level=[0]).tail(10).reset_index() 

''' combine top 10 markets with their total value in a tuple ''' 
df3=df2
df3['combined']= list(zip(df3['mkt'],df3['bkgs']))
grouped = df3.groupby('affiliate_id')
L2 = []
for _, grp in grouped:
    L2.append( grp[['combined']].values.ravel().tolist()) 
    
''' transform the market nominal values into numerical values and get the top 50 markets for each aff_id based on the total value'''    
df_mkt = preprocessing.LabelEncoder()
df4=df.groupby(['affiliate_id', 'mkt'])['bkgs'].sum()
df4 = df4.groupby(level=[0]).tail(50).reset_index() 
df4.mkt = df_mkt.fit_transform(df4.mkt)
grouped = df4.groupby('affiliate_id')

''' normalise the bkgs columns '''
minL = min(df4['bkgs'])
maxL = max(df4['bkgs'])
df4['bkgs'] =(df4['bkgs']-minL )/(maxL-minL)

L = []
for _, grp in grouped:
    L.append( grp[['mkt']].values.ravel().tolist()) 

 
''' train top 11 k-nn for the data set based on the top 10 markets + their values for each aff_id '''
''' we trained 11 knn because the first one would be the one on the lit would be its own self '''
nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(L)
distances, indices = nbrs.kneighbors(L)

''' transform the similar_parters values based on the indices from knn and discard the first element'''
L3 = [indices[i:i+1].astype('int32').tolist() for i in range(0, len(indices), 1)]
L3 = list(itertools.chain(*L3)) 
L3 = [(x[1:])  for x in L3]
 
''' create a dictionary with the desired values '''
a = dict(
    partner_name=L1,
    top10markets=L2,
    similar_partners=L3  
)

''' create an intermediate csv file to store the values from the dictionary based on the desired output format'''
keys = ['partner_name',  'top10markets','similar_partners']
with open('csvfile.csv', 'wb') as f:  # Just use 'w' mode in 3.x
    w = csv.writer(f)
    w.writerow(keys)
    w.writerows(zip(*[a[key] for key in keys]))

''' open the csv file and transform it into a json file '''    
csvfile = open('csvfile.csv', 'r')   
jsonfile = open('jsonfile.json', 'w')

reader = csv.DictReader( csvfile, keys)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')
    