#############################################################################################
#Record Linkage using RecordLinkage library
#Last modified 5/16/18

#Optimization Opportunities include:
#Index Method used for making record pairs for comparison:
#			-Blocking v Sorted Neighbor
#feature(s) to use for Indexing
#Features to use for comparison
#Type of comparison to make (exact, string, other)
#Method to use for string comparisons (e.g., Jaro v Levenshtein v other)
#Thresholds to use for string comparisons (default is .85), which will vary from feature to feature
#computation for determining a match (currently using a simply summation of feature vectors)
#assignement of weights to feature vectors

##about the data being used##################################################################
#10K simulated dataset:
#	6K original recs
#	4K duplicated recs
#	max_modification_per_field = 3
#	max_modification_per_record = 3
#	distribution = zipf
#	modification_types = all
#	family number = 0
#
#
#script used to create the training data:  https://github.com/J535D165/FEBRL-fork-v0.4.2/blob/master/dsgen/generate2.py
###############################################################################################
#start imports
import random
import re
import sys
import time
import recordlinkage as rl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#end imports###################################################################################

#global variables
training_data=r"C:\Users\Andrew\Desktop\RecordLinkage\TrainingData\infile_dob_abbr.csv"
features=pd.DataFrame()
matches=pd.DataFrame()
rec_num=[]
####read data, set index to rec_id 
df=pd.read_csv(training_data,index_col=0,encoding = "ISO-8859-1")
df['rec_id']=df.index
###extract record number from rec_id for testing purposes
for item in df['rec_id']:
	item=str(item)
	item=re.search('rec-(.+?)-', item)
	rec_num.append(item.group(1))
df["rec_num"]=rec_num
#####################################
#split data into training and test
train,test=train_test_split(df)
#Define matches
def ScoreRecords():
	global features
	global df
	cols = df.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	df=df[cols]
	## create pairs
	indexer = rl.SortedNeighbourhoodIndex(on='given_name',window=5)
	pairs=indexer.index(train,test)
	compare_cl = rl.Compare()
	## methodology for scoring
	compare_cl.exact('postcode', 'postcode',label='postcode')
	compare_cl.string('surname', 'surname',method='jaro',threshold=.95,label='surname')
	compare_cl.string('given_name', 'given_name',method='jaro', threshold=.95, label='name')
	compare_cl.string('date_of_birth', 'date_of_birth',method='jaro', threshold=0.85,label='dob')
	compare_cl.string('suburb', 'suburb',method='jaro',label='suburb',threshold=.85)
	compare_cl.string('state', 'state',label='state',method='jaro', threshold=.85)
	compare_cl.string('address_1', 'address_1', method='jaro',threshold=0.9,label='address_1')
	compare_cl.exact('rec_num','rec_num',label='rec_num')
	##compute feature vector
	features = compare_cl.compute(pairs,train, test)
	total_score=[]
	features["Total_Score"]=features.sum(axis=1)
	features.fillna(0)
	y=[]
	for row in features["Total_Score"]:
		if row >=7:
			y.append(1)
		else:
			y.append(0)
	features["target"]=y
	features.to_csv('feature_vectors.csv',sep=",",encoding='utf-8')
	return (features)
	ScoreRecords()
####################Known Matches#################################################################################################################
def knownMatches():
	match=ScoreRecords()
	known_match=np.sum(match['rec_num'])
	return known_match
knownMatches()
###################Define Matches##################################################################################################################
def CreateMatches():
	global features
	matches = features[features.sum(axis=1) >6]
	return matches
CreateMatches()
###############################################################Evaluation of Compare################################################################
###Join matchced records and print to file for analysis purposes
def JoinMatches():
	global df
	#split matches multi-index in columns to join to paired record values
	matches=CreateMatches()
	df_matches=pd.DataFrame(matches.index,columns=['record'])
	dfm=pd.DataFrame(df_matches['record'].values.tolist())
	dfm.columns=['record1','record2']
	#join first record in matched pair
	matched_records=pd.merge(dfm, df, left_on='record1',right_on='rec_id', how='inner')
	#join second record in matched pair
	matched_records=pd.merge(matched_records, df, left_on='record2',right_on='rec_id',how='inner')
	#reorder columns to faciliate comparison and quality of matching
	matched_records=matched_records[['record1','record2','rec_num_x','rec_num_y','surname_x', 'surname_y','given_name_x','given_name_y',
						   'date_of_birth_x','date_of_birth_y','address_1_x','address_1_y','state_x','state_y','suburb_x','suburb_y','postcode_x','postcode_y']]
	matched_records.fillna(0)
	#write matches to file
	matched_records.to_csv('matched_records.csv',sep=",",encoding = "utf-8")
	return (matched_records)
JoinMatches()
###identify incorrect matches and print to file for analysis purposes
def findFalsePos():
	matched_records=JoinMatches()
	matched_records_errors=matched_records
	matched_records_errors=matched_records_errors.loc[matched_records['rec_num_x']!=matched_records['rec_num_y']]
	matched_records_errors.dropna()
	matched_records_errors.to_csv('false_pos_matches.csv',sep=",",encoding='utf-8')
	false_positives=len(matched_records_errors)
	return false_positives
findFalsePos()
###identify missed matches and print to file for analysis purposes
def missedMatches():
	record_set=ScoreRecords()
	##taking the record pairs that were not considered to be matches
	missed_matches = record_set[record_set.sum(axis=1) <=6]
	misses=np.sum(missed_matches['rec_num'])
	return misses
missedMatches()

############################################################Classification#######################################################
###data prep###################################################
def prepData():
	data=pd.read_csv('feature_vectors.csv',sep=",",encoding='utf-8')
	del data["Total_Score"]
	del data["rec_id"]
	del data["rec_id.1"]
	###calculate known matches, then delete for classification
	del data['rec_num']
	data.to_csv('feature_vectors_clean.csv',sep=",", encoding='utf-8')
	return data
prepData()
####Evaluate vector scoring methodology###########################################################################################
known_matches=knownMatches()
missed_matches=missedMatches()
false_positives=findFalsePos()
pairs=ScoreRecords()
print "number of comparison pairs in index:",len(pairs)
print "number of matching pairs:",known_matches
print "number of missed matches:",missed_matches
print "number of false positives:",false_positives
#supervised#######################################################################################################
#unsupervised methods#############################################################################################
###k-means####################################################
data=prepData()
kmeans = rl.KMeansClassifier()
result_kmeans = kmeans.learn(data)
print 'number of predicted pairs using K-means clustering:',len(result_kmeans)
###ECM Maximization###########################################
ecm = rl.ECMClassifier()
result_ecm = ecm.learn((data > 0.8).astype(int))
print 'the number of predicted pairs using ECM Maximization:',len(result_ecm)
