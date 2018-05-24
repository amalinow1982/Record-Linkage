#############################################################################################
#Record Linkage using RecordLinkage library
#Last modified 5/8/18

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
from tpot import TPOTClassifier
#from sklearn.cluster import KMeans
#from sklearn import preprocessing
#from sklearn.linear_model import LogisticRegression
#from sklearn import model_selection
#from pandas.plotting import scatter_matrix
#from sklearn.utils import shuffle
#end imports###################################################################################

#global variables
training_data=r"C:\Users\Andrew\Desktop\RecordLinkage\TrainingData\infile_dob_abbr.csv"
features=pd.DataFrame()
matches=pd.DataFrame()
rec_num=[]
pipeline_optimizer = TPOTClassifier()

####read data, set index to rec_id and add target variable, 'y' column
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
train,test=train_test_split(df, test_size=0.2)

#Define matches
def ScoreRecords():
	global features
	global df
	cols = df.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	df=df[cols]
	##create pairs
	x_train,y_train=train_test_split(train, test_size=0.5)
	indexer = rl.SortedNeighbourhoodIndex(on='given_name',window=3)
	pairs=indexer.index(x_train,y_train)
	compare_cl = rl.Compare()
	compare_cl.string('postcode', 'postcode', method='jaro', threshold=0.85,label='postcode')
	compare_cl.string('surname', 'surname',method='jaro',threshold=.95,label='surname')
	compare_cl.string('given_name', 'given_name',method='jaro', threshold=.95, label='name')
	compare_cl.string('date_of_birth', 'date_of_birth',method='jaro', threshold=0.85,label='dob')
	compare_cl.string('postcode', 'postcode', method='jaro', threshold=0.85,label='postcode')
	compare_cl.string('suburb', 'suburb',method='jaro',label='suburb',threshold=.85)
	compare_cl.string('state', 'state',label='state',method='jaro', threshold=.85)
	compare_cl.string('address_1', 'address_1', method='jaro',threshold=0.9,label='address_1')
	compare_cl.exact('rec_num','rec_num',label='rec_num')
	##compute feature vector
	features = compare_cl.compute(pairs,x_train, y_train)
	total_score=[]
	features["Total_Score"]=features.sum(axis=1)
	##look at distribution of scores across record-pairs
	plt.hist(features['Total_Score'],bins=8)
	plt.title(r'Distribution of feature score totals for record pairs')
	plt.show()
	y=[]
	for row in features["Total_Score"]:
		if row >=6:
			y.append(1)
		else:
			y.append(0)
			
	features["y"]=y
	###replace string in index columns with int
	features.to_csv('feature_vectors.csv',sep=",",encoding='utf-8')
	return (features)
	
ScoreRecords()
print "number of scored pairs:",len(features)

###############################################################Evaluation of Compare################################################################
###Join matchced records and print to file for analysis pusposes
def JoinMatches():
	global df
	#split matches multi-index in columns to join to paired record values
	matches=ScoreRecords()
	df_matches=pd.DataFrame(matches.index,columns=['record'])
	dfm=pd.DataFrame(df_matches['record'].values.tolist())
	dfm.columns=['record1','record2']

	#join first record in matched pair
	matched_records=pd.merge(dfm, df, left_on='record1',right_on='rec_id', how='inner')
	#join second record in matched pair
	matched_records=pd.merge(matched_records, df, left_on='record2',right_on='rec_id',how='inner')

	#reorder columns to faciliate comparison and quality of matching
	matched_records=matched_records[['rec_id_y','rec_id_x','record1','record2','surname_x', 'surname_y','given_name_x','given_name_y',
						   'date_of_birth_x','date_of_birth_y','address_1_x','address_1_y']]
	#write matches to file
	matched_records.to_csv('matched_records.csv',sep=",",encoding = "utf-8")
	return (matched_records)
JoinMatches()



############################################################Classification#######################################################
###data prep###################################################
data=pd.read_csv('feature_vectors.csv',sep=",",encoding='utf-8')
del data["Total_Score"]
del data["rec_id"]
del data["rec_id.1"]
###calculate known matches, then delete for classification
known_match=tot_match=np.sum(data['rec_num'])
del data['rec_num']

#print data.head(10)
#unsupervised methods#############################################################################################
###k-means####################################################
print "number of matching pairs:",known_match
kmeans = rl.KMeansClassifier()
result_kmeans = kmeans.learn(data)

print 'number of predicted pairs using K-means clustering:',len(result_kmeans)
###ECM Maximization###########################################
ecm = rl.ECMClassifier()
result_ecm = ecm.learn((data > 0.8).astype(int))

print 'the number of predicted pairs using ECM Maximization:',len(result_ecm)

###get rid of annoying divide by zero runtime error



#####Logistic Regression#######################
#split training data into features and dependent variable, 'y'#
#matches=GoldenPairs()
#y=matches['y']
#x_train, y_train=train_test_split(matches,y,test_size=.5,random_state=42)
# Initialize the classifier
#logreg = LogisticRegression()
#logreg.fit(x_train, y_train)

#print ("Coefficients: ", logreg.coefficients)

#data.fillna(0)
##total true matches
#tot_match=np.sum(data['y'])
#print tot_match
####create golden_pairs
#golden_pairs=data[0:5000]
###create subset of known matches
#data.sort_values(['y'],inplace=True,ascending=False)

#matched_records=df.sample(2000)
#golden_matches_index = matched_records.index
#Train the classifier
#nb = rl.NaiveBayesClassifier()
#nb.learn(golden_pairs, golden_matches_index)

# Predict the match status for all record pairs
#result_nb = nb.predict(data)
#print len(result_nb)