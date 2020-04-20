# post preprocessing supervised classification
# lunch the script and it will guide you

import pandas as pd
import numpy as np
import os

# print('Welcome to the supervised classification script for CAIA vector')
# try:
# 	cores = int(input('Please enter the number of CPU cores to use (in a simple computer, better to use max = 2)\n'))
# 	if cores > 30:
# 		print('\nWARNING: number of cores is too high ({})').format(cores)
# 	algos = list(input('Which algorithms do you want to use:\n 1 - DT\n 2 - RF\n 3 - L2\n 4 - NB\n 5 - SVM\n 6 - kNN\nPlease input a list of choices such as : 1,3,5,6\n'))
# except:
# 	print('Input parameters are wrong, try again!')
# 	exit(1)

DT_state = True
RF_state = True
L2_state = True
NB_state = True
SVM_state = True
kNN_state = True

# if 1 in algos:
# 	DT_state = True
# if 2 in algos:
# 	RF_state = True
# if 3 in algos:
# 	L2_state = True
# if 4 in algos:
# 	NB_state = True
# if 5 in algos:
# 	SVM_state = True
# if 6 in algos:
# 	kNN_state = True


# read csvs and join them in a single dataset
data17 = pd.read_csv("17_detailed_caia.csv").fillna(0)
data22 = pd.read_csv("22_detailed_caia.csv").fillna(0)

data = pd.concat([data17, data22])
del data17
del data22

# get rid of the key
X = data.drop(['forward_flowStartSeconds',
             'forward_destinationTransportPort',
             'forward_destinationIPAddress',
             'forward_sourceIPAddress',
             'forward_sourceTransportPort',
             'forward_protocolIdentifier',
             'Label'
            ], axis = 1)

# save the labels in y
y = data['Label']

# split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_train = pd.DataFrame(X_train, columns=X.columns.values.tolist()).rename(columns={'Attack subcategory': 'Attack_subcategory', 'Attack Name': 'Attack_Name', 'Attack Reference': 'Attack_Reference'})
X_test = pd.DataFrame(X_test, columns=X.columns.values.tolist()).rename(columns={'Attack subcategory': 'Attack_subcategory', 'Attack Name': 'Attack_Name', 'Attack Reference': 'Attack_Reference'})
y_train = pd.DataFrame(y_train, columns=['Label'])
y_test = pd.DataFrame(y_test, columns=['Label'])

print(X_train.columns.values.tolist())
print(X_test.columns.values.tolist())
print(type(X_train))

# split furter the data into numerical data and attacks description
description_train = X_train[['attack_cat', 'Attack_subcategory', 'Attack_Name', 'Attack_Reference']]
description_test = X_test[['attack_cat', 'Attack_subcategory', 'Attack_Name', 'Attack_Reference']]
X_trai = X_train.drop(['attack_cat', 'Attack_subcategory', 'Attack_Name', 'Attack_Reference', '.'], axis = 1)
X_tes = X_test.drop(['attack_cat', 'Attack_subcategory', 'Attack_Name', 'Attack_Reference', '.'], axis = 1)

feat_set = "Consensus"

# free-up memory
del data

# preparing results file
file = open("results_train_test.csv","w")
# file.write(','.join(list(training)))
# file.write("\n")

# starting from now we have X_train, X_test, y_train, y_test, description_train, description_test
print('training_testing tables created')


# define the log transform
def logtransfo(x):
    return np.log10(1+x)

# apply the log transformation, (take care of selecting which features needs to be transformed, (use .drop))
# X_train = X_train.apply(logtransfo)
# X_test = X_test.apply(logtransfo)


# min max scaling
from sklearn.preprocessing import MinMaxScaler

scalermm = MinMaxScaler()
scalermm.fit(X_trai)

X_train = scalermm.transform(X_trai)
X_test = scalermm.transform(X_tes)


# DT to see importance
from sklearn.tree import DecisionTreeClassifier

# DT manually adjusted to avoid overfit 
dtree = DecisionTreeClassifier(min_samples_leaf=1, max_depth=9)
dtree.fit(X_train, y_train)


print('\nfeatures importance')
feat_imp = np.multiply(dtree.feature_importances_, 100)
print(feat_imp)
feat_imp.tofile(file, sep=",", format="%.3f")
file.write("\n")

# features selection, choose index of features
indices = np.where(dtree.feature_importances_ == 0)
X_train = np.delete(X_train, indices[0], axis=1)
X_test = np.delete(X_test, indices[0], axis=1)

print('\nirrelevant features (deleted)')
print(indices[0])

# center the data (- mean)
from sklearn.preprocessing import StandardScaler

scalerm = StandardScaler(with_std = False)
scalerm.fit(X_train)

X_train = scalerm.transform(X_train)
X_test = scalerm.transform(X_test)

# apply PCA
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print('\nPCA variance')
print(np.multiply(pca.explained_variance_ratio_,100))


# DT to see importance after PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

dtree = DecisionTreeClassifier(min_samples_leaf=1,max_depth=9)
dtree.fit(X_train_pca,y_train)
predict_train = dtree.predict(X_train_pca)
predict = dtree.predict(X_test_pca)


print('\nimportance of feaatures after PCA')
print(np.multiply(dtree.feature_importances_,100))

# feature reduction (select feature index)
indices = np.where(dtree.feature_importances_ == 0)
X_train_pca = np.delete(X_train_pca, indices[0], axis=1)
X_test_pca = np.delete(X_test_pca, indices[0], axis=1)
num_features=len(X_train[0])

print('\nirrelevant features (deleted)')
print(indices[0])

# importing metrics and X-val
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# DT training
if DT_state:
	print('\n####################### DT classification #######################')
	dtree = DecisionTreeClassifier(min_samples_leaf=1,max_depth=9)
	dtree.fit(X_train_pca,y_train)
	scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
	sc_tr = cross_validate(dtree, X_train_pca, y_train, scoring=scoring, cv=20, return_train_score=False)
	sc_ts = cross_validate(dtree, X_test_pca, y_test, scoring=scoring, cv=20, return_train_score=False)
	pred_train = dtree.predict(X_train_pca)
	pred = dtree.predict(X_test_pca)

	print('-------------TRAINING--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_train,pred_train)
	print(cm)
	print(classification_report(y_train,pred_train))
	print('roc_auc:')
	rauc = roc_auc_score(y_train, pred_train)
	print(rauc)
	file.write("feature set, algorithm, train/test, accuracy, precision, recall, f1score, roc_auc\n")
	acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
	file.write("%s, DT, training, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_training = pd.concat([X_trai, y_train, pd.DataFrame(pred_train, list(X_trai.index.values), columns=['Label_DT']), description_train], axis=1)
	#out.drop(out[out.y_train == out.predict_train or (out.y_train == 0 and out.predict_train == 1)].index).to_csv('caia_train_DT.csv', index=False, header= True)


	print('\n-------------TEST--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_test,pred)
	print(cm)
	print(classification_report(y_test,pred))
	print('roc_auc:')
	rauc = roc_auc_score(y_test, pred)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
	file.write("%s, DT, test, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_testing = pd.concat([X_tes, y_test, pd.DataFrame(pred, list(X_tes.index.values), columns=['Label_DT']), description_test], axis=1)
	#os.system('mailx -s "script" e1617933@student.tuwien.ac.at <<< "DT done"')

# Naive bayes
if NB_state:
	print('\n####################### NB classification #######################')
	from sklearn.naive_bayes import BernoulliNB
	gnb = BernoulliNB(alpha=47.076)
	gnb.fit(X_train_pca,y_train)

	scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
	sc_tr = cross_validate(gnb, X_train_pca, y_train, scoring=scoring, cv=15, return_train_score=False)
	sc_ts = cross_validate(gnb, X_test_pca, y_test, scoring=scoring, cv=15, return_train_score=False)
	pred = gnb.predict(X_test_pca)
	pred_train = gnb.predict(X_train_pca)

	print('-------------TRAINING--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_train,pred_train)
	print(cm)
	print(classification_report(y_train,pred_train))
	print('roc_auc:')
	rauc = roc_auc_score(y_train, pred_train)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
	file.write("%s, NB, training, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_training = pd.concat([out_training, pd.DataFrame(pred_train, list(out_training.index.values), columns=['Label_NB'])], axis=1)

	print('\n-------------TEST--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_test,pred)
	print(cm)
	print(classification_report(y_test,pred))
	print('roc_auc:')
	rauc = roc_auc_score(y_test, pred)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
	file.write("%s, NB, test, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_testing = pd.concat([out_testing, pd.DataFrame(pred, list(out_testing.index.values), columns=['Label_NB'])], axis=1)
	#os.system('mailx -s "script" e1617933@student.tuwien.ac.at <<< "NB done"')

# l2 logistic regression
if L2_state:
	print('\n####################### L2 classification #######################')
	from sklearn import linear_model

	l_reg = linear_model.LogisticRegression(penalty='l2', C= 94.789, tol=0.0009)
	l_reg.fit(X_train_pca,y_train)
	sc_tr = cross_validate(l_reg, X_train_pca, y_train, scoring=scoring, cv=15, return_train_score=False)
	sc_ts = cross_validate(l_reg, X_test_pca, y_test, scoring=scoring, cv=15, return_train_score=False)
	pred = l_reg.predict(X_test_pca)
	pred_train = l_reg.predict(X_train_pca)

	print('-------------TRAINING--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_train,pred_train)
	print(cm)
	print(classification_report(y_train,pred_train))
	print('roc_auc:')
	rauc = roc_auc_score(y_train, pred_train)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
	file.write("%s, L2, training, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_training = pd.concat([out_training, pd.DataFrame(pred_train, list(out_training.index.values), columns=['Label_L2'])], axis=1)

	print('\n-------------TEST--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_test,pred)
	print(cm)
	print(classification_report(y_test,pred))
	print('roc_auc:')
	rauc = roc_auc_score(y_test, pred)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
	file.write("%s, L2, test, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_testing = pd.concat([out_testing, pd.DataFrame(pred, list(out_testing.index.values), columns=['Label_L2'])], axis=1)
	#os.system('mailx -s "script" e1617933@student.tuwien.ac.at <<< "L2 done"')

# RF
if RF_state:
	print('\n####################### RF classification #######################')
	from sklearn.ensemble import RandomForestClassifier

	rfc = RandomForestClassifier(n_estimators=int(np.round(1+num_features/2)), min_samples_leaf=1,max_depth=9)
	rfc.fit(X_train_pca,y_train)
	sc_tr = cross_validate(rfc, X_train_pca, y_train, scoring=scoring, cv=15, return_train_score=False)
	sc_ts = cross_validate(rfc, X_test_pca, y_test, scoring=scoring, cv=15, return_train_score=False)
	pred = rfc.predict(X_test_pca)
	pred_train = rfc.predict(X_train_pca)

	print('-------------TRAINING--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_train,pred_train)
	print(cm)
	print(classification_report(y_train,pred_train))
	print('roc_auc:')
	rauc = roc_auc_score(y_train, pred_train)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
	file.write("%s, RF, training, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_training = pd.concat([out_training, pd.DataFrame(pred_train, list(out_training.index.values), columns=['Label_RF'])], axis=1)

	print('\n-------------TEST--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_test,pred)
	print(cm)
	print(classification_report(y_test,pred))
	print('roc_auc:')
	rauc = roc_auc_score(y_test, pred)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
	file.write("%s, RF, test, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_testing = pd.concat([out_testing, pd.DataFrame(pred, list(out_testing.index.values), columns=['Label_RF'])], axis=1)
	#os.system('mailx -s "script" e1617933@student.tuwien.ac.at <<< "RF done"')

# SVM
if SVM_state:
	print('\n####################### SVM classification #######################')
	from sklearn.svm import SVC

	svc = SVC(C=100,gamma=1000, max_iter=400)
	svc.fit(X_train_pca,y_train)
	sc_tr = cross_validate(svc, X_train_pca, y_train, scoring=scoring, cv=15, return_train_score=False)
	sc_ts = cross_validate(svc, X_test_pca, y_test, scoring=scoring, cv=15, return_train_score=False)
	pred = svc.predict(X_test_pca)
	pred_train = svc.predict(X_train_pca)

	print('-------------TRAINING--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_train,pred_train)
	print(cm)
	print(classification_report(y_train,pred_train))
	print('roc_auc:')
	rauc = roc_auc_score(y_train, pred_train)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
	file.write("%s, SVM, training, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_training = pd.concat([out_training, pd.DataFrame(pred_train, list(out_training.index.values), columns=['Label_SVM'])], axis=1)

	print('\n-------------TEST--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_test,pred)
	print(cm)
	print(classification_report(y_test,pred))
	print('roc_auc:')
	rauc = roc_auc_score(y_test, pred)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
	file.write("%s, SVM, test, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_testing = pd.concat([out_testing, pd.DataFrame(pred, list(out_testing.index.values), columns=['Label_SVM'])], axis=1)
	#os.system('mailx -s "script" e1617933@student.tuwien.ac.at <<< "SVM done"')

# kNN
if kNN_state:
	print('\n####################### kNN classification #######################')
	from sklearn.neighbors import KNeighborsClassifier
	gnb = KNeighborsClassifier(n_neighbors = 5)
	gnb.fit(X_train_pca,y_train)

	scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
	sc_tr = cross_validate(gnb, X_train_pca, y_train, scoring=scoring, cv=15, return_train_score=False)
	sc_ts = cross_validate(gnb, X_test_pca, y_test, scoring=scoring, cv=15, return_train_score=False)
	pred = gnb.predict(X_test_pca)
	pred_train = gnb.predict(X_train_pca)

	print('-------------TRAINING--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_train,pred_train)
	print(cm)
	print(classification_report(y_train,pred_train))
	print('roc_auc:')
	rauc = roc_auc_score(y_train, pred_train)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
	file.write("%s, kNN, training, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_training = pd.concat([out_training, pd.DataFrame(pred_train, list(out_training.index.values), columns=['Label_kNN'])], axis=1)

	print('\n-------------TEST--------------------')
	print('confusion matrix')
	cm = confusion_matrix(y_test,pred)
	print(cm)
	print(classification_report(y_test,pred))
	print('roc_auc:')
	rauc = roc_auc_score(y_test, pred)
	print(rauc)
	acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
	rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
	f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
	file.write("%s, kNN, test, %s, %s, %s, %s, %s\n" % (feat_set, acc, prec, rec, f1, rauc))
	out_testing = pd.concat([out_testing, pd.DataFrame(pred, list(out_testing.index.values), columns=['Label_kNN'])], axis=1)
	# os.system('mailx -s "script" e1617933@student.tuwien.ac.at <<< "kNN done"')

file.close()

out_testing.to_csv('out_testing.csv', index= False, header = True)
out_training.to_csv('out_training.csv', index= False, header = True)
