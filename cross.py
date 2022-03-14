import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean

import utils

RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
    kf = KFold(n_splits = 5, random_state = RANDOM_STATE,shuffle=True)
    accList = []
    aucList = []
    for train, test in kf.split(X):
        Y_pred = models_partc.logistic_regression_pred(X[train],Y[train],X[test])
        acc,auc,precision,recall,f1score = models_partc.classification_metrics(Y_pred, Y[test])
        accList.append(acc)
        aucList.append(auc)
    return round(mean(accList),2), round(mean(aucList),2)


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
    ss = ShuffleSplit(n_splits=iterNo, test_size=test_percent,random_state = RANDOM_STATE)
    accList = []
    aucList = []
    for train, test in ss.split(X):
        Y_pred = models_partc.logistic_regression_pred(X[train],Y[train],X[test])
        acc,auc,precision,recall,f1score = models_partc.classification_metrics(Y_pred, Y[test])
        accList.append(acc)
        aucList.append(auc)
    return round(mean(accList),2), round(mean(aucList),2)


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

