'''
@lptMusketeers 2017.10.20
'''
from __future__ import unicode_literals
import codecs
import numpy as np
import pylab as pl
from itertools import *
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import cross_validation
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.ensemble import RandomForestClassifier
import datetime
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score  
from sklearn.metrics import classification_report  
from sklearn.svm import SVC 
from sklearn import metrics

class DropoutPredict(object):
    def pyplot_performance(self,y,pic_name):
        x = [int(i) for i in range(1,11)]
        pl.figure(1)
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x,y,label=pic_name)
        pl.legend()
        pl.savefig("picture/"+pic_name+".png")
        
    def pyplot_roc(self,y_true, y_scores,title):
        auc_value = roc_auc_score(y_true, y_scores) 
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)  
        pl.figure(2) 
        pl.plot(fpr, tpr, label=title+' (area = %0.4f)' % auc_value)  
        pl.plot([0, 1], [0, 1])  
        pl.xlim([0.0, 1.0])  
        pl.ylim([0.0, 1.05])  
        pl.xlabel('False Positive Rate')  
        pl.ylabel('True Positive Rate')  
        #pl.title('ROC '+title)  
        pl.legend(loc="lower right")  
        pl.savefig("picture/roc_"+title+".png")
        
    def loadData(self,filename):
        print("loadData...")
        df1 = pd.read_csv(filename)
        df1.drop(["enrollment_id","course_id"],inplace=True,axis=1)
        df2 = df1.drop("dropout",inplace=False,axis=1)
        x = df2.values #DataFrame的值组成的二维数组
        x = scale(x) #去均值后规范化
        y = np.ravel(df1["dropout"])
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=20)
        return x_train,x_test,y_train,y_test
            
    def logistic_regression(self,x_train,y_train,x_test,y_test):
        print("logistic_regression...")
        starttime = datetime.datetime.now()
        
        clf1 = LogisticRegression()
        score1 = cross_validation.cross_val_score(clf1,x_train,y_train,cv=10,scoring="accuracy")
        self.pyplot_performance(score1,"LogisticRegression")
        '''
        x = [int(i) for i in range(1,11)]
        y = score1
        
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x,y,label='LogisticRegression')
        pl.legend()
        pl.savefig("picture/LogisticRegression.png")
        '''
        print (np.mean(score1))
        
        clf1.fit(x_train, y_train)
        y_true = y_test  
        y_pred = clf1.predict(x_test)  
        y_pred_pro = clf1.predict_proba(x_test)  
        y_scores = pd.DataFrame(y_pred_pro, columns=clf1.classes_.tolist())[1].values  
        print(classification_report(y_true, y_pred)) 
        print ("accuracy_score:",metrics.accuracy_score(y_true, y_pred))
        self.pyplot_roc(y_true, y_scores,"LogisticRegression")
        '''
        auc_value = roc_auc_score(y_true, y_scores) 
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)  
        pl.figure()
        lw = 2  
        pl.plot(fpr, tpr, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)  
        pl.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')  
        pl.xlim([0.0, 1.0])  
        pl.ylim([0.0, 1.05])  
        pl.xlabel('False Positive Rate')  
        pl.ylabel('True Positive Rate')  
        pl.title('ROC characteristic logistic regression')  
        pl.legend(loc="lower right")  
        pl.show()  
        '''
        endtime = datetime.datetime.now()
        print ("runtime:"+str((endtime - starttime).seconds / 60))
        
    def svm(self,x_train,y_train,x_test,y_test):
        print("svm...")
        starttime = datetime.datetime.now()
        clf=SVC(probability=True)
        #clf=SVC(kernel='rbf',probability=True)
        #clf = svm.LinearSVC(random_state=2016)
        score = cross_validation.cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')
        #print score2
        print ('The accuracy of SVM:')
        print (np.mean(score))
        '''
        x = [int(i) for i in range(1, 11)]
        y = score2
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='SVM')
        pl.legend()
        pl.savefig("picture/SVM.png")
        '''
        self.pyplot_performance(score,"SVM")
        clf.fit(x_train, y_train)
        y_true = y_test  
        y_pred = clf.predict(x_test)  
        y_pred_pro = clf.predict_proba(x_test)  
        y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values  
        print(classification_report(y_true, y_pred)) 
        print ("accuracy_score:",metrics.accuracy_score(y_true, y_pred))
        self.pyplot_roc(y_true, y_scores,"SVM")
        endtime = datetime.datetime.now()
        print ("runtime:"+str((endtime - starttime).seconds / 60))
        
    def naive_bayes(self,x_train,y_train,x_test,y_test):
        print("naive_bayes...")   
        starttime = datetime.datetime.now()
           
        clf = GaussianNB()
        score =  cross_validation.cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')
        print ("The accuracy of Naive Bayes:")
        print (np.mean(score))
        self.pyplot_performance(score,"NaiveBayes")
        '''
        x = [int(i) for i in range(1, 11)]
        y = score3
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='NB')
        pl.legend()
        pl.savefig("picture/NB.png")  
        '''
        clf.fit(x_train, y_train)
        y_true = y_test  
        y_pred = clf.predict(x_test)  
        y_pred_pro = clf.predict_proba(x_test)  
        y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values  
        print(classification_report(y_true, y_pred)) 
        print ("accuracy_score:",metrics.accuracy_score(y_true, y_pred))
        self.pyplot_roc(y_true, y_scores,"NaiveBayes")
        
        endtime = datetime.datetime.now()
        print ("runtime:"+str((endtime - starttime).seconds / 60))
          
    def decision_tree(self,x_train,y_train,x_test,y_test):
        print("decision_tree...") 
        starttime = datetime.datetime.now()
        
        clf = tree.DecisionTreeClassifier()
        score = cross_validation.cross_val_score(clf,x_train,y_train,cv=10,scoring="accuracy")
        print ('The accuracy of decision_tree:')
        print (np.mean(score))
        '''
        x = [int(i) for i in range(1, 11)]
        y = score4
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='DT')
        pl.legend()
        pl.savefig("picture/DT.png")
        '''
        self.pyplot_performance(score,"DecisionTree")
        
        clf.fit(x_train, y_train)
        y_true = y_test  
        y_pred = clf.predict(x_test)  
        y_pred_pro = clf.predict_proba(x_test)  
        y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values  
        print(classification_report(y_true, y_pred)) 
        print ("accuracy_score:",metrics.accuracy_score(y_true, y_pred))
        self.pyplot_roc(y_true, y_scores,"DecisionTree")
        
        endtime = datetime.datetime.now()
        print ("runtime:"+str((endtime - starttime).seconds / 60))
        
    def gradient_boosting(self,x_train,y_train,x_test,y_test):
        print("gradient_boosting...")     
        
        starttime = datetime.datetime.now()
        clf = GradientBoostingClassifier()
        score = cross_validation.cross_val_score(clf,x_train,y_train,cv=10,scoring="accuracy")
        print ('The accuracy of GradientBoosting:')
        print (np.mean(score))
        '''
        x = [int(i) for i in range(1, 11)]
        y = score5
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='GBDT')
        pl.legend()
        pl.savefig("picture/GBDT.png")
        '''
        self.pyplot_performance(score,"GradientBoosting")
        
        clf.fit(x_train, y_train)
        y_true = y_test  
        y_pred = clf.predict(x_test)  
        y_pred_pro = clf.predict_proba(x_test)  
        y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values  
        print(classification_report(y_true, y_pred)) 
        print ("accuracy_score:",metrics.accuracy_score(y_true, y_pred))
        self.pyplot_roc(y_true, y_scores,"GradientBoosting")
        endtime = datetime.datetime.now()
        print ("runtime:"+str((endtime - starttime).seconds / 60))
        
    def mlp(self,x_train,y_train,x_test,y_test):   
        print("mlp...") 
        starttime = datetime.datetime.now()
        
        clf = MLPClassifier(hidden_layer_sizes=(300,300,300),
                            activation='logistic', solver='sgd',
                            learning_rate_init = 0.001, max_iter=100000)
        score = cross_validation.cross_val_score(clf,x_train,y_train,cv=10,scoring="accuracy")
        print ('The accuracy of MLP:')
        print (np.mean(score))
        '''
        x = [int(i) for i in range(1, 11)]
        y = score
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='MLP')
        pl.legend()
        pl.savefig("picture/MLP.png")
        '''
        self.pyplot_performance(score,"MLP")
        
        clf.fit(x_train, y_train)
        y_true = y_test  
        y_pred = clf.predict(x_test)  
        y_pred_pro = clf.predict_proba(x_test)  
        y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values  
        print(classification_report(y_true, y_pred)) 
        print ("accuracy_score:",metrics.accuracy_score(y_true, y_pred))
        self.pyplot_roc(y_true, y_scores,"MLP")
        endtime = datetime.datetime.now()
        print ("runtime:"+str((endtime - starttime).seconds / 60))
        
    def random_forest(self,x_train,y_train,x_test,y_test): 
        print("random_forest...")   
        starttime = datetime.datetime.now()
             
        clf = RandomForestClassifier(n_estimators=100)   
        score = cross_validation.cross_val_score(clf,x_train,y_train,cv=10,scoring="accuracy")
        print ('The accuracy of RandomForest:')
        print (np.mean(score))
        '''
        x = [int(i) for i in range(1, 11)]
        y = score
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='RandForest')
        pl.legend()
        pl.savefig("picture/RandomForest.png")
        '''
        self.pyplot_performance(score,"RandomForest")
        
        clf.fit(x_train, y_train)
        y_true = y_test  
        y_pred = clf.predict(x_test)  
        y_pred_pro = clf.predict_proba(x_test)  
        y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values  
        print(classification_report(y_true, y_pred))
        print ("accuracy_score:",metrics.accuracy_score(y_true, y_pred)) 
        self.pyplot_roc(y_true, y_scores,"RandomForest")
        endtime = datetime.datetime.now()
        print ("runtime:"+str((endtime - starttime).seconds / 60))
        
        
    def drop_predict(self):
        filename = 'feature/final_feature_all.csv'
        x_train,x_test,y_train,y_test = self.loadData(filename)
        
        self.logistic_regression(x_train,y_train,x_test,y_test)
        #self.naive_bayes(x_train,y_train,x_test,y_test)
        self.gradient_boosting(x_train,y_train,x_test,y_test)
        self.random_forest(x_train,y_train,x_test,y_test)
        
        #self.decision_tree(x_train,y_train,x_test,y_test)
        self.svm(x_train,y_train,x_test,y_test)
        self.mlp(x_train,y_train,x_test,y_test)
        
