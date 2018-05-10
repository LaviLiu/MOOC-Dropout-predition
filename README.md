# MOOC-Dropout-predition
The competition of KDD CUP of 2015. Predict a student on the MOOC weather dropout or not.
# List
1.Python version 3.6.2
2.Develop tool eclipse
3.main.py is the entry of the project. You should run main.py.
4.Some files is not on the github. You can download them from BaiduCloudDisk. You can also download all the project from BaiduCloudDisk.
the link is [link](https://pan.baidu.com/s/1mz4B1EJI1Sy88AI107NtpA)
# Result
![svm](https://github.com/Laviyy/MOOC-Dropout-predition/blob/master/picture/svm.png)
![roc_svm](https://github.com/Laviyy/MOOC-Dropout-predition/blob/master/picture/roc_svm.png)

logistic_regression... 0.944986290601
             precision    recall  f1-score   support

          0       0.89      0.83      0.86      7439
          1       0.96      0.97      0.96     28724

avg / total       0.94      0.94      0.94     36163

accuracy_score: 0.943450488068
runtime:3.85

gradient_boosting...
The accuracy of GradientBoosting:
0.952962135379
             precision    recall  f1-score   support

          0       0.90      0.86      0.88      7439
          1       0.96      0.98      0.97     28724

avg / total       0.95      0.95      0.95     36163

accuracy_score: 0.951884522855
runtime:10.35

random_forest...
The accuracy of RandomForest:
0.950331154873
             precision    recall  f1-score   support

          0       0.88      0.87      0.87      7439
          1       0.97      0.97      0.97     28724

avg / total       0.95      0.95      0.95     36163

accuracy_score: 0.948593866659
runtime:3.8666666666666667

svm...
The accuracy of SVM:
0.941608655363
             precision    recall  f1-score   support

          0       0.87      0.83      0.85      7439
          1       0.96      0.97      0.96     28724

avg / total       0.94      0.94      0.94     36163

accuracy_score: 0.940159831872
runtime:707.75

mlp...
The accuracy of MLP:
0.945401110332
             precision    recall  f1-score   support

          0       0.89      0.84      0.86      7439
          1       0.96      0.97      0.97     28724

avg / total       0.94      0.95      0.94     36163

accuracy_score: 0.945054337306
runtime:300.95
