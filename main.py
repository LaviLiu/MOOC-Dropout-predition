'''
@lptMusketeers 2017.10.20
'''
from data_trans import PreProcess
from ext_feature import FeatureEngineering
from dropout_predict import DropoutPredict



if __name__=='__main__':
    '''
            程序入口，只需要执行该代码，即可完成数据预处理，特征抽取，预测分类
            如果特征数据已经有了，可以把预处理部分注释，直接运行分类预测
    '''
    
    #数据预处理
    '''
    preprocess = PreProcess()
    preprocess.data_trans()
    '''
    #抽取特征
    '''
    feature_engineering = FeatureEngineering()
    feature_engineering.ext_feature()
    '''
    #预测分类
    
    prediction = DropoutPredict()
    prediction.drop_predict()
    
    print("...done...")