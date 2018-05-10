# -*- coding:utf-8 -*-

'''
@lptMusketeers 2017.10.20
'''
import pickle
import pandas as pd
import numpy as np
from unittest.mock import inplace

class PreProcess(object):
    def gen_courseid_dict(self,source_path):
        df = pd.read_csv(source_path,usecols=[0])
        course_map = pd.factorize(df.course_id)[1]
        course_dict = dict(zip(course_map,range(len(course_map))))
        print ("course_dict done...")
        return course_dict
    
    def gen_username_dict(self,source_path_train,source_path_test):
        df = pd.read_csv(source_path_train,usecols=[1])
        username_map = pd.factorize(df.username)[1]
        username_dict = dict(zip(username_map,range(len(username_map))))
        
        df2 = pd.read_csv(source_path_test,usecols=[1])
        username_map2 = pd.factorize(df2.username)[1]
        diff = [w for w in username_map2 if w not in username_map]
        username_dict2 =dict(zip(diff,np.arange(len(username_map),len(username_map)+len(diff))))
        
        username_dict.update(username_dict2)
        print ("username_dict done...")
        return username_dict
    
    def course_map(self,x):
        return self.course_dict[x]
    
    def username_map(self,x):
        return self.username_dict[x]
    
    def time_split(self,x):
        x = x[:10]
        return x
    
    def enrollment_map(self,source_path_train,source_path_test,target_path_train,target_path_test):
        print ("read enrollment_train.csv")
        # enrollment_train.csv enrollment_id username course_id
        df1 = pd.read_csv(source_path_train,usecols=[0,1,2],converters={1:self.username_map,2:self.course_map})
        df1.to_csv(target_path_train,index=False)
        print ("read enrollment_test.csv")
        df2 = pd.read_csv(source_path_test,usecols=[0,1,2],converters={1:self.username_map,2:self.course_map})
        df2.to_csv(target_path_test,index=False)
        
    def date_map(self,source_path,target_path):
        print ("read date.csv")
        df1 = pd.read_csv(source_path,converters={0:self.course_map})
        df1["day_nums"]= (pd.to_datetime(df1["to"]) - pd.to_datetime(df1["from"]))
        df1["day_nums"] = df1["day_nums"].map(lambda x: x.days)
        df1.to_csv(target_path,index=False)
    
    def log_clean(self,source_path_train,target_path_train,source_path_test,target_path_test):
        print ("read log_train.csv ")
        df1 = pd.read_csv(source_path_train,usecols=[0,1,3]) #change
        df1["date"] = df1["time"].map(lambda x: x[:10])
        df1["time"] = df1["time"].map(lambda x: x[11:])
        df1.to_csv(target_path_train,index=False)
        print ("read log_test.csv ")
        df2 = pd.read_csv(source_path_test,usecols=[0,1,3]) #change
        df2["date"] = df1["time"].map(lambda x: x[:10])
        df2["time"] = df1["time"].map(lambda x: x[11:])
        df2.to_csv(target_path_test,index=False)     
    
    def course_enrollment(self,source_path_train,source_path_test,source_path_date,target_path_train,target_path_test):
        print("course_enrollment....")
        df1 = pd.read_csv(source_path_train) #如果不设置index，read_csv读取是默认index(序号)，不是第一列
        df2 = pd.read_csv(source_path_test)
        df3 = pd.read_csv(source_path_date)
        df4 = pd.merge(df1,df3,how="left",left_on="course_id",right_on="course_id")
        
        df5 = pd.merge(df2,df3,how="left",on="course_id")
        df4.to_csv(target_path_train,index=False)
        df5.to_csv(target_path_test,index=False)
        
    def log_interval(self,source_path_log_train,source_path_enrol_train,target_path):
        print("log_interval....")
        df1 = pd.read_csv(source_path_log_train)
        df2 = pd.read_csv(source_path_enrol_train,usecols=[0,3])
        df3 = pd.merge(df1,df2,how="left",on="enrollment_id")
        df3["interval"]= (pd.to_datetime(df3["date"]) - pd.to_datetime(df3["from"]))
        df3["interval"] = df3["interval"].map(lambda x: x.days+1)
        df3.drop(["from"],axis=1,inplace=True)
        df3.to_csv(target_path,index=False)
    
    def enrollment_dropout(self,source_path_enrol_train,source_path_truth,target_path):
        print("merge_enrollment")
        df1 = pd.read_csv(source_path_enrol_train)
        df2 = pd.read_csv(source_path_truth,names=['enrollment_id','dropout']) #如果文件没有列名可以在读取文件的时候为文件指定列名
        df3 = pd.merge(df1,df2,how="left",on="enrollment_id")
        gpby_user = df3.groupby("username");
        df4 = gpby_user.course_id.count().to_frame()
        df4.rename(columns={'course_id':'course_num'}, inplace = True)
        gpby_user_dropout = df3.groupby(["username","dropout"]);
        df5 = gpby_user_dropout.course_id.count().unstack().fillna(0)
        df5.rename(columns={0:'nondropout_num', 1:'dropout_num'}, inplace = True)
        df5.drop(["dropout_num"],axis=1,inplace=True)
        df6 = pd.merge(df1,df4,how="left",left_on="username",right_index=True)
        df6 = pd.merge(df6,df5,how="left",left_on="username",right_index=True)
        df7 = pd.merge(df6,df2,how="left",on="enrollment_id")
        df7.to_csv(target_path,index=False)
    
    def enrollment_dropout2(self,source_path_enrol_train,source_path_enrol_test,source_path_truth,target_path_train,target_path_test):
        print("merge_enrollment")
        df1 = pd.read_csv(source_path_enrol_train)
        df2 = pd.read_csv(source_path_truth,names=['enrollment_id','dropout']) #如果文件没有列名可以在读取文件的时候为文件指定列名
        df3 = pd.merge(df1,df2,how="left",on="enrollment_id")
        gpby_user = df3.groupby("username");
        df4 = gpby_user.course_id.count().to_frame()
        df4.rename(columns={'course_id':'course_num'}, inplace = True)
        gpby_user_dropout = df3.groupby(["username","dropout"]);
        df5 = gpby_user_dropout.course_id.count().unstack().fillna(0)
        df5.rename(columns={0:'nondropout_num', 1:'dropout_num'}, inplace = True)
        df5.drop(["dropout_num"],axis=1,inplace=True)
        
        df6 = pd.merge(df1,df4,how="left",left_on="username",right_index=True)
        df6 = pd.merge(df6,df5,how="left",left_on="username",right_index=True)
        #训练样本把类标加进入
        df7 = pd.merge(df6,df2,how="left",on="enrollment_id")
        df7.to_csv(target_path_train,index=False)
        
        #在enrollment_train中没有出现过的用户，没有类标无法计算辍学课程的数目，默认为0
        df8 = pd.read_csv(source_path_enrol_test)
        df9 = pd.merge(df8,df4,how="left",left_on="username",right_index=True)
        df10 = pd.merge(df9,df5,how="left",left_on="username",right_index=True)
        df10.fillna(0,inplace=True)
        df10.to_csv(target_path_test,index=False)
        
            
    def data_trans(self):
        
        source_path = 'original_data/train/date.csv'
        self.course_dict = self.gen_courseid_dict(source_path)
        
        source_path_train = 'original_data/train/enrollment_train.csv'
        source_path_test = 'original_data/test/enrollment_test.csv'
        self.username_dict = self.gen_username_dict(source_path_train,source_path_test)
        
        source_path_train='original_data/train/enrollment_train.csv'
        source_path_test='original_data/test/enrollment_test.csv'
        
        target_path_train="preprocess_data/enrollment_train#.csv"
        target_path_test="preprocess_data/enrollment_test#.csv"
        self.enrollment_map(source_path_train,source_path_test,target_path_train,target_path_test)
        
        source_path='original_data/train/date.csv'
        target_path="preprocess_data/date#.csv"
        self.date_map(source_path,target_path)
        
        source_path_train='original_data/train/log_train.csv'
        target_path_train="preprocess_data/log_train#.csv"
        source_path_test='original_data/test/log_test.csv'
        target_path_test="preprocess_data/log_test#.csv"
        self.log_clean(source_path_train,target_path_train,source_path_test,target_path_test)
        
        source_path_train='preprocess_data/enrollment_train#.csv'
        source_path_test='preprocess_data/enrollment_test#.csv'
        source_path_date='preprocess_data/date#.csv'
        target_path_train="preprocess_data/course_enrollment_train#.csv"
        target_path_test="preprocess_data/course_enrollment_test#.csv"
        self.course_enrollment(source_path_train,source_path_test,source_path_date,target_path_train,target_path_test)
        
        #处理log_train
        source_path_log_train='preprocess_data/log_train#.csv'
        source_path_enrol_train='preprocess_data/course_enrollment_train#.csv'
        target_path_train="preprocess_data/log_train_final#.csv"
        self.log_interval(source_path_log_train,source_path_enrol_train,target_path_train)
        
        #处理log_test
        source_path_log_test='preprocess_data/log_test#.csv'
        source_path_enrol_test='preprocess_data/course_enrollment_test#.csv'
        target_path_test="preprocess_data/log_test_final#.csv"
        self.log_interval(source_path_log_test,source_path_enrol_test,target_path_test)
        
        '''
        source_path_enrol_train='preprocess_data/enrollment_train#.csv'
        source_path_truth='original_data/train/truth_train.csv'
        target_path="preprocess_data/enrollment_dropout#.csv"
        self.enrollment_dropout(source_path_enrol_train,source_path_truth,target_path)
        '''
        source_path_enrol_train='preprocess_data/enrollment_train#.csv'
        source_path_enrol_test='preprocess_data/enrollment_test#.csv'
        source_path_truth='original_data/train/truth_train.csv'
        target_path_train="preprocess_data/enrollment_dropout#.csv"
        target_path_test="preprocess_data/enrollment_dropout_test#.csv"
        self.enrollment_dropout2(source_path_enrol_train,source_path_enrol_test,source_path_truth,target_path_train,target_path_test)
        
        
'''
if __name__=='__main__':
    
    time_dict = gen_time_dict()
    course_dict = gen_courseid_dict()
    obj_dict = gen_object_dict()
    username_dict = gen_username_dict()
    weekday_dict = date2weekday()
    
    
    username_dict = gen_username_dict()
    course_dict = gen_courseid_dict()
    
    
    enrollment_map()
    date_map()
    
    
    #log_clean()
    
    enrollment_map()
    course_enrollment()
    course_enrollment()

    #log_interval()
    
    enrollment_dropout()
    print("...done...")
'''