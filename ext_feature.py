'''
@lptMusketeers 2017.10.20
'''
import pandas as pd
import datetime
from functools import reduce
import codecs
import csv
from decimal import *
import numpy as np

class FeatureEngineering(object):
    def nondrop_precent(self,source_path,target_path):
        print("nondrop_precent...")
        df1 = pd.read_csv(source_path)
        df1["nondrop_precent"]=df1["nondropout_num"]/df1["course_num"]
        df1.to_csv(target_path,index=False)
        
    def add(self,x,y):
        return x+y    
        
    def op_character(self,source_path,target_path):
        print("op_character...")
        df1 = pd.read_csv(source_path)
        gpby_enrol = df1.groupby("enrollment_id")
        
        enrol_list = list()
        interval_list = list()
        last_minutes = list()
        valid_opnum = list()
        all_opnum = list()
        
        for enrollment_id,group in gpby_enrol:
            group.groupby("interval")
            for interval,group2 in group.groupby('interval'):
                enrol_list.append(enrollment_id)
                
                interval_list.append(interval)
                timelist = group2.time.tolist()
                h1 = datetime.datetime.strptime(timelist[0],'%H:%M:%S')
                h2 = datetime.datetime.strptime(timelist[len(timelist)-1],'%H:%M:%S')
                hh = h2-h1
                last_minutes.append(hh.seconds/60+1)
                valid_len = [0,0,0,0]
                valid_len[0] = len(group2[group2.event=='problem'])
                valid_len[1] = len(group2[group2.event=='video'])
                valid_len[2] = len(group2[group2.event == 'wiki'])
                valid_len[3] = len(group2[group2.event == 'discussion'])
                valid_opnum.append(reduce(self.add,valid_len))
                all_opnum.append(len(group2))
        df2 = pd.DataFrame({"enrollment_id":enrol_list,"interval":interval_list,"last_minutes":last_minutes,"valid_opnum":valid_opnum,"all_opnum":all_opnum})
        df2 = df2[["enrollment_id","interval","last_minutes","valid_opnum","all_opnum"]] #对DataFrame的列进行排序
        df2.to_csv(target_path,index=False)       
    
    def op_of_day(self,source_path,target_path1,target_path2):
        print("op_of_day...")
        log_file = codecs.open(source_path,'r','utf-8')
        log_final_file = codecs.open(target_path1,'w+','utf-8')
        log_statistic_file = codecs.open(target_path2,'w+','utf-8')
        framedata1 = pd.read_csv(log_file)
        writer1 = csv.writer(log_final_file,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer2 = csv.writer(log_statistic_file,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writedata = list()
        for i in range(0,111):
            writedata.append('')
        writedata[0]="enrollment_id"
        index =1
        for i in range(1,31):
            writedata[i]="all_opnum_"+str(i)
            writedata[i+30]="valid_opnum_"+str(i)
            writedata[i+60]="last_minutes_"+str(i)
            index += 3
        name_array1 = ["pre","mid","last","thirty_day"]
        name_array2 = ["min","max","sum","mean","std"]
        for name1 in name_array1:
            for name2 in name_array2:
                writedata[index]=name1+"_"+name2
                index += 1
                
        writer1.writerow(writedata)    
        writer2.writerow(writedata[0:1]+writedata[91:])        
        for enrollment_id,group in framedata1.groupby('enrollment_id'):
            writedata[0]=enrollment_id
            interval_list = group.interval.tolist()
            last_minutes_list = group.last_minutes.tolist()
            valid_num_list = group.valid_opnum.tolist()
            all_num_list = group.all_opnum.tolist()
            tag = 0
            for i in range(1,31):
                if i in interval_list:  #如果用户今天参加课程
                    writedata[i] = all_num_list[tag] #第一个30特征，记录操作总次数
                    writedata[i+30] = valid_num_list[tag] #第二个30特征，记录有效操作次数
                    writedata[i+60] = last_minutes_list[tag] #第三个30特征，记录持续时间
                    tag = tag + 1
                else:   #如果用户今天没有操作
                    writedata[i] = 0
                    writedata[i+30] = 0
                    writedata[i+60] = 0
            tag = 0
            
            '''
                            分前中后三个阶段统计总操作次数特征
            '''
            preall = list()
            midall = list()
            lastall = list()
            for i in range(1, 31):
                if i in interval_list:
                    if i > 0 and i <= 10:
                        preall.append(all_num_list[tag])
                    if i > 10 and i <= 20:
                        midall.append(all_num_list[tag])
                    if i > 20 and i <= 30:
                        lastall.append(all_num_list[tag])
                    tag = tag + 1
                else:
                    if i > 0 and i <= 10:
                        preall.append(0)
                    if i > 10 and i <= 20:
                        midall.append(0)
                    if i > 20 and i <= 30:
                        lastall.append(0)
    
            ########处理前十天的相关统计#######
            writedata[91] = min(preall)   #前十天中最小的操作次数
            writedata[92] = max(preall)     #前十天中最大的操作次数
            writedata[93] = np.array(preall).sum()    #前十天的总操作总次数
            writedata[94] = int(np.array(preall).mean())#前十天的平均次数
            writedata[95] = Decimal(np.array(preall).std()).quantize(Decimal('0.00')) #操作次数的标准差
    
            #########处理中间十天的相关统计#########
            writedata[96] = min(midall)
            writedata[97] = max(midall)
            writedata[98] = np.array(midall).sum()
            writedata[99] = int(np.array(midall).mean())
            writedata[100]=  Decimal(np.array(midall).std()).quantize(Decimal('0.00'))
    
            ########处理后十天的相关统计############
            writedata[101] = min(lastall)
            writedata[102] = max(lastall)
            writedata[103] = np.array(lastall).sum()
            writedata[104] = int(np.array(lastall).mean())
            writedata[105] =  Decimal(np.array(lastall).std()).quantize(Decimal('0.00'))
            ########处理三十天的相关统计############
            tag = 0
            writedata[106] = min(all_num_list)
            writedata[107] = max(all_num_list)
            templist = all_num_list
            writedata[108] = np.array(templist).sum()
            for i in range(0,30-len(all_num_list)):
                templist.append(0)
            writedata[109] = int(np.array(templist).mean())
            writedata[110] = Decimal(np.array(templist).std()).quantize(Decimal('0.00'))
            #print ('正在处理中....',enrollment_id)
            writer1.writerow(writedata)  #写入文件     
            writer2.writerow(writedata[0:1]+writedata[91:111])
    
    def feature_all(self,source_path1,source_path2,target_path):
        print("feature_all...")
        df1 = pd.read_csv(source_path1)
        df2 = pd.read_csv(source_path2)
        df2 = df2[["enrollment_id","username","course_id","course_num","nondropout_num","nondrop_precent","dropout"]] #调整列的顺序，把类标签放在最后一列
        df3 = pd.merge(df1,df2,on="enrollment_id",how="left")
        df3.to_csv(target_path,index=False)
        
    def feature_reduction(self,source_path1,source_path2,target_path):
        print("feature_reduction...")
        df1 = pd.read_csv(source_path1)
        df2 = pd.read_csv(source_path2)
        df2 = df2[["enrollment_id","username","course_id","course_num","nondropout_num","nondrop_precent","dropout"]] #调整列的顺序，把类标签放在最后一列
        df3 = pd.merge(df1,df2,on="enrollment_id",how="left")
        df3.to_csv(target_path,index=False)
    
    def feature_new(self):
        print("feature_new...")
        
    def ext_feature(self):
        
        source_path='preprocess_data/enrollment_dropout#.csv'
        target_path="feature/enrollment_nondrop_precent#.csv"
        self.nondrop_precent(source_path,target_path)
        
        source_path='preprocess_data/log_train_final#.csv'
        target_path="feature/log_feature#.csv"
        self.op_character(source_path,target_path)
        
        source_path="feature/log_feature#.csv"
        target_path1="feature/log_feature_final.csv"
        target_path2="feature/log_feature_statistic.csv"
        self.op_of_day(source_path,target_path1,target_path2)
        
        source_path1='feature/log_feature_final.csv'
        source_path2='feature/enrollment_nondrop_precent#.csv'
        target_path="feature/final_feature_all.csv"
        self.feature_all(source_path1,source_path2,target_path)
        
        source_path1='feature/log_feature_statistic.csv'
        source_path2='feature/enrollment_nondrop_precent#.csv'
        target_path="feature/final_feature_reduction.csv"
        self.feature_reduction(source_path1,source_path2,target_path)
    
  
if __name__=='__main__':
    #nondrop_precent()
    #op_character()
    #op_of_day()
    
    #nondrop_precent()
    #feature_all()
    feature_engineering = FeatureEngineering()
    feature_engineering.ext_feature()
    print("...done...")