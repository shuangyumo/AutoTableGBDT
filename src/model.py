import numpy as np   # We recommend to use numpy arrays
import gc
import os
from os.path import isfile
import random
os.system("pip3 install lightgbm==2.2.3")
import lightgbm as lgb
os.system('pip3 install pandas==0.24.2')
import pandas as pd
import time
import hashlib
import math
from collections import Counter
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve,log_loss
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn import metrics
from scipy import sparse
import json
import re
import operator
from collections import deque
from multiprocessing import Pool
from multiprocessing import cpu_count

print("CPU count", cpu_count())
#from gensim.models.word2vec import Word2Vec
os.system('pip3 install hyperopt==0.1.2')
from hyper_tune import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from feature_util import *

#################### UTILS #############################
SEED = 2016
# seed everything
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(SEED)

#################### Model define #############################
class Model():
    def __init__(self, info):
        # meta information
        self.time_budget = info["time_budget"]
        self.time_col = info["time_col"]
        self.start_time = info["start_time"]
        self.tables_info = info["tables"]
        self.relations = info["relations"]
        self.table_num = len(info["tables"])
        self.train_data = None
        self.train_label = None
        
        ### AutoML Matience
        self.nunique_dict = {}
        self.use_auto_hyper = False
        self.best_iter = None
        self.drop_name = []
        self.train_test_rate = 0.2
        self.random_state = 2018
        self.feature_importance = None
        self.related_table_feature_importance = None
        self.params_lgb = {
            "objective": "binary", 
            "metric":"auc", 
            'verbose': -1, 
            "seed": self.random_state, 
            #'boosting_type':'goss',
            #'two_round': False, 
            'num_threads': 4,
            'num_leaves':64, 
            'learning_rate': 0.05,
            'min_data': 200, 
            'bagging_fraction': 0.5,
            'feature_fraction': 0.5,
            #'neg_bagging_fraction' : 0.01,
            'max_depth': -1 ,
            #'lambda_l1': 0.5, 
            #'lambda_l2': 0.5,
        }
        self.begin_runtime = None
        self.use_time_count = True
   
    @timmer
    def fit(self, train_data, train_label, time_remain):
        self.begin_runtime =  time.time()
        # please check if random split is on A (array([ 13934, 476313, 177584, ..., 571920, 698955,   5763]), array([631538, 713372, 553668, ..., 493323, 768023, 138463]))
        self.train_data = train_data
        self.train_label = train_label
        return 
  
    @timmer    
    def predict(self, test_data, time_remain):
        # gen config
        print('Time Budget for this task', self.time_budget)
        df_train_main = self.train_data["main"]
        train_time_span = df_train_main[self.time_col].max() -  df_train_main[self.time_col].min()
        df_train_test = test_data
        test_time_span = df_train_test[self.time_col].max() -  df_train_test[self.time_col].min()
        print(train_time_span, test_time_span)
        drip_level = (train_time_span/np.timedelta64(1, 'h')/df_train_main.shape[1]) / (test_time_span/ np.timedelta64(1, 'h')/df_train_test.shape[1]) 
        print('Drift drip_level',drip_level)
        if drip_level> 3 or drip_level< 1/3:
            self.use_time_count = False
        df_train_main['label'] = self.train_label
        len_train = len(df_train_main)
        df_train_main = df_train_main.append(df_train_test).reset_index(drop = True)
        df_memory = df_train_main.memory_usage().sum() / 1024**2
        self.total_num = df_train_main.shape[1]
        self.gen_config()

        print("df_memory_usage", df_memory)
        max_column = int(16 * 1024 / (df_memory / df_train_main.shape[1]))
        print("df_max_column_usage", max_column)
        del df_train_test
        gc.collect()
        # get feature importance on raw feature
        #imp = self.lgb_select(df_train_main)
        #self.feature_importance = imp
        # drop no_use feature
        # feature extract
        #df_train_main = self.feature_extract(df_train_main)
        df_train_main = self.realted_table_parse(df_train_main)
        
        # static for nunique, delte all same
        for i in df_train_main.columns:
            #if i.startswith('m_') or i.startswith('c_') or i.startswith('n_'):
            self.nunique_dict[i] = df_train_main[i].fillna(-9999).nunique()
            if i in self.nunique_dict.keys() and self.nunique_dict[i] <= 1:
                print("DELETA------", i)
                del df_train_main[i]
                gc.collect()
        
        cat_mv_df = df_train_main[[i for i in df_train_main.columns if i.startswith('c_') or i.startswith('m_') or i.startswith('n_')] + ['label']]
        imp = self.lgb_select(cat_mv_df.loc[len_train//2:])
        self.feature_importance = imp
        self.drop_name.extend(list(imp[(imp.gain==0) & (imp.split==0)]['feature']))
        del cat_mv_df
        gc.collect()
        
        #df_train_main = self.feature_extract(df_train_main)
        df_train_main = self.basic_encode_feature(df_train_main)
        gc.collect()
        df_train_main = self.high_order_feature(df_train_main)
        gc.collect()
        # reverse string to catgory_sub_table
        for i in df_train_main.columns:
            if df_train_main.loc[:,i].dtypes == 'object' :
                #del df_train_main[i]
                # use count encoding to replace the result
                #if self.nunique_dict[i] > 16:
                df_train_main.loc[:,i] = LabelEncoder().fit_transform(df_train_main.loc[:,i].fillna('na'))
                #else:
                #    df_train_main[i] = df_train_main[i].astype('category')
            if df_train_main.loc[:,i].dtypes == 'datetime64[ns]':
                self.drop_name.append(i)
                #del df_train_main[i]
        # model train and predict use this feature
        #self.params = Tuner().hyperopt_lgb(df_train_main, self.drop_name)
        if self.use_auto_hyper:
            search_para = HyperLGB(max_trials = 20).hyperopt_lightgbm(df_train_main, self.drop_name, self.params_lgb)
            self.params_lgb.update(search_para)
        
        # make this work
        
        # for i in range(10):
        #     if i != 0:
        #         df_train_main = self.high_order_feature(df_train_main)
        #     #select_col = [i for i in df_train_main.columns if i]
        #     imp_epoch = self.lgb_select(df_train_main.loc[df_train_main.label.isnull()==False], False)
        #     drop_num = min(128, len(imp_epoch) // 4 * 3)
        #     drop_num = max(96, drop_num)
        #     delcol = list(imp_epoch[drop_num:]['feature'])
        #     self.drop_name.extend(delcol)
        #     usecol = [i for i in df_train_main.columns if i not in delcol]
        #     #print(usecol)
        #     #print(delcol)
        #     for i in delcol:
        #         if i != self.time_col:
        #             del df_train_main[i]
        #     gc.collect()
        #     if self.time_col not in usecol:
        #         usecol.append(self.time_col)
        #     df_train_main = df_train_main[usecol]
        
        
        # run 100 epoch to select feature from trail
        # if self.train_label.mean() < self.imbalanced_rate or self.train_label.mean() > 1 - self.imbalanced_rate or len(self.train_label) < 250000 :
        #     y = self.lgb_kfold_train(df_train_main, _epoch = 50, _sub =False)
        # else:
        #     y = self.lgb_mode_train_preict(df_train_main, _epoch = 50,  _sub=False)
        # # delete feature not important
        # self.drop_name.extend(self.epoch_importance[len(self.epoch_importance) // 3 * 2 :]['feature'])
        # train_agin
        
        imp_epoch = self.lgb_select(df_train_main.loc[:len_train//5,], False)
        drop_num = min(self.limit_feature_num , len(imp_epoch) // 5 * 4)
        delcol = list(imp_epoch[drop_num:]['feature'])
        delcol = [i for i in delcol if i != self.time_col]
        self.drop_name.extend(delcol)
        usecol = [i for i in df_train_main.columns if i not in delcol]
        for i in delcol:
            del df_train_main[i]
        gc.collect()
        df_train_main = df_train_main.loc[:,usecol]
        
        if self.train_label.mean() < self.imbalanced_rate or self.train_label.mean() > 1 - self.imbalanced_rate or len(self.train_label) < 250000 \
            or self.time_budget > 2400:
            y = self.lgb_kfold_train(df_train_main, _epoch = 800, _sub =True)
        else:
            time_start_ = time.time()
            r = []
            w = []
            r.append(self.lgb_mode_train_preict(df_train_main, _epoch = 800, _sub =True))
            w.append(3)
            time_end_ = time.time()
            print('left time', self.time_budget - (time_end_ - self.begin_runtime))
            print('use_time', (time_end_ - time_start_))
            if self.time_budget - (time_end_ - self.begin_runtime) > 1.4 * (time_end_ - time_start_):
                r.append(self.lgb_mode_train_preict(df_train_main, _epoch = 800, _sub =True))
            w.append(4)
            if self.time_budget - (time.time() - self.begin_runtime) > 1.4 * (time_end_ - time_start_):
                self.best_iter = None
                self.params_lgb['num_leaves'] = 32
                r.append(self.lgb_mode_train_preict(df_train_main, _epoch = 800, _sub =True))                
                w.append(2)
            w = np.array(w) / np.sum(w)
            y = np.average(r,weights = w,axis=0)
        print('Final left time', self.time_budget - (time.time() - self.begin_runtime))
        y = pd.DataFrame({'label':y})['label']
        return y



#################### Auto Config  #############################  
    @timmer
    def gen_config(self):
        #use for debug
        self.time_validation = True
        self.gp_rate = 1.0
        self.imbalanced_rate = 0.05
        self.cross_count_rate = 1.0
        self.kfold_num = 10
        if self.time_budget <= 300:
            self.limit_feature_num = 100
        elif self.time_budget <= 600:
            self.limit_feature_num = 118
        else:
            self.limit_feature_num = 128
        # if self.total_num >= 1000000 * 2:
        #     self.limit_feature_num = 128
        # elif self.total_num >= 800000 * 2:
        #     self.limit_feature_num = 118
        # else:
        #     self.limit_feature_num = 138
                #self.kfold_num = 5

        #p = { : len(self.train_data['main'])}
        # if self.time_budget:
        # elif self.time_budget:
        # elif self.time_budget:
        return 

#################### Related tables  #############################  
    @timmer
    def realted_table_parse(self, df):
        print(" -------now the feature number------------", len(df))
        # check if there is error in df.
        for i in self.relations:
            print(i)
            if i['table_A'] == 'main':
                df = self.get_related_table_feature(df, i['table_B'], i['key'], i['type'])
            else:
                pass
                # only handle many2one and one2one 2level tables others ommit.
                '''
                if i['type'] == 'many_to_one' or i['type'] == 'one_to_one':
                    key = i['key']
                    merge_name = i['table_B']
                    level2_df = self.train_data[i['table_B']]
                    level2_df = level2_df.add_suffix('_{}'.format(merge_name))
                    # this is a level A merge result.
                    left_merge_name = i['table_A']
                    left_merge_df =  self.train_data[left_merge_name]
                    # Attention: hack by this use mergekey format 
                    merget_back_key = [i for i in left_merge_df.columns if i.startswith('c_0')]
                    level2_df = left_merge_df[merget_back_key].merge(level2_df, left_on = key, right_on = ['{}_{}'.format(i, merge_name) for i in key], how = 'left')
                    for j in self.relations:
                        if j['table_B'] == left_merge_name and j['table_A'] == 'main':
                            first = [x for x in df.columns]
                            #print(level2_df)
                            df = self.get_related_table_feature(df, level2_df, j['key'], j['type'])
                            #print(df[[x for x in df.columns if x not in first]].head(10))
                    # level2_df need to merge
                    #level2_df = self.get_related_table_feature(level2_df, i['table_B'], i['key'], i['type'])
                    #
                '''
        return df


    @timmer
    def get_related_table_feature(self, df,  merge_df, key, types):
        assert len(key) == 1 #
        if isinstance(merge_df, str):
            df_related = self.train_data[merge_df]
        else:
            df_related = merge_df.copy()
            merge_df = '2level'
            print('begin 2level parse')
        # select
        for i in df_related.columns:
            if df_related[i].fillna(-9999).nunique() <= 1:
                del df_related[i]
                print('DROP_MERGE_KEY', i)
        if types == 'many_to_one' or types == 'one_to_one':
            # many2one or one2one can easily merged to main table  
            #_df_related = df_related.copy()
            #agg_r = df[[key[0], 'label']].groupby(key)['label'].agg({'label':'mean'}).reset_index()
            #select_importance_df =  df_related.merge(agg_r, on=key, how = 'left')
            # sort realted table importance
            #df_realated_imp = self.lgb_select(select_importance_df)
            #self.related_table_feature_importance = df_realated_imp
            #df_related = self.count_encode(df_related)
            #df_related = self.bi_count_encode(df_related, table_name=merge_df)
            #df_related = self.time_encode(df_related)
            df_related = df_related.add_suffix('_{}'.format(merge_df))
            df = df.merge(df_related, left_on = key, right_on = ['{}_{}'.format(i, merge_df) for i in key], how = 'left')
            del df['{}_{}'.format(key[0], merge_df)]
            gc.collect()
        elif types == 'one_to_many' or types == 'many_to_many':
            # this is the merge key target encoding no useful 
            _df_related = df_related.copy()
            agg_r = df[[key[0], 'label']].groupby(key)['label'].agg({'label':'mean'}).reset_index()
            select_importance_df =  df_related.merge(agg_r, on=key, how = 'left')
            # sort realted table importance
            df_realated_imp = self.lgb_select(select_importance_df.loc[select_importance_df.label.isnull()==False], related = True)
            #fea_imp_related, r_df = self.skf_lgb_select(select_importance_df, merge_df, key)
            self.related_table_feature_importance = df_realated_imp
            # this is target encoding result for related table
            #df_related = df_related.merge(r_df, on = key, how = 'left')            
            df_related = self.group_by_encoding_mp(df, _df_related, key[0], merge_df)
            #df_related = self.rolling_agg(df_related, _df_related, key[0], merge_df) # things not work
            df_related = self.get_agg_cat_count_id(df_related, _df_related, key[0], merge_df)
            df_related = self.time_encode(df_related, table_name = merge_df)
            df = df_related
            #df_related = df_related.add_suffix('_{}'.format(merge_df))
        print(" -------now the feature number------------")
        print(len(df))
        return df


#################### feature engerinner ############################# 
    @timmer
    def basic_encode_feature(self, df):
        df = self.target_encode(df)
        df = self.time_encode(df)
        df = self.count_encode(df)
        return df

    @timmer
    def high_order_feature(self, df):
        df = self.bi_count_encode(df)
        gc.collect()
        df = self.agg_num_encoding(df)
        gc.collect()
        #df = self.mv_embedding_encode(df)
        #df = self.cross_count_encode(df)
        if self.use_time_count:
            df = self.bi_time_count(df)
        gc.collect()
        return df

    @timmer
    def feature_extract(self, df):
        #df = self.hash_encode(df)
        df = self.target_encode(df)
        df = self.count_encode(df)
        df = self.bi_count_encode(df)
        df = self.time_encode(df)
        df = self.agg_num_encoding(df)
       # df = self.bi_time_count(df)
        #print("embedding-faeture")
        #df = self.mv_embedding_encode(df)
        #df = self.cross_time(df)
        return df
    

    @timmer
    def target_encode(self, df):
        num_df0 = df.shape[1]
        count_columns = []
        #imp_cat = self.get_imp_fea_seperate('CAT+MV')
        for i in df.columns:
            if i.startswith('c_') or i.startswith('m_') and i not in self.drop_name:
                nuni = self.nunique_dict[i]
                if nuni <= 8  or nuni > len(df) // 512:
                    print('SKIP TARGET_', i)
                    continue
                count_columns.append([df.loc[:,[i, 'label']], i])
        count_result = universe_mp_generator(get_target_mean, count_columns)
        del count_columns
        gc.collect()
        count_result.append(df)
        df = concat(count_result)
        num_df1 = df.shape[1]
        print("number of count feature is {}".format(num_df1 - num_df0))
        return df

    @timmer
    def count_encode(self, df):
        num_df0 = df.shape[1]
        count_columns = []
        for i in df.columns:
            if i.startswith('c_') or i.startswith('m_') or (self.use_time_count and i.startswith('MinFromZero_')) and i not in self.drop_name:
                if i in self.nunique_dict.keys() and self.nunique_dict[i] < 8:
                    print('SKIP COUNT', i)
                    continue
                count_columns.append([df.loc[:,[i]].fillna('-9999'), i])
        count_result = universe_mp_generator(count_helper, count_columns)
        del count_columns
        gc.collect()
        count_result.append(df)
        df = concat(count_result)
        num_df1 = df.shape[1]
        print("number of count feature is {}".format(num_df1 - num_df0))
        return df


    def bi_time_count(self, df):
        num_df0 = df.shape[1]
        import_cat_col = self.get_imp_fea_seperate('CAT+MV', 0.5)
        df[self.time_col] = df[self.time_col].values.astype(np.int64) // 10 ** 9 //60#/ 60 / 60 / 24# df[] // 60
        #print(df[self.time_col].value_counts())
        bi_count_columns = []
        for i in import_cat_col[:16]:
            if i in self.drop_name:
                continue
            bi_count_columns.append([df.loc[:,[i, self.time_col]], i, self.time_col])
            #count += 1        
        bi_count_results = universe_mp_generator(bi_count_helper, bi_count_columns)
        del bi_count_columns
        gc.collect()
        bi_count_results.append(df)
        df = concat(bi_count_results)
        num_df1 = df.shape[1]
        print("number of bi_time_count feature is {}".format(num_df1 - num_df0))
        return df      

    @timmer
    def hash_encode(self, df):
        num_df0 = df.shape[1]
        count_columns = []
        for i in df.columns:
            if i.startswith('c_') or i.startswith('m_') :
                count_columns.append([df[[i]], i])
                self.drop_name.append(i)
        count_result = universe_mp_generator(hash_helper, count_columns)
        del count_columns
        gc.collect()
        count_result.append(df)
        df = concat(count_result)
        num_df1 = df.shape[1]
        print("number of hash feature is {}".format(num_df1 - num_df0))
        return df
    
    @timmer
    def time_encode(self,df, table_name = 'main'):
        for i in df.columns:
            if i.startswith('t_'):
                df["DAY_{}_{}".format(table_name, i)] = df[i].dt.day#.astype(np.uint16)
                df["HOUR_{}_{}".format(table_name, i)] = df[i].dt.hour#.astype(np.uint16)
                df["MinFromZero_{}_{}".format(table_name, i)] = df[i].dt.hour*60 + df[i].dt.minute #attention their is bias
                df["RAWTIME_{}_{}".format(table_name, i)] = df[i].values.astype(np.int64) // 10 ** 9
                if i == self.time_col:
                    if df["DAY_{}_{}".format(table_name, i)].nunique() <= 3:
                        del  df["DAY_{}_{}".format(table_name, i)]
                    if df["HOUR_{}_{}".format(table_name, i)].nunique() <= 3:
                        del  df["HOUR_{}_{}".format(table_name, i)],  df["MinFromZero_{}_{}".format(table_name, i)]
                    del df["RAWTIME_{}_{}".format(table_name, i)]
                    #df["{}_minute_{}".format(table_name, i)] = df[i].dt.minute#.astype(np.uint8)
        return df
    
    @timmer
    def mv_embedding_encode(self, df):
        important_mv = self.get_imp_fea_seperate('MV', 1)
        count = 0
        for i in important_mv:
            if i.startswith('m_') and df[i].nunique() > 1024:
                if count >=2:
                    break
                count += 1
                print("embedding:", i)
                df = embedding_helper(df, i)
                if count == 4:
                    break
        return df 

    @timmer
    def cross_time(self, df, table_name = 'main'):
        raw_time = [i for i in df.columns if i.startswith('RAWTIME_')]
        print("encoding raw time", raw_time)
        if len(raw_time) <= 1:
            return df
        for index, i in enumerate(list(combinations(raw_time, 2))):
            df['TIME_DELAT_{}_{}'.format(i[0],i[1])] = df[i[0]] - df[i[1]]
            #if index >= 5:
            #    return df
        return df




    @timmer
    def get_top_hist(self, main_df, df, key, table_name = 'main'):
        num_df0 = main_df.shape[1]
        gp_columns = []
        for col in df.columns:
            if (col.startswith('c_'))and col != key:
                gp_columns.append([main_df[[key]], df[[key, col]], col, key, table_name])
        gp_result = universe_mp_generator(hist_helper, gp_columns)
        del gp_columns
        gc.collect()
        gp_result.append(main_df)
        df = concat(gp_result)
        num_df1 = df.shape[1]
        print("number of hist feature is {}".format(num_df1 - num_df0))                   
        return df
    
    @timmer
    def get_agg_cat_count_id(self, main_df, df, key, table_name = 'main'):
        """
        fast encoding method for 2many cat or mv
        """
        num_df0 = main_df.shape[1]
        bi_count_columns = []
        cols = self.get_imp_fea_seperate('CAT+MV', self.gp_rate, False)
        for col in cols[:3]:
            if (col.startswith('c_') or col.startswith('m_')) and col != key:
                bi_count_columns.append([df[[key,col]], key, col])
        # get_bicount_key
        bi_count_results = universe_mp_generator(bi_count_helper, bi_count_columns)
        bi_count_results.append(df[[key]])
        bi_count_df = concat(bi_count_results)
        gp_columns = []
        # get_bi_count_agg_by_key
        for col in bi_count_df.columns:
            if col.startswith('bicount_'):
                gp_columns.append([main_df.loc[:,[key]], bi_count_df.loc[:,[key, col]], col, key, table_name])
        gp_result = universe_mp_generator(agg_helper, gp_columns)
        gc.collect()
        gp_result.append(main_df)
        df = concat(gp_result)
        num_df1 = df.shape[1]
        print("number of agg_cat_count_id feature is {}".format(num_df1 - num_df0))                   
        return df
       
    @timmer    
    def group_by_encoding_mp(self, main_df, df, key, table_name = 'main'):
        """
        fast aggregate method for 2many num feature 
        """
        num_df0 = main_df.shape[1]
        if table_name == 'main':
            is_main = True
        else:
            is_main = False
        cols = self.get_imp_fea_seperate('NUM', self.gp_rate, is_main)
        gp_columns = []
        for col in cols:
            if col.startswith('n_') or col.startswith('target_encoding'):
                gp_columns.append([main_df.loc[:,[key]], df.loc[:,[key, col]], col, key, table_name])
        gp_result = universe_mp_generator(agg_helper, gp_columns)
        del gp_columns
        gc.collect()
        gp_result.append(main_df)
        df = concat(gp_result)
        num_df1 = df.shape[1]
        print("number of groupby feature is {}".format(num_df1 - num_df0))                   
        return df
    

    @timmer    
    def rolling_agg(self, main_df, df, key, table_name = 'main'):
        """
        fast aggregate method for 2many num feature 
        """
        assert type(key) == str
        num_df0 = main_df.shape[1]
        if table_name == 'main':
            is_main = True
        else:
            is_main = False
        if self.time_col in main_df.columns and self.time_col in df.columns:
            cols = self.get_imp_fea_seperate('NUM', self.gp_rate, is_main)
            gp_columns = []
            for col in cols:
                if col.startswith('n_'):
                    # (u, v, num_col, key, time_col, table_name):
                    gp_columns.append([main_df[[self.time_col, key]], df[[self.time_col, col]], col, key, self.time_col, table_name])
            gp_result = universe_mp_generator(temporal_agg_helper, gp_columns)
            del gp_columns
            gc.collect()
            gp_result.append(main_df)
            df = concat(gp_result)
            num_df1 = df.shape[1]
            print("number of rolling agg feature is {}".format(num_df1 - num_df0))
            print(df.head())               
            return df
        else:
            return main_df
    

    @timmer    
    def agg_num_encoding(self, df):
        num_df0 = df.shape[1]
        NUM_cols = self.get_imp_fea_seperate('NUM', self.gp_rate, True)
        CAT_cols = self.get_imp_fea_seperate('CAT+MV', self.gp_rate, True)
        count = 0
        gp_columns = []
        temp_df = []
        for key in CAT_cols[:4]:
            if self.nunique_dict[key]<=6:
                print("AGG SKIP", key)
                continue
            for col in NUM_cols[:5]:
                gen_name = 'AGG_mean_{}_{}_{}'.format('main', col, key)
                
                if key in self.drop_name or col in self.drop_name or \
                    gen_name in self.drop_name or gen_name in df.columns:
                    continue
                
                gp_columns.append([df.loc[:,[key]], df.loc[:,[key, col]], col, key, 'main'])
                count += 1
                if len(gp_columns) >=4:
                    temp_df.extend(universe_mp_generator(agg_helper, gp_columns))
                    del gp_columns
                    gp_columns = []
                if count >= 5: #hard number
                    break
        gp_result = universe_mp_generator(agg_helper, gp_columns)
        del gp_columns
        gc.collect()
        gp_result.append(df)
        gp_result.extend(temp_df)
        df = concat(gp_result)
        num_df1 = df.shape[1]
        print("number of groupby feature is {}".format(num_df1 - num_df0))                   
        return df

    @timmer
    def cross_count_encode(self, df):
        num_df0 = df.shape[1]
        count_columns = []
        import_cat_col = self.get_imp_fea_seperate('CAT+MV+MAIN', self.cross_count_rate, True)
        for index, i in enumerate(list(combinations(import_cat_col[:4], 3))):
            i = list(i)
            count_columns.append([df.loc[:,i],i])
        count_result = universe_mp_generator(cross_count_helper, count_columns)
        del count_columns
        gc.collect()
        count_result.append(df)
        df = concat(count_result)
        num_df1 = df.shape[1]
        print("number of count feature is {}".format(num_df1 - num_df0))
        return df

    @timmer
    def bi_count_encode(self, df, table_name = 'main'):
        num_df0 = df.shape[1]
        if table_name == 'main':
            is_main = True
        else:
            is_main = False

        import_cat_col = self.get_imp_fea_seperate('CAT+MV', self.cross_count_rate, is_main)

        top =  min(25, len(import_cat_col))
        if is_main == False:
            top =  min(15, len(import_cat_col))
        count = 0

        bi_count_columns = []
        temp_df = []
        first = max(5, int(len(import_cat_col)/5 * 4))
        second = max(6, int(len(import_cat_col)/5 * 4))
        for x in range(first):
            for y in range(x+1, second):
                if count == top:
                    break
                
                if x >= len(import_cat_col) or y >= len(import_cat_col) :
                    continue
                    
                i = import_cat_col[x]
                j = import_cat_col[y]
                if i > j:
                    s = j
                    j = i
                    i = s

                gen_name = 'bicount_{}_{}'.format(i,j)
                if i in self.drop_name or j in self.drop_name or \
                    gen_name in self.drop_name or gen_name in df.columns:
                    continue
                
                uni_i = self.nunique_dict[i]
                uni_j = self.nunique_dict[j]

                flag = False
                if uni_i < uni_j:
                    flag = df[[i,j]][:5000].groupby([j])[i].nunique(dropna=False).max() == 1
                else:
                    flag = df[[i,j]][:5000].groupby([i])[j].nunique(dropna=False).max() == 1
                if flag: 
                    print('------SKIP BI COUNT ----------',i , j)
                    self.drop_name.append(gen_name)
                    continue
                # memory optimize
                bi_count_columns.append([df.loc[:,[i,j]].fillna('-99999'), i, j])
                count += 1
                if len(bi_count_columns) >=4:
                    temp_df.extend(universe_mp_generator(bi_count_helper, bi_count_columns))
                    del bi_count_columns
                    gc.collect()
                    bi_count_columns = []

        # add_merge_key:
        merge_key_list = [i for i in df.columns  if i.startswith('c_0')]
        for index, i in enumerate(list(combinations(merge_key_list, 2))):
            print(i)
            bi_count_columns.append([df[[i[0],i[1]]].fillna('-99999'), i[0], i[1]])
        bi_count_results = universe_mp_generator(bi_count_helper, bi_count_columns)
        del bi_count_columns
        gc.collect()
        
        
        bi_count_results.append(df)
        bi_count_results.extend(temp_df)

        df = concat(bi_count_results)
        num_df1 = df.shape[1]
        print("number of bi_count feature is {}".format(num_df1 - num_df0))
        return df


    @timmer
    def num_diff_last_time_mp(self, df):
        num_df0 = df.shape[1]
        if self.time_col is not None and self.time_col in list(df.columns):
            import_cat_col = self.get_imp_fea_seperate('CAT+MV', 0.5)
            gp_columns = []
            for num_col in self.get_imp_fea_seperate('NUM', 1)[:4]:
                for col in import_cat_col[:3]:
                    gen_name = 'diff_{}_{}'.format(col, num_col)
                    if col in self.drop_name or num_col in self.drop_name or gen_name in self.drop_name:
                        continue
                    gp_columns.append([df[[self.time_col, col, num_col]], self.time_col, col, num_col])
            gp_result = universe_mp_generator(diff_num_helper, gp_columns)
            del gp_columns
            gc.collect()
            gp_result.append(df)
            df = concat(gp_result)
            num_df1 = df.shape[1]
            print("number of num_diff feature is {}".format(num_df1 - num_df0))             
        return df

    @timmer
    def id_lag_encode(self, df):
        num_df0 = df.shape[1]
        if self.time_col is not None and self.time_col in list(df.columns):
            import_cat_col = self.get_imp_fea_seperate('CAT+MV', 0.3)
            gp_columns = []
            for col in import_cat_col:
                gen_name = 'LAG_{}_{}'.format(col, self.time_col)
                if col in self.drop_name  or gen_name in self.drop_name:
                    continue
                gp_columns.append([df[[self.time_col, col]], self.time_col, col])
            gp_result = universe_mp_generator(lag_id_helper, gp_columns)
            del gp_columns
            gc.collect()
            gp_result.append(df)
            df = concat(gp_result)
            num_df1 = df.shape[1]
            print("number of num_diff feature is {}".format(num_df1 - num_df0))             
        return df
#################### feature seletor #############################    

    def train_test_split(self, X, y, test_size, random_state=2018):
        sss = list(StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state).split(X, y))
        print(sss[0])
        X_train = np.take(X, sss[0][0], axis=0)
        X_test = np.take(X, sss[0][1], axis=0)
        y_train = np.take(y, sss[0][0], axis=0)
        y_test = np.take(y, sss[0][1], axis=0)
        return [X_train, X_test, y_train, y_test]


    def gain_filter_f(self, fold_importance_df):
        lgb_drop = fold_importance_df[fold_importance_df['gain'] <= 0.001]['feature']
        lgb_drop= [i for i in lgb_drop if i not in self.drop_name]
        self.drop_name.extend(lgb_drop)
        print("LGB SELECTION DROP", lgb_drop)
        return

    @timmer
    def lgb_select(self, train_x, related = False):
        lgb_params = {
            "objective": "binary", 
            "metric": "auc", 
            'verbose': -1, 
            "seed": self.random_state, 
            'num_threads': 4,
            'boosting': 'rf',
            'num_leaves': 618,
            'max_depth': 10, #maybe slow for this
            'bagging_freq': 1,
            'bagging_fraction': 0.8, 
            'feature_fraction': min(max(math.log(train_x.shape[1], 2) / train_x.shape[1], 0.1), 0.5),
        }
        feature_name = [i for i in train_x.columns if i not in self.drop_name + ['label']]
        train_y =  train_x['label'].values
        del train_x['label']
        for i in train_x.columns:
            if train_x[i].dtypes == 'datetime64[ns]':
                train_x.loc[:,i] = train_x.loc[:,i].values.astype(np.int64) // 10 ** 9
            if train_x[i].dtypes == 'object':
                if train_x[i].fillna('na').nunique() < 16:
                    train_x.loc[:,i] = train_x.loc[:,i].fillna('na').astype('category')
                else:
                    train_x.loc[:,i] = LabelEncoder().fit_transform(train_x.loc[:,i].fillna('na').astype(str))#.astype('category')
                    
        clf = lgb.train(lgb_params, lgb.Dataset(train_x, train_y),
                        verbose_eval=100, num_boost_round=100)
        del train_x, train_y
        gc.collect()
        fold_importance_df = self.get_fea_importance(clf, feature_name)
        if not related:
            print('=============IMPORTANCE ORDER=================')
            print(fold_importance_df.head(30))
            print('=============IMPORTANCE ORDER=================')
        return fold_importance_df
    
    @timmer
    def skf_lgb_select(self, df, merge_df, key):
        lgb_params = self.params_lgb.copy()
        lgb_params['objective'] = 'xentropy'
        lgb_params['metric'] = 'logloss'
        lgb_params['learning_rate'] = 0.1
        # remove merge key not to leak
        predictors = [i for i in df.columns if i not in self.drop_name + ['label'] + key]
        for i in df.columns:
            if i[:2] == 't_':
                df[i] = df[i].values.astype(np.int64) // 10 ** 9
            if df[i].dtypes == 'object':
                df[i] = df[i].astype('category')
        train_df = df[df['label'].isnull()==False]
        test_df = df[df['label'].isnull()!=False]
        for i in train_df.columns:
            print(train_df[i].dtypes, i)

        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])
        feature_importance_df = pd.DataFrame()
        folds = KFold(n_splits = 4, shuffle=True, random_state=SEED)
        target = 'label'
        feats = predictors
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[predictors])):
            print("nfold_{}".format(n_fold))
            train_x, train_y = train_df[feats].iloc[train_idx], train_df[target].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target].iloc[valid_idx]
            print(train_x.shape)
            print(valid_x.shape)
            xgtrain = lgb.Dataset(train_x, label=train_y.values,
                                feature_name=predictors,
                                )
            xgvalid = lgb.Dataset(valid_x, label=valid_y.values,
                                feature_name=predictors,
                                )
            clf = lgb.train(lgb_params, 
                            xgtrain, 
                            valid_sets=[xgtrain, xgvalid], 
                            valid_names=['train','valid'], 
                            num_boost_round=50,
                            #early_stopping_rounds=50,
                            verbose_eval=50,
                            #feval=log_loss,
                            )
            oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
            sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits
            gain = clf.feature_importance('gain')
            fold_importance_df = pd.DataFrame({'feature':clf.feature_name(),
                                            'split':clf.feature_importance('split'),
                                            'gain':100*gain/gain.sum(),
                                            'fold':n_fold,                        
                                            }).sort_values('gain',ascending=False)
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            #print('Fold %2d SCORE : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()
        r_df = train_df[key].append(test_df[key]).reset_index(drop = True)
        r_df['target_encoding_' + merge_df ] = np.concatenate([oof_preds, sub_preds], axis = 0)

       # print('Full Score %.6f' % roc_auc_score(train_df[target], oof_preds))
        ft = feature_importance_df[["feature", "split","gain"]].groupby("feature").mean().sort_values(by="gain", ascending=False)

        return ft, r_df


    def balance(self, df, _random_state = SEED):
        print("balance before: len X: is", df.shape)
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]
        if len(pos_df) > len(neg_df):
            more_df, less_df =  pos_df, neg_df
        else:
            less_df, more_df =  pos_df, neg_df
        if len(less_df) * 1.0 / len(more_df) < self.imbalanced_rate:
            more_df = more_df.sample(n = int(less_df.shape[0] / self.imbalanced_rate * 0.25), random_state = _random_state)
        df = less_df.append(more_df).reset_index(drop=True).sample(frac = 1)
        del less_df, more_df, pos_df, neg_df
        gc.collect()
        print("balance after: len X: is", df.shape)
        return df

    def lgb_kfold_train(self, df, _epoch=1000, _sub=False):
        test = df.loc[df['label'].isnull()==True]
        #df = df[df['label'].isnull()==False]
        feature_name = [i for i in df.columns if i not in (self.drop_name + ['label']) and 'target' not in i]
        train_X = df.loc[df['label'].isnull()==False]
        train_Y = train_X[['label']]
        feature_importance_df = pd.DataFrame()
        skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state=self.random_state)
        sub_preds = np.zeros(test.shape[0])
        best_iter = []
        for n_fold, (train_idx, valid_idx) in enumerate(skf.split(train_X, train_Y['label'])):
            # time stop
            epoch_start_time = time.time()
            if n_fold >= self.kfold_num:
                break
            if train_Y['label'].mean() < self.imbalanced_rate:
                b_df = self.balance(train_X.iloc[train_idx])
                train_x, train_y = b_df[feature_name], b_df['label']
                stamp = b_df[self.time_col]
            else:
                train_x, train_y = train_X.iloc[train_idx][feature_name], train_Y.iloc[train_idx]
                stamp = train_X.iloc[train_idx][self.time_col]

            valid_x, valid_y = train_X.iloc[valid_idx][feature_name], train_Y.iloc[valid_idx]
            
            stamp = stamp.astype(np.int64) // 10 ** 9
            train_weight = ((stamp -stamp.min()//2)/stamp.max()).values

            #lgb_train =  lgb.Dataset(train_x, train_y, weight=train_weight)
            lgb_train =  lgb.Dataset(train_x, train_y)
            lgb_val = lgb.Dataset(valid_x, valid_y, reference=lgb_train)
            clf = lgb.train(self.params_lgb, lgb_train , valid_sets = lgb_val,feature_name = feature_name,
                            verbose_eval=50, early_stopping_rounds=500, num_boost_round= _epoch)
            best_iter.append(clf.best_score['valid_0']['auc'])
            fold_importance_df_fold = self.get_fea_importance(clf, feature_name)
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df_fold], axis=0)
            del train_x, train_y, valid_x, valid_y
            gc.collect()
            if _sub:
                sub_preds += clf.predict(test[feature_name], num_iteration=clf.best_iteration)/ skf.n_splits
            else:
                sub_preds = None
            epoch_end_time = time.time()
            if self.time_budget - (epoch_end_time - self.begin_runtime) < 1.4 * (epoch_end_time - epoch_start_time):
                break
        del train_X, train_Y
        gc.collect()
        feature_importance_df = feature_importance_df[
            ["feature", "split","gain","gain_percent"]
        ].groupby("feature",as_index=False).mean().sort_values(by="gain", ascending=False)
        self.epoch_importance = feature_importance_df
        print(feature_importance_df)
        print("BEST AUC SKF", np.mean(best_iter))
        return sub_preds

    def get_imp_fea_seperate(self, kind, rate, main_table = True):
        if main_table:
            fea_imp = self.feature_importance
        else:
            fea_imp = self.related_table_feature_importance
        if kind == 'CAT':
            df = fea_imp[fea_imp.feature.str.startswith('c_')]
            return list(df['feature'][:int(len(df) * rate)])
        elif kind == 'NUM':
            df = fea_imp[fea_imp.feature.str.startswith('n_')]
            return list(df['feature'][:int(len(df) * rate)])
        elif kind == 'MV':
            df = fea_imp[fea_imp.feature.str.startswith('m_')]
            return list(df['feature'][:int(len(df) * rate)])
        elif kind == 'TIME':
            df = fea_imp[fea_imp.feature.str.startswith('t_')]
            return list(df['feature'][:int(len(df) * rate)])
        elif kind == 'CAT+MV':
            df = fea_imp[fea_imp.feature.str.startswith('c_') | fea_imp.feature.str.startswith('m_')]
            return list(df['feature'][:int(len(df) * rate)])
        elif kind == 'CAT+MV+MAIN':
            df = fea_imp[fea_imp.feature.str.startswith('c_') | fea_imp.feature.str.startswith('m_')]
            df = df[~df.feature.str.contains('table')]
            return list(df['feature'][:int(len(df) * rate)])        
    def get_fea_importance(self, clf, feature_name):
        gain = clf.feature_importance('gain')
        split_= clf.feature_importance('split')
        importance_df = pd.DataFrame({
            'feature':clf.feature_name(),
            'split': split_,
            'gain': gain, # * gain / gain.sum(),
            'gain_percent':100 *gain / gain.sum(),
            'split_percent':100 *split_ / split_.sum(),
            })#
        importance_df['cross_score'] =  (0.3*  importance_df['gain_percent'].rank() + 0.7 * importance_df['split_percent'].rank() ) / len(importance_df)
        #importance_df['cross_score'] =  (0.45 *  importance_df['gain_percent'].rank() + 0.55 * importance_df['split_percent'].rank() ) / len(importance_df)

        importance_df = importance_df.sort_values('cross_score',ascending=False)
        return importance_df


#################### model train #############################    


    @timmer
    def lgb_model_train(self, df, _epoch):
        df = df.loc[df['label'].isnull()==False]
        feature_name = [i for i in df.columns if i not in self.drop_name + ['label']]

        print(df['label'].mean())
        print("predict use feature shape", df[feature_name].shape)

        if self.best_iter is None:
            if self.time_validation:
                # add this 
                df = df.sort_values([self.time_col])
                X_train = df[feature_name][:len(df)//5*4]
                X_val = df[feature_name][len(df)//5*4:]
                y_train = df['label'][:len(df)//5*4].values
                y_val = df['label'][len(df)//5*4:].values
                stamp = df[self.time_col][:len(df)//5*4]
            else:
                X_train, X_val, y_train, y_val = self.train_test_split(df[feature_name + [self.time_col]], df['label'].values, self.train_test_rate, self.random_state)
                stamp = X_train[[self.time_col]]
                del X_train[self.time_col], X_val[self.time_col]
            del df
            gc.collect()
                        
            stamp = stamp.astype(np.int64) // 10 ** 9
            train_weight = ((stamp -stamp.min()//2)/stamp.max()).values

 #           lgb_train = lgb.Dataset(X_train, y_train, weight = train_weight)
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

            del X_train, X_val, y_train, y_val
            gc.collect()
            clf = lgb.train(
                self.params_lgb, lgb_train, valid_sets=lgb_val, valid_names='eval', 
                verbose_eval=50, early_stopping_rounds=100, num_boost_round=_epoch)
            # debug
            self.best_iter = clf.best_iteration
        else:
            lgb_train = lgb.Dataset(df[feature_name], df['label'])
            gc.collect()
            clf = lgb.train(
                self.params_lgb, lgb_train,  verbose_eval=100,  num_boost_round=int(self.best_iter*1.1))
            # debug
            #self.best_iter = clf.best_iteration
        fea_importance_now = self.get_fea_importance(clf, feature_name)
        self.epoch_importance = fea_importance_now
        if self.best_iter is not None:
            print("--------------Most important feature--------------")
            print(fea_importance_now.head(30))
            print("--------------Latest important feature------------")
            print(fea_importance_now.tail(20))
        return clf
    


    @timmer    
    def lgb_mode_train_preict(self, df, _epoch = 1000, _sub=False):
        #df = self.reduce_mem_usage(df)
        clf = self.lgb_model_train(df, _epoch) 
        if _sub:
            feature_name = [i for i in df.columns if i not in self.drop_name + ['label']]        
            X = df.loc[df['label'].isnull()==True][feature_name]
            y = clf.predict(X, num_iteration=int(clf.best_iteration))#+ 5 * (self.batch_idx-1)))
            del X
            gc.collect()
            return pd.DataFrame({'label':y})['label']
        else:
            return None

    @timmer
    def reduce_mem_usage(self, df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            if col == 'label':
                continue
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df.loc[:, col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                        df.loc[:, col] = df[col].astype(np.uint8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df.loc[:, col]  = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                        df.loc[:, col]  = df[col].astype(np.uint16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df.loc[:, col]  = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                        df.loc[:, col]  = df[col].astype(np.uint32)                        
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df.loc[:, col]  = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df.loc[:, col]  = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df.loc[:, col]  = df[col].astype(np.float32)
                    else:
                        df.loc[:, col]  = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df