
import numpy as np   # We recommend to use numpy arrays
import gc
import pandas as pd
import time
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#from gensim.models.word2vec import Word2Vec
from  sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD

def timmer(func):
    def warpper(*args,**kwargs):
        strat_time = time.time()
        r = func(*args, **kwargs)
        stop_time = time.time()
        print("[Success] Info: function: {}() done".format(func.__name__))
        print("the func run time is %.2fs" %(stop_time-strat_time))
        return r
    return warpper

def left_merge(data1, data2, on):
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp,on=on,how='left')
    result = result[columns]
    return result

def universe_mp_generator(gen_func, feat_list):
    """
    tools for multy thread generator
    """
    pool = Pool(4)
    result = [pool.apply_async(gen_func, feats) for feats in feat_list]
    pool.close()
    pool.join()
    return [aresult.get() for aresult in result]

def concat(L):
    """
    tools for concat new dataframe
    """
    result = None
    for l in L:
        if l is None:
            continue
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except Exception as err:
                print(err)
                print(l.head())
    return result

#################### UTILS for FE#############################

def count_helper(df, i):
    """
    tools for multy thread count generator
    """
    df['count_' + i] = df.groupby(i)[i].transform('count')
    return df[['count_' + i]].fillna(-99999).astype(np.int32)

def hash_helper(df, i):
    """
    tools for multy thread hash generator
    """
    df['hash_' + i] = df[i].apply(hash)
    return df[['hash_' + i]]

def bi_count_helper(df, i, j):
    """
    tools for multy thread bi_count
    """
    df['bicount_{}_{}'.format(i,j)] = df.groupby([i,j])[i].transform('count') 
    return df[['bicount_{}_{}'.format(i,j)]].fillna(-99999).astype(np.int32)

def cross_count_helper(df, i):
    """
    tools for multy thread bi_count
    """
    name = "count_"+ '_'.join(i)
    df[name] = df.groupby(i)[i[0]].transform('count') 
    return df[[name]].fillna(-99999).astype(np.int32)

def fast_join(x):
    r = ''
    for i in x:
        r += str(i) + ' '
    return r

def agg_helper(main_df, df, num_col, col, table_name):
    agg_dict = {}
    agg_dict['AGG_min_{}_{}_{}'.format(table_name, num_col, col)] = 'min'
    agg_dict['AGG_max_{}_{}_{}'.format(table_name, num_col, col)] = 'max'
    agg_dict['AGG_mean_{}_{}_{}'.format(table_name, num_col, col)] = 'mean'
    agg_dict['AGG_median_{}_{}_{}'.format(table_name, num_col, col)] = 'median'
    #agg_dict['AGG_skew_{}_{}_{}'.format(table_name, num_col, col)] = 'skew'
    agg_dict['AGG_var_{}_{}_{}'.format(table_name, num_col, col)] = 'var'
    agg_result = df.groupby(col)[num_col].agg(agg_dict)
    merget_result = left_merge(main_df[[col]], agg_result, on = [col])
    #merget_result = main_df[[col]].merge(agg_result, on = [col], how = 'left')
    return merget_result[[i for i in merget_result.columns if i != col]] #df[['ke_cnt_' + col]]

def hist_helper(main_df, df, cat_col, key, table_name):
    nfrac = min(10000, len(df)) / len(df)
    #print(nfrac)
    df = df.sample(frac = nfrac)
    df[cat_col] = df[cat_col].cat.add_categories("NAN").fillna("NAN")
    #seq_all = df.groupby([key])[cat_col].apply(lambda x:' '.join([str(i) for i in list(x)]))
    seq_all = df.groupby([key])[cat_col].apply(lambda x: list(x))
    #print(seq_all)
    seq_all_uid = seq_all.index
    cv = CountVectorizer(max_features = 4, analyzer= lambda xs:xs)#token_pattern='(?u)\\b\\w+\\b')
    #vectorizer.get_feature_names()
    seq_all_count = cv.fit_transform(seq_all.values)
    print(cv.get_feature_names())
    seq_all_lad = seq_all_count
    seq_all_lad = pd.DataFrame(seq_all_lad.todense())
    seq_all_lad.columns = ["TOP_HIST_FETURE_{}_{}_{}_{}".format(table_name, key, cat_col ,i) for i in seq_all_lad.columns]
   # print(seq_all_lad.head(5))
    seq_all_lad[key] = list(seq_all_uid)
    #seq_all_lad[",".join(base_feat)] = list(seq_all_uid)
    result = seq_all_lad
    result = left_merge(main_df, result, on = [key])
    return result

def diff_num_helper(df, time_col, col, num_col):
    df['ke_cnt_' + col] = df.groupby(col)[time_col].rank(ascending=False,method = 'first')
    df2 = df[[col, 'ke_cnt_' + col, num_col]].copy()
    df2['ke_cnt_' + col] = df2['ke_cnt_' + col] - 1
    df3 = pd.merge(df, df2, on=[col, 'ke_cnt_' + col], how='left')
    df['LAG_{}_{}'.format(col, num_col)] = df3[num_col +'_x'] - df3[num_col + '_y']
    del df2,df3
    gc.collect()
    return df[['DIFF_{}_{}'.format(col, num_col)]]

def lag_id_helper(df, time_col, col):
    df['ke_cnt_' + col] = df.groupby(col)[time_col].rank(ascending=False,method = 'first')
    df2 = df[[col, 'ke_cnt_' + col, time_col]].copy()
    df2['ke_cnt_' + col] = df2['ke_cnt_' + col] - 1
    df3 = pd.merge(df, df2, on=[col, 'ke_cnt_' + col], how='left')
    df['LAG_{}_{}'.format(col, time_col)] = (df3[time_col +'_x'] - df3[time_col + '_y'])
    df['LAG_{}_{}'.format(col, time_col)]  = df['LAG_{}_{}'.format(col, time_col)] .values.astype(np.int64) // 10 ** 9
    del df2,df3
    gc.collect()
    return df[['LAG_{}_{}'.format(col, time_col)]]


def base_embedding(x, model, size):
    vec = np.zeros(size)
    x = [item for item in x if model.wv.__contains__(item)]
    for item in x:
        vec += model.wv[str(item)]
    if len(x) == 0:
        return vec
    else:
        return vec / len(x)


def embedding_helper(df, col):
    input_ = df[col].fillna('NA').apply(lambda x: str(x).split(' '))
    model = Word2Vec(input_, size=12, min_count=2, iter=5, window=5, workers=4)
    data_vec = []
    for row in input_:
        data_vec.append(base_embedding(row, model, size=12))
    svdT = TruncatedSVD(n_components=6)
    data_vec = svdT.fit_transform(data_vec)
    column_names = []
    for i in range(6):
        column_names.append('embedding_{}_{}'.format(col, i))
    data_vec = pd.DataFrame(data_vec, columns=column_names)
    df = pd.concat([df, data_vec], axis=1)
    return df

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def add_smoth(series, p, a = 1):
    return (series.sum() + p / series.count() + a)

def get_target_mean(df, col):
    df[col] = df[col].fillna('-9999999')
    mean_of_target = df['label'].mean()

    kf = KFold(n_splits = 5, shuffle = True, random_state=2019)
    col_mean_name = "target_{}".format(col)
    X = df[df['label'].isnull() == False].reset_index(drop=True)
    X_te = df[df['label'].isnull()].reset_index(drop=True)
    X.loc[:, col_mean_name] = np.nan
    
    for tr_ind, val_ind in kf.split(X):
        X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
        X.loc[df.index[val_ind], col_mean_name] = X_val[col].map(X_tr.groupby(col)["label"].apply(lambda x: add_smoth(x, 0.5, 1)))

    tr_agg =  X[[col, "label"]].groupby([col])["label"].apply(lambda x: add_smoth(x, 0.5, 1)).reset_index()#['label'].mean().reset_index()
    tr_agg.columns = [col, col_mean_name]
    #print(tr_agg)

    X_te = X_te.merge(tr_agg, on = [col], how = 'left')
    _s = np.array(pd.concat([X[col_mean_name], X_te[col_mean_name]]).fillna(mean_of_target))
    df[col_mean_name] =  _s#add_noise(_s, mean_of_target / 20)
    return df[[col_mean_name]].fillna(-99999).astype(np.float32)



def temporal_agg_helper(u, v, num_col, key, time_col, table_name):
    tmp_u = u[[time_col, key]]
    tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    rehash_key = f'rehash_{key}'
    tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x))
    tmp_u.sort_values(time_col, inplace=True)
    agg_dict = {}
    agg_dict['ROLL_mean_{}_{}_{}'.format(table_name, num_col, key)] = 'mean'
    tmp_u = tmp_u.groupby(rehash_key)[num_col].rolling(5).agg(agg_dict)
    tmp_u.reset_index(0, drop=True, inplace=True)  # drop rehash index
   # print(tmp_u)
    ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)
    #print(ret)
    return ret[['ROLL_mean_{}_{}_{}'.format(table_name, num_col, key)]]


