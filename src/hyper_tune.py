from hyperopt import hp, tpe, STATUS_OK, Trials
from hyperopt.fmin import fmin
from hyperopt import space_eval
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import random
from sklearn.model_selection import train_test_split
import gc

class AutoHyperOptimizer:
    def __init__(self, max_samples=50000, max_evaluations=25, seed=1, parameter_space={}):
        self.max_samples = max_samples
        self.max_evaluations = max_evaluations
        self.test_size = 0.25  ## fraction of data used for internal validation
        self.shuffle = False
        self.best_params = {}
        self.seed = seed
        self.param_space = parameter_space

    def gbc_objective(self, space):
        model = lgb.train(space, self.lgb_train, valid_sets=self.lgb_val, valid_names='eval',
                              verbose_eval=False, early_stopping_rounds=30, num_boost_round=88)
        fpr, tpr, thresholds = metrics.roc_curve(
            self.ys_test, model.predict(self.Xe_test, num_iteration=model.best_iteration))
        auc = metrics.auc(fpr, tpr)
        return {'loss': -auc, 'status': STATUS_OK}

    def fit(self, X, y, indicator):
        num_samples = len(X)
        print('Total samples passed for' \
              'hyperparameter tuning:', num_samples)
        if num_samples > self.max_samples:
            removeperc = 1.0 - (float(self.max_samples) / num_samples)
            print ('Need to downsample for managing time:,' \
                   'I will remove data percentage', removeperc)
            XFull, yFull = self.random_sample_in_order(X, y.reshape(-1, 1), removeperc)
            print('downsampled data length', len(XFull))
        else:
            XFull = X
            yFull = y

        Xe_train, self.Xe_test, ys_train, self.ys_test = \
            train_test_split(XFull, yFull.ravel(), test_size=self.test_size, random_state=self.seed, shuffle=True)

        self.lgb_train = lgb.Dataset(Xe_train, ys_train)#, free_raw_data=True)
        self.lgb_val = lgb.Dataset(self.Xe_test, self.ys_test)#, free_raw_data=True, reference=self.lgb_train)
        del X
        del y
        del Xe_train, ys_train
        gc.collect()
        if indicator == 1:
            ## just fit lightgbm once to obtain the AUC w.r.t a fixed set of hyper-parameters ##
            model = lgb.train(self.param_space, self.lgb_train, valid_sets=self.lgb_val, valid_names='eval',
                              verbose_eval=False, early_stopping_rounds=30, num_boost_round=100)
            fpr, tpr, thresholds = metrics.roc_curve(
                self.ys_test, model.predict(self.Xe_test, num_iteration=model.best_iteration))
            auc = metrics.auc(fpr, tpr)
            print("FIX AUC is", auc)
            return auc
        else:
            trials = Trials()
            best = fmin(fn=self.gbc_objective, space=self.param_space, algo=tpe.suggest, trials=trials,
                        max_evals=self.max_evaluations)
            params = space_eval(self.param_space, best)
            print('Best hyper-parameters', params)
            self.best_params = params
            model = lgb.train(params, self.lgb_train, valid_sets=self.lgb_val, valid_names='eval',
                              verbose_eval=False, early_stopping_rounds=30, num_boost_round=100)
            fpr, tpr, thresholds = metrics.roc_curve(
                self.ys_test, model.predict(self.Xe_test, num_iteration=model.best_iteration))
            auc = metrics.auc(fpr, tpr)
            print("Tune AUC is", auc)
            return params, auc

    def random_sample_in_order(self, X, y, removeperc, seed=1):
        if removeperc == 0:
            return X, y
        num_train_samples = len(X)
        rem_samples = int(num_train_samples * removeperc)
        np.random.seed(seed)
        skip = sorted(random.sample(range(num_train_samples), num_train_samples - rem_samples))
        print('[Utils]:Random sample length:', num_train_samples - rem_samples)
        return X[skip, :], y[skip, :]


class HyperLGB:
    def __init__(self, max_trials):
        #self.X = X
        #self.y = y
        self.max_trials = max_trials
    
    def data_split(self, X, y, test_size=0.5):
        len_y = int(len(X) * test_size)
        X_train  = X[:len_y]
        y_train = y[:len_y]
        X_valid = X[len_y:]
        y_valid = y[len_y:]
        return  X_train, X_valid, y_train, y_valid

    def hyperopt_lightgbm(self, df, drop_name, params):
        #X = df[]
        df = df[df.label.isnull() == False][:20000]
        feature_name = [i for i in df.columns if i not in (drop_name + ['label']) and 'target' not in i]
        X = df[feature_name]
        y = df[['label']]
        X_train, X_val, y_train, y_val = self.data_split(X, y, test_size=0.5)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            #"learning_rate": hp.loguniform("learning_rate", np.log(0.03), np.log(0.1)),
            "max_depth": hp.choice("max_depth", [-1, 4, 5, 6]),
            "num_leaves": hp.choice("num_leaves", np.linspace(32, 128, 32, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 0.8, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 0.8, 0.1),
            #"bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            #"min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 30,
                            valid_data, early_stopping_rounds=30, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            # in classification, less is better
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                            algo=tpe.suggest, max_evals=20, verbose=1,
                            rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        print(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        return hyperparams

class Tuner:
    
    def __init__(self):
        pass

    def hyperopt_lgb(self, df, drop_name,  max_iter=800, eta=2):
        feature_name = [i for i in df.columns if i not in drop_name + ['label']]
        X = df[df['label'].isnull()==True][feature_name]
        y = df[df['label'].isnull()==True]['label'].values
        print(X.shape,y.shape)
        #delta_n_estimators = 50
        delta_learning_rate = 0.005
        delta_max_depth = 1
        delta_feature_fraction = 0.1
        delta_bagging_fraction = 0.1
        delta_bagging_freq = 1
        delta_num_leaves = 20
        ## max number of function evaluation for hyperopt ##
        if X.shape[0] < 50000:
            lr = 0.005
            max_evaluation = 35
        elif X.shape[0] < 100000:
            lr = 0.01
            max_evaluation = 30
        elif X.shape[0] < 200000:
            lr = 0.03
            max_evaluation = 25
        else:
            lr = 0.1
            max_evaluation = 20
        param_choice_fixed = {'learning_rate': lr, \
                              'num_leaves': 60, \
                              'feature_fraction': 0.6, \
                              'bagging_fraction': 0.6, \
                              'bagging_freq': 2, \
                              'boosting_type': 'gbdt', \
                              'num_threads': 8,
                              'objective': 'binary', \
                              'metric': 'auc'}

        # Get the AUC for the fixed hyperparameter on the internal validation set
        autohyper = AutoHyperOptimizer(parameter_space=param_choice_fixed)
        best_score_choice1 = autohyper.fit(X, y, 1)
        print("---------------------------------------------------------------------------------------------------")
        print("[StreamSaveRetrainPredictor]:Fixed hyperparameters:", param_choice_fixed)
        print("[StreamSaveRetrainPredictor]:Best scores obtained from Fixed hyperparameter only is:",
              best_score_choice1)
        print("---------------------------------------------------------------------------------------------------")
        num_leaves_low = 5 if (param_choice_fixed['num_leaves'] - delta_num_leaves) < 5 else param_choice_fixed[
                                                                                                      'num_leaves'] - delta_num_leaves
        num_leaves_high = param_choice_fixed['num_leaves'] + delta_num_leaves

        feature_fraction_low = np.log(0.05) if (param_choice_fixed[
                                                    'feature_fraction'] - delta_feature_fraction) < 0.05 else np.log(
            param_choice_fixed['feature_fraction'] - delta_feature_fraction)
        feature_fraction_high = np.log(1.0) if (param_choice_fixed[
                                                    'feature_fraction'] + delta_feature_fraction) > 1.0 else np.log(
            param_choice_fixed['feature_fraction'] + delta_feature_fraction)

        bagging_fraction_low = np.log(0.05) if (param_choice_fixed[
                                                    'bagging_fraction'] - delta_bagging_fraction) < 0.05 else np.log(
            param_choice_fixed['bagging_fraction'] - delta_bagging_fraction)
        bagging_fraction_high = np.log(1.0) if (param_choice_fixed[
                                                    'bagging_fraction'] + delta_bagging_fraction) > 1.0 else np.log(
            param_choice_fixed['bagging_fraction'] + delta_bagging_fraction)

        bagging_freq_low = 1 if (param_choice_fixed['bagging_freq'] - delta_bagging_freq) < 1 else \
        param_choice_fixed['bagging_freq'] - delta_bagging_freq
        bagging_freq_high = param_choice_fixed['bagging_freq'] + delta_bagging_freq

        boosting_type = param_choice_fixed['boosting_type']
        objective = param_choice_fixed['objective']
        metric = param_choice_fixed['metric']

        param_space_forFixed = {
            'objective': "binary",
            'num_leaves': hp.choice('num_leaves', np.arange(num_leaves_low, num_leaves_high + 10, 10, dtype=int)),
            'feature_fraction': hp.loguniform('feature_fraction', feature_fraction_low, feature_fraction_high),
            'bagging_fraction': hp.loguniform('bagging_fraction', bagging_fraction_low, bagging_fraction_high),
            'bagging_freq': hp.choice('bagging_freq', np.arange(bagging_freq_low, bagging_freq_high + 1, 1, dtype=int)),
            'learning_rate': lr,#hp.loguniform('learning_rate', learning_rate_low, learning_rate_high),
            'lambda_l1':  hp.loguniform('lambda_l1', np.log(0.4), np.log(0.6)),
            'boosting_type': boosting_type,
            'num_threads': 8,
            'metric': metric,
            'verbose': -1
        }

        # run Hyperopt to search nearby region in the hope to obtain a better combination of hyper-parameters
        autohyper = AutoHyperOptimizer(max_evaluations=max_evaluation, parameter_space=param_space_forFixed)
        best_hyperparams_choice2, best_score_choice2 = autohyper.fit(X, y, 0)
        print("---------------------------------------------------------------------------------------------------")
        print(
        "[StreamSaveRetrainPredictor]:Best hyper-param obtained from Fixed Hyperparameters + Runtime Hyperopt is:",
        best_hyperparams_choice2)
        print(
        "[StreamSaveRetrainPredictor]:Best score obtained from Fixed Hyperparameter + Runtime Hyperopt is:",
        best_score_choice2)
        print("---------------------------------------------------------------------------------------------------")

        # Compare choice-1 & choice-2 and take the better one
        if best_score_choice1 >= best_score_choice2:
            best_hyperparams = param_choice_fixed
        else:
            best_hyperparams = best_hyperparams_choice2
        #self.best_params_hyperopt=best_hyperparams
        return best_hyperparams#,None