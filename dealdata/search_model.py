# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error, r2_score

# %%
# 读取player 的数据
player = pd.read_csv("../datafrom200/players.csv")
# x_list = player.columns[7:-5]
# x_list = ['Crossing', 'Finishing', 'Heading_Accuracy', 'Short_Passing', 'Volleys',
#        'Dribbling', 'Curve', 'FK_Accuracy', 'Long_Passing', 'Ball_Control',
#        'Acceleration', 'Sprint_Speed', 'Agility', 'Reactions', 'Balance',
#        'Shot_Power', 'Jumping', 'Stamina', 'Strength', 'Long_Shots',
#        'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
#        'Composure', 'Defensive_Awareness']
# x_list = ['Ball_Control', 'Sprint_Speed', 'Reactions', 'Stamina', 'Composure',
#        'Standing_Tackle', 'Sliding_Tackle']  # xgboost feature selection
x_list = ['Finishing', 'Short_Passing', 'Dribbling', 'Long_Passing',
       'Ball_Control', 'Acceleration', 'Sprint_Speed', 'Reactions', 'Balance',
       'Shot_Power', 'Stamina', 'Strength', 'Aggression', 'Vision',
       'Penalties', 'Standing_Tackle', 'Sliding_Tackle']  # random forest auto - feature selection
x_list = ['Acceleration', 'Heading_Accuracy', 'Defensive_Awareness', 'Vision', 
          'Volleys', 'Sprint_Speed', 'Long_Passing', 'Positioning', 'Standing_Tackle', 
          'Dribbling', 'FK_Accuracy', 'Short_Passing', 'Interceptions', 'Penalties', 
          'Finishing', 'Reactions', 'Ball_Control', 'Stamina', 
          'Crossing', 'Strength', 'Shot_Power', 'Sliding_Tackle']
print(x_list)
X = player[x_list]
y = player["value"]
len(x_list)

# %%
# 划分训练集 train, test : 0.8, 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
# X_validation, X_test, y_validation, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=23)

# %%
def calculate_metrics(model, best_params, X_train, y_train, X_test, y_test, path):
    # 在训练集上进行 k 折交叉验证并输出每次得分
    cv_scores_rmse = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_scores_rmse = np.sqrt(-cv_scores_rmse)  # 转换为正的 RMSE 值
    for i, score in enumerate(cv_scores_rmse):
        print(f'RMSE for Fold {i+1}: {score}')

    # 计算平均 RMSE 值
    mean_rmse = np.mean(cv_scores_rmse)
    print('Mean RMSE: ', mean_rmse)

    # 在训练集上进行 k 折交叉验证并输出每次r2得分
    cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5)
    for i, score in enumerate(cv_scores_r2):
        print(f'Score for Fold {i+1}: {score}')
    # 计算平均r2得分
    mean_score_r2 = np.mean(cv_scores_r2)
    print('Mean R^2: ', mean_score_r2)

    # 在测试集上评估模型并计算 RMSE 值和 R^2 值
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    # 将结果保存到文件
    with open(path, 'w') as f:
        # Best parameters
        f.write('Best parameters: ' + str(best_params) + '\n')
        # RMSE 
        f.write('RMSE for each fold:\n')
        for i, score in enumerate(cv_scores_rmse):
            f.write(f'Fold {i+1}: {score}\n')
        f.write('Mean RMSE: ' + str(mean_rmse) + '\n')
        # R^2
        f.write('R^2 for each fold:\n')
        for i, score in enumerate(cv_scores_r2):
            f.write(f'Fold {i+1}: {score}\n')
        f.write('Mean R^2: ' + str(mean_score_r2) + '\n')
        # Test set
        f.write('RMSE on test set: ' + str(test_rmse) + '\n')
        f.write('R^2 on test set: ' + str(test_r2) + '\n')

    print('Results saved to model_scores.txt')

# %%
# XGBoost
import xgboost as xgb
model = xgb.XGBRegressor()
params = {'learning_rate': [0.1, 0.2, 0.3], 
          'max_depth': np.arange(3, 15, 2), 
        #   'min_child_weight' : np.arange(3, 15, 2)
          }

grid_search = GridSearchCV(model, params, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 将结果保存到文件
path = "/mnt/d/桌面/英文文章标准档案/5. Table/xgb_results.txt"
# 获取最佳参数和最优模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
calculate_metrics(best_model, best_params, X_train, y_train, X_test, y_test, path)

# %%
import lightgbm as lgb
model = lgb.LGBMRegressor()
params = {'n_estimators': np.arange(100, 1100, 200),
            'max_depth': np.arange(3, 16, 4),
            'subsample': [0.7, 0.8, 0.9, 1],
            'colsample_bytree': [0.7, 0.8, 0.9, 1],
            'learning_rate': [1e-5, 1e-3, 1e-1],
            'num_leaves': np.arange(10, 110, 20)}

grid_search = GridSearchCV(model, params, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 获取最佳参数和最优模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
path = "/mnt/d/桌面/英文文章标准档案/5. Table/lgb_results.txt"
calculate_metrics(best_model, best_params, X_train, y_train, X_test, y_test, path)

# %%
# 随机森林
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
params = {'criterion': ['gini', 'entropy'],
          'max_depth': np.arange(3, 16, 2),
          'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_leaf': np.arange(3, 15, 2),
          'min_samples_split': np.arange(3, 15, 2),
           'n_estimators' : np.arange(100, 1100, 100) }

grid_search = GridSearchCV(model, params, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 获取最佳参数和最优模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
path = "/mnt/d/桌面/英文文章标准档案/5. Table/rf_results.txt"
calculate_metrics(best_model, best_params, X_train, y_train, X_test, y_test, path)

# %% [markdown]
# ----------------random forest starts--------------------
# [Parallel(n_jobs=-1)]: Done 7560 out of 7560 | elapsed:  3.0min finished
# ('Best parameters: ', {'max_features': 'auto', 'min_samples_split': 5, 'max_depth': 13, 'min_samples_leaf': 3})
# ('Best cross-validation score: ', 0.8485611469873839)
# ('Score on test set: ', 0.8261229514037156)
# ----------------random forest ends--------------------

# %%
# GBDT
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
params = {'n_estimators': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'max_depth': [1, 2, 3, 5, 7, 10],
            # 'min_samples_split': np.arange(3, 16, 2),
            # 'min_samples_leaf': np.arange(3, 16, 2),
            # 'subsample': np.arange(0.7, 1, 0.05),
            'learning_rate': [1e-2, 1e-1]
            }
grid_search = GridSearchCV(model, params, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 获取最佳参数和最优模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
# 将结果保存到文件
path = "/mnt/d/桌面/英文文章标准档案/5. Table/gbdt_results.txt"

calculate_metrics(best_model, best_params, X_train, y_train, X_test, y_test, path)


# %% [markdown]
# ----------------GBDT starts--------------------
# Fitting 10 folds for each of 132 candidates, totalling 1320 fits
# [Parallel(n_jobs=-1)]: Done 132 tasks      | elapsed:    5.1s
# [Parallel(n_jobs=-1)]: Done 437 tasks      | elapsed:   44.5s
# [Parallel(n_jobs=-1)]: Done 787 tasks      | elapsed:  4.7min
# [Parallel(n_jobs=-1)]: Done 1320 out of 1320 | elapsed: 11.2min finished
# ('Best parameters: ', {'n_estimators': 800, 'learning_rate': 0.1, 'max_depth': 5})
# ('Best cross-validation score: ', 0.8841836610071835)
# ('Score on test set: ', 0.8927505622156233)
# ----------------GBDT ends--------------------

# %%
# Adaboost
from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
params = {
    'n_estimators': np.arange(50, 1100, 100),
    'learning_rate': [0.1, 0.5, 1],
    'loss': ['linear', 'square', 'exponential']
}
grid_search = GridSearchCV(model, params, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 获取最佳参数和最优模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
# 将结果保存到文件
path = "/mnt/d/桌面/英文文章标准档案/5. Table/adaboost_results.txt"

calculate_metrics(best_model, best_params, X_train, y_train, X_test, y_test, path)


# %% [markdown]
# ----------------Adaboost starts--------------------
# Fitting 10 folds for each of 99 candidates, totalling 990 fits
# [Parallel(n_jobs=-1)]: Done 104 tasks      | elapsed:   25.8s
# [Parallel(n_jobs=-1)]: Done 354 tasks      | elapsed:  1.5min
# [Parallel(n_jobs=-1)]: Done 704 tasks      | elapsed:  2.7min
# [Parallel(n_jobs=-1)]: Done 990 out of 990 | elapsed:  3.8min finished
# ('Best parameters: ', {'n_estimators': 150, 'loss': 'square', 'learning_rate': 0.1})
# ('Best cross-validation score: ', 0.7726255786280866)
# ('Score on test set: ', 0.7546916447363856)
# ----------------Adaboost ends--------------------

# %%
# Catboost
import catboost
model = catboost.CatBoostRegressor()
params = {'depth' : [16],
            # 'iterations' : [500, 1000, 1500],
            # 'learning_rate' : [0.01, 0.02, 0.03],
            # 'subsample' : [0.7, 0.8, 0.9, 1]
            }

grid_search = GridSearchCV(model, params, cv=5, verbose=2, n_jobs=4)
grid_search.fit(X_train, y_train)

# 将结果保存到文件
path = "/mnt/d/桌面/英文文章标准档案/5. Table/catboost_results.txt"
# 获取最佳参数和最优模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
calculate_metrics(best_model, best_params, X_train, y_train, X_test, y_test, path)


# %%
# # Adaboost
# from sklearn.ensemble import AdaBoostRegressor
# i = 10
# clf = AdaBoostRegressor(n_estimators=i)
# clf.fit(X_train, y_train)
# print("R", clf.score(X_validation, y_validation))
# print("RMSE", np.sqrt(mean_squared_error(clf.predict(X_validation), y_validation)))
# print("MAE", mean_absolute_error(clf.predict(X_validation), y_validation))
# print("RMSLE", mean_squared_log_error(clf.predict(X_validation), y_validation))

# %%
# list_adaboost # adaboost: 10, 0.6945326685800794
# n_list_lbm = np.array(list_lightgbm) # leaves, depth, n, score : 31, 19, 500, 0.8007898902056965
# list_randomforest # [80, 0.7839175337533079]
# list_gbdt # [n = 600, 0.8264955831199144]
# catboost  defalt: 0.8027687798692258
# list_xgboost # [60, 20, 0.7812782293507701]

# %% [markdown]
# # Adaboost
#  0.7678737783884331 R    
# 
#  4334351.772382652 RMSE 
# 
# 2034372.1610311233 MAE   
# 
#  1.4164760588279637 RMSLE
# 
# # Xgboost
# 0.8868637133128766 R
# 
# 3025958.826846494 RMSE 
# 
# 923966.9140471614 MAE 
# 
# 0.15334408868795865 RMSLE
# 
# # lgb
# 0.8948018678333571 R 
# 
# 2917870.8720710855 RMSE 
# 
# 863827.241753701 MAE 
# 
# # 随机森林
# 0.8810732244650263 R 
# 
# 3102429.1779398853 RMSE 
# 
# 971012.3118455497 MAE 
# 
# 0.15934332311326996 RMSLE 
# 
# # GBDT
# 0.900796741780507 R 
# 
# 2833511.7791834464 RMSE 
# 
# 948986.9158787137 MAE 
# 
# # Catboost
# 0.8959925992883182 R 
# 
# 2901310.272421131 RMSE 
# 
# 879135.1904306137 MAE 


