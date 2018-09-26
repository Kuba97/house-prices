from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from mlens.ensemble import SuperLearner


ridge = Ridge(alpha=10)
elastic_net = ElasticNet(alpha=0.001, l1_ratio=.5, max_iter=256)
xgboost = XGBRegressor(max_depth=3,
                       n_estimators=1024,
                       learning_rate=0.1,
                       colsample_bytree=0.6,
                       colsample_bylevel=0.4,
                       reg_alpha=0.2,
                       reg_lambda=0.8,
                       n_jobs=4)

light_gbm = LGBMRegressor(num_leaves=8,
                          max_depth=5,
                          n_estimators=256,
                          colsample_bytree=0.2,
                          min_child_samples=1)


def get_super_learner():
    base_learners = [elastic_net, xgboost, light_gbm]
    meta_learner = LinearRegression(fit_intercept=False)
    ensemble = SuperLearner(folds=2, shuffle=False)
    ensemble.add(base_learners)
    ensemble.add_meta(meta_learner)
    return ensemble


super_learner = get_super_learner()
