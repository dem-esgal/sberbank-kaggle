import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing
import xgboost as xgb
MILLION_W = 0.7
MILLION2_W = 0.3
MILLION3_W = 0.2
N_THREAD = 4

def predict(train, test, y_target=None):
    # Add month-year
    month_year = (train.timestamp.dt.month * 30 + train.timestamp.dt.year * 365)
    month_year_cnt_map = month_year.value_counts().to_dict()
    train['month_year_cnt'] = month_year.map(month_year_cnt_map)

    month_year = (test.timestamp.dt.month * 30 + test.timestamp.dt.year * 365)
    month_year_cnt_map = month_year.value_counts().to_dict()
    test['month_year_cnt'] = month_year.map(month_year_cnt_map)

    # Add week-year count
    week_year = (train.timestamp.dt.weekofyear * 7 + train.timestamp.dt.year * 365)
    week_year_cnt_map = week_year.value_counts().to_dict()
    train['week_year_cnt'] = week_year.map(week_year_cnt_map)

    week_year = (test.timestamp.dt.weekofyear * 7 + test.timestamp.dt.year * 365)
    week_year_cnt_map = week_year.value_counts().to_dict()
    test['week_year_cnt'] = week_year.map(week_year_cnt_map)

    # Add month and day-of-week
    train['month'] = train.timestamp.dt.month
    train['dow'] = train.timestamp.dt.dayofweek

    test['month'] = test.timestamp.dt.month
    test['dow'] = test.timestamp.dt.dayofweek

    # Other feature engineering
    train['rel_floor'] = 0.05 + train['floor'] / train['max_floor'].astype(float)
    train['rel_kitch_sq'] = 0.05 + train['kitch_sq'] / train['full_sq'].astype(float)

    test['rel_floor'] = 0.05 + test['floor'] / test['max_floor'].astype(float)
    test['rel_kitch_sq'] = 0.05 + test['kitch_sq'] / test['full_sq'].astype(float)
    #0.328934

    #train.apartment_name = train.sub_area + train['metro_min_rn'].astype(str)
    #test.apartment_name = test.sub_area + train['metro_min_rn'].astype(str)

    train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
    test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

    train['area_per_room'] = train['life_sq'] / train['num_room'].astype(float)  # rough area per room
    train['livArea_ratio'] = train['life_sq'] / train['full_sq'].astype(float)  # rough living area
    train['gender_ratio'] = train['male_f'] / train['female_f'].astype(float)
    train['lifesq_x_state'] = train['life_sq'] * train['state'].astype(float)  # life_sq times the state of the place
    train['floor_x_state'] = train['floor'] * train['state'].astype(float)  # relative floor * the state of the place

    test['area_per_room'] = test['life_sq'] / test['num_room'].astype(float)
    test['livArea_ratio'] = test['life_sq'] / test['full_sq'].astype(float)
    test['gender_ratio'] = test['male_f'] / test['female_f'].astype(float)
    test['lifesq_x_state'] = test['life_sq'] * test['state'].astype(float)
    test['floor_x_state'] = test['floor'] * test['state'].astype(float)
    #########################################################################################################

    train['price_doc'] = train['price_doc']
    y_train = train["price_doc"]

    #########################################################################################################

    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    # x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)

    num_train = len(x_train)
    x_all = pd.concat([x_train, x_test])

    for c in x_all.columns:
        if x_all[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_all[c].values))
            x_all[c] = lbl.transform(list(x_all[c].values))

    x_train = x_all[:num_train]
    x_test = x_all[num_train:]

    xgb_params = {
        'nthread': N_THREAD,
        'eta': 0.02,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 1,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'seed':420,
    }
    is_avoider = np.logical_and((y_train > 990000), (y_train <= 1e6))
    wts = 1 - MILLION_W * (is_avoider) - MILLION2_W * (y_train == 2e6) - MILLION3_W * (y_train == 3e6)
    num_boost_rounds = 700  # 500

    y_train = np.log(y_train+1)
    if y_target is not None:
        y_target = np.log(y_target+1)
        dtrain = xgb.DMatrix(x_train, y_train, weight=wts)
        dtest = xgb.DMatrix(x_test, y_target)
        model = xgb.train(dict(xgb_params, silent=1), dtrain, evals= [(dtrain, 'train'), (dtest, 'eval')], num_boost_round=num_boost_rounds)
    else:
        dtrain = xgb.DMatrix(x_train, y_train, weight=wts)
        dtest = xgb.DMatrix(x_test)
        model = xgb.train(dict(xgb_params, silent=1), dtrain,num_boost_round=num_boost_rounds)

    #watchlist =

    #cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=200,   verbose_eval=10, show_stdv=False)

    #y_predict = model.predict(dtest)
    y_predict = np.exp(model.predict(dtest))-1
    return y_predict
