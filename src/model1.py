import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
N_THREAD = 4

def predict(train,test):

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
    train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
    train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

    test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
    test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

    train.apartment_name = train.sub_area + train['metro_km_avto'].astype(str)
    test.apartment_name = test.sub_area + train['metro_km_avto'].astype(str)

    train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
    test['room_size'] = test['life_sq'] / test['num_room'].astype(float)


    rate_2015_q2 = 1
    rate_2015_q1 = rate_2015_q2 / 0.9932
    rate_2014_q4 = rate_2015_q1 / 1.0112
    rate_2014_q3 = rate_2014_q4 / 1.0169
    rate_2014_q2 = rate_2014_q3 / 1.0086
    rate_2014_q1 = rate_2014_q2 / 1.0126
    rate_2013_q4 = rate_2014_q1 / 0.9902
    rate_2013_q3 = rate_2013_q4 / 1.0041
    rate_2013_q2 = rate_2013_q3 / 1.0044
    rate_2013_q1 = rate_2013_q2 / 1.0104  # This is 1.002 (relative to mult), close to 1:
    rate_2012_q4 = rate_2013_q1 / 0.9832  # maybe use 2013q1 as a base quarter and get rid of mult?
    rate_2012_q3 = rate_2012_q4 / 1.0277
    rate_2012_q2 = rate_2012_q3 / 1.0279
    rate_2012_q1 = rate_2012_q2 / 1.0279
    rate_2011_q4 = rate_2012_q1 / 1.076
    rate_2011_q3 = rate_2011_q4 / 1.0236
    rate_2011_q2 = rate_2011_q3 / 1
    rate_2011_q1 = rate_2011_q2 / 1.011

    # train 2015
    train['average_q_price'] = 1

    train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[
        train['timestamp'].dt.month < 7].index
    train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2

    train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[
        train['timestamp'].dt.month < 4].index
    train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1

    # train 2014
    train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[
        train['timestamp'].dt.month <= 12].index
    train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

    train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[
        train['timestamp'].dt.month < 10].index
    train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

    train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[
        train['timestamp'].dt.month < 7].index
    train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

    train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[
        train['timestamp'].dt.month < 4].index
    train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1

    # train 2013
    train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[
        train['timestamp'].dt.month <= 12].index
    train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

    train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[
        train['timestamp'].dt.month < 10].index
    train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

    train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[
        train['timestamp'].dt.month < 7].index
    train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

    train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[
        train['timestamp'].dt.month < 4].index
    train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1

    # train 2012
    train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[
        train['timestamp'].dt.month <= 12].index
    train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

    train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[
        train['timestamp'].dt.month < 10].index
    train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

    train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[
        train['timestamp'].dt.month < 7].index
    train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

    train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[
        train['timestamp'].dt.month < 4].index
    train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1

    # train 2011
    train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[
        train['timestamp'].dt.month <= 12].index
    train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

    train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[
        train['timestamp'].dt.month < 10].index
    train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

    train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[
        train['timestamp'].dt.month < 7].index
    train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

    train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[
        train['timestamp'].dt.month < 4].index
    train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

    train['price_doc'] = train['price_doc'] * train['average_q_price']

    #########################################################################################################

    mult = 1.054880504

    train['price_doc'] = train['price_doc'] * mult
    y_train = train["price_doc"]

    #########################################################################################################

    x_train = train.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
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
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.6,
        'colsample_bytree': 1,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)

    num_boost_rounds = 422#500
    #cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=200,
    #                   verbose_eval=10, show_stdv=False)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

    y_predict = model.predict(dtest)
    return y_predict
