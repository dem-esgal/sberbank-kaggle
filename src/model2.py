from sklearn import model_selection, preprocessing
import xgboost as xgb
N_THREAD = 4


def predict(train,test):
    id_test = test.id

    mult = .969

    y_train = train["price_doc"] * mult + 10
    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)

    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values))
            x_train[c] = lbl.transform(list(x_train[c].values))

    for c in x_test.columns:
        if x_test[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_test[c].values))
            x_test[c] = lbl.transform(list(x_test[c].values))

    xgb_params = {
        'nthread': N_THREAD,
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)

    num_boost_rounds = 385  # This was the CV output, as earlier version shows
    #cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=200,
    #                   verbose_eval=10, show_stdv=False)

    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

    y_predict = model.predict(dtest)
    return y_predict
