import numpy as np
from sklearn.cross_validation import train_test_split
import scipy as sp

def clearData(train, test):
    # clean data
    bad_index = train[train.kremlin_km<0.08].index
    train.loc[bad_index, "kremlin_km"] = np.NaN
    bad_index = test[test.kremlin_km<0.08].index
    test.loc[bad_index, "kremlin_km"] = np.NaN

    bad_index = train[train.life_sq > train.full_sq].index
    train.loc[bad_index, "life_sq"] = np.NaN
    equal_index = [601, 1896, 2791]
    test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]
    bad_index = test[test.life_sq > test.full_sq].index
    test.loc[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.life_sq < 5].index
    train.loc[bad_index, "life_sq"] = np.NaN
    bad_index = test[test.life_sq < 5].index
    test.loc[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.full_sq < 5].index
    train.loc[bad_index, "full_sq"] = np.NaN
    bad_index = test[test.full_sq < 5].index
    test.loc[bad_index, "full_sq"] = np.NaN
    kitch_is_build_year = [13117]
    train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]
    bad_index = train[train.kitch_sq >= train.life_sq].index
    train.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[test.kitch_sq >= test.life_sq].index
    test.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
    train.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
    test.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
    train.loc[bad_index, "full_sq"] = np.NaN
    bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
    test.loc[bad_index, "full_sq"] = np.NaN
    bad_index = train[train.life_sq > 300].index
    train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
    bad_index = test[test.life_sq > 200].index
    test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
    train.product_type.value_counts(normalize=True)
    test.product_type.value_counts(normalize=True)
    bad_index = train[train.build_year < 1500].index
    train.loc[bad_index, "build_year"] = np.NaN
    bad_index = test[test.build_year < 1500].index
    test.loc[bad_index, "build_year"] = np.NaN
    bad_index = train[train.num_room == 0].index
    train.loc[bad_index, "num_room"] = np.NaN
    bad_index = test[test.num_room == 0].index
    test.loc[bad_index, "num_room"] = np.NaN
    bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
    train.loc[bad_index, "num_room"] = np.NaN
    bad_index = [3174, 7313]
    test.loc[bad_index, "num_room"] = np.NaN
    bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
    train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
    bad_index = train[train.floor == 0].index
    train.loc[bad_index, "floor"] = np.NaN
    bad_index = train[train.max_floor == 0].index
    train.loc[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.max_floor == 0].index
    test.loc[bad_index, "max_floor"] = np.NaN
    bad_index = train[train.floor > train.max_floor].index
    train.loc[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.floor > test.max_floor].index
    test.loc[bad_index, "max_floor"] = np.NaN
    train.floor.describe(percentiles=[0.9999])
    bad_index = [23584]
    train.loc[bad_index, "floor"] = np.NaN
    train.material.value_counts()
    test.material.value_counts()
    train.state.value_counts()
    bad_index = train[train.state == 33].index
    train.loc[bad_index, "state"] = np.NaN
    test.state.value_counts()

    # brings error down a lot by removing extreme price per sqm
    train.loc[train.full_sq == 0, 'full_sq'] = 50
    train = train[train.price_doc / train.full_sq <= 600000]
    train = train[train.price_doc / train.full_sq >= 10000]
    return train, test

def scale_miss(  # Scale shifted logs and compare raw stdev to old raw stdev
        alpha,
        shifted_logs,
        oldstd,
        new_logmean
):
    newlogs = new_logmean + alpha * (shifted_logs - new_logmean)
    newstd = np.std(np.exp(newlogs))
    return (newstd - oldstd) ** 2


def shift_logmean_but_keep_scale(  # Or change the scale, but relative to the old scale
        data,
        new_logmean,
        rescaler
):
    logdata = np.log(data)
    oldstd = data.std()
    shift = new_logmean - logdata.mean()
    shifted_logs = logdata + shift
    scale = sp.optimize.leastsq(scale_miss, 1, args=(shifted_logs, oldstd, new_logmean))
    alpha = scale[0][0]
    newlogs = new_logmean + rescaler * alpha * (shifted_logs - new_logmean)
    return np.exp(newlogs)

def split(T):
    #T1, T2 = train_test_split(T, random_state=420, train_size=0.8 )
    #return T1, T2
    T["yearmonth"] = T["timestamp"].dt.year * 100 + T["timestamp"].dt.month
    val_time = 201407
    dev_indices = np.where(T["yearmonth"] < val_time)
    val_indices = np.where(T["yearmonth"] >= val_time)
    return T.loc[dev_indices], T.loc[val_indices]


def rmsle(h, y):
    return np.sqrt(np.square(np.log(h+1) - np.log(y+1)).mean())
