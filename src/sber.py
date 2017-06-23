# thank's to https://www.kaggle.com/aharless for code sharing
# https://www.kaggle.com/aharless/jiwon-small-improvements-for-magic-number-results/versions

# Parameters
prediction_stderr = 0.008  # assumed standard error of predictions
#  (smaller values make output closer to input)
train_test_logmean_diff = 0.1  # assumed shift used to adjust frequencies for time trend
probthresh = 80  # minimum probability*frequency to use new price instead of just rounding
rounder = 2  # number of places left of decimal point to zero

import numpy as np
import pandas as pd

# load files
import src.utils as utils
import src.model1 as model1
from sklearn.cross_validation import train_test_split

def split(T):
    a_train, a_test = train_test_split(T, test_size=0.2, random_state=42)
    return a_train, a_test

def rmsle(h, y):
    return np.sqrt(np.square(np.log(h+1) - np.log(y+1)).mean())

train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
fx = pd.read_excel('../input/BAD_ADDRESS_FIX.xlsx').drop_duplicates('id').set_index('id')

train.update(fx)
test.update(fx)

id_test = test.id
train, test = utils.clearData(train, test)

train_v, test_v = split(train)

id_test = test_v.id
y_target = pd.DataFrame({'id': id_test, 'price_doc': test_v.price_doc})

print(test_v.shape)
test_v.drop(['price_doc'], axis=1, inplace=True)
y_predict = model1.predict(train_v, test_v, y_target.price_doc)

gunja_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
######################################################################################################
import src.model2 as model2
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train, test = utils.clearData(train, test)

train_v, test_v = split(train)
print(test_v.shape)
test_v.drop(['price_doc'], axis=1, inplace=True)
y_predict = model2.predict(train_v, test_v, y_target.price_doc)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
#######################################################################################################

import src.model3 as model3
df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])
df_train, df_test = utils.clearData(df_train, df_test)

df_train_v, df_test_v = split(df_train)
df_test_v.drop(['price_doc'], axis=1, inplace=True)

y_pred = model3.predict(train_v, test_v, df_train_v, df_test_v, y_target.price_doc, df_macro)
print(rmsle(y_pred, y_target.price_doc))

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

first_result = output.merge(df_sub, on="id", suffixes=['_louis', '_bruno'])
first_result["price_doc"] = np.exp(.714 * np.log(first_result.price_doc_louis) +
                                   .286 * np.log(first_result.price_doc_bruno))

result = first_result.merge(gunja_output, on="id", suffixes=['_follow', '_gunja'])

result["price_doc"] = np.exp(.78 * np.log(result.price_doc_follow) +
                             .22 * np.log(result.price_doc_gunja))

result["price_doc"] = result["price_doc"] * 0.9915
result.drop(["price_doc_louis", "price_doc_bruno", "price_doc_follow", "price_doc_gunja"], axis=1, inplace=True)
result.head()
result.to_csv('same_result.csv', index=False)

result = result.merge(y_target, on="id", suffixes=['_1', '_2'])

print(rmsle(result.price_doc_1, result.price_doc_2))
#0.457333070579