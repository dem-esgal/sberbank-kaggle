# thank's to https://www.kaggle.com/aharless for code sharing
# https://www.kaggle.com/aharless/jiwon-small-improvements-for-magic-number-results/versions

# Parameters
prediction_stderr = 0.008  # assumed standard error of predictions
#  (smaller values make output closer to input)
train_test_logmean_diff = 0.1  # assumed shift used to adjust frequencies for time trend
probthresh = 80  # minimum probability*frequency to use new price instead of just rounding
rounder = 2  # number of places left of decimal point to zero

jason_weight = .2
bruno_weight = .2
reynaldo_weight = 1 - jason_weight - bruno_weight

import numpy as np
import pandas as pd

# load files
import src.utils as utils
import src.model1 as model1

train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

id_test = test.id
train, test = utils.clearData(train, test)
'''
train, test = utils.split(train)
y_target = pd.DataFrame({'id': test.id, 'price_doc': test.price_doc})

test_v_i = test[test["product_type"] == "Investment"]
test_v_ni = test[test["product_type"] != "Investment"]

million_1 = test_v_i[(test_v_i.price_doc > 990000) & (test_v_i.price_doc <= 1010000)]
UNDERSAMPLE = 3

million_1 = million_1.sample(round(UNDERSAMPLE*million_1.shape[0]/10))
million_2 = test_v_i[test_v_i.price_doc == 2e6]
million_2 = million_2.sample(round(UNDERSAMPLE*million_2.shape[0]/10))
million_3 = test_v_i[test_v_i.price_doc == 3e6]
million_3 = million_3.sample(round(UNDERSAMPLE*million_3.shape[0]/10))
nonmillion= test_v_i[((test_v_i.price_doc <= 990000) | (test_v_i.price_doc > 1010000)) & (test_v_i.price_doc != 2e6) & (test_v_i.price_doc != 3e6)]
#nonmillion = nonmillion.sample(round(6.6*nonmillion.shape[0]/10))

test_v_i = pd.concat([million_1, million_2, million_3, nonmillion], axis=0)

test = pd.concat([test_v_ni, test_v_i])
test = test.dropna(subset=['price_doc'])
train = train.dropna(subset=['price_doc'])

y_target = pd.DataFrame({'id': test.id, 'price_doc': test.price_doc})
test.drop(['price_doc'], axis=1, inplace=True)
print(test.shape)
'''
#y_target = pd.DataFrame({'id': test.id, 'price_doc': test.price_doc})

#train = train.dropna(subset=['price_doc'])

#train_ni = train[train["product_type"] != "Investment"]
#test_ni = test[test["product_type"] != "Investment"]

#train_i = train[train["product_type"] == "Investment"]
#test_i = test[test["product_type"] == "Investment"]
#test.drop(['price_doc'], axis=1, inplace=True)


#y_target_i = pd.DataFrame({'id': test_i.id, 'price_doc': test_i.price_doc})
#test_i.drop(['price_doc'], axis=1, inplace=True)
#y_predict = modeli.predict(train_i, test_i)
#part2 = pd.DataFrame({'id': test_i.id, 'price_doc': y_predict})
#print(utils.rmsle(part2.price_doc, y_target_i.price_doc))

#y_target_ni = pd.DataFrame({'id': test_ni.id, 'price_doc': test_ni.price_doc})
#test_ni.drop(['price_doc'], axis=1, inplace=True)
#y_predict = modelni.predict(train_ni, test_ni)
#part1 = pd.DataFrame({'id': test_ni.id, 'price_doc': y_predict})
#print(utils.rmsle(part1.price_doc, y_target_ni.price_doc))

#0.128526336088
#0.579393381433

#result = pd.concat([part1, part2])
#print(utils.rmsle(result.price_doc, y_target.price_doc))

#0.177036975482
#0.556572415928

y_predict = model1.predict(train, test)
result = pd.DataFrame({'id': test.id, 'price_doc': y_predict})

#result = pd.DataFrame({'id': test.id,'pt':test.product_type, 'price_doc': y_predict})

#result_i = result[result["pt"] == "Investment"]
#result_ni = result[result["pt"] != "Investment"]

#print(utils.rmsle(result_ni.price_doc, y_target_ni.price_doc))
#print(utils.rmsle(result_i.price_doc, y_target_i.price_doc))

#print(utils.rmsle(result.price_doc, y_target.price_doc))
#.032 - result
#result.to_csv('result.csv', index=False)
# - model1 0.31684
result.to_csv('result_yar.csv', index=False)
#0.   43
'''
y_predict = model1.predict(train, test)


gunja_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
gunja_output.to_csv('gunja_output2.csv', index=False)
'''
######################################################################################################
'''import src.model2 as model2
train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
train, test = utils.clearData(train, test)

train_v, test_v = utils.split(train)
print(test_v.shape)
test_v.drop(['price_doc'], axis=1, inplace=True)
y_predict = model2.predict(train_v, test_v, y_target.price_doc)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
'''
#######################################################################################################
'''
import src.model3 as model3
df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])
df_train, df_test = utils.clearData(df_train, df_test)

df_train_v, df_test_v = utils.split(df_train)
df_test_v.drop(['price_doc'], axis=1, inplace=True)

y_pred = model3.predict(train_v, test_v, df_train_v, df_test_v, y_target.price_doc, df_macro)
print(utils.rmsle(y_pred, y_target.price_doc))

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

print(utils.rmsle(result.price_doc_1, result.price_doc_2))
#0.457333070579
#0.457463109738

first_result = output.merge(df_sub, on="id", suffixes=['_louis', '_bruno'])
first_result["price_doc"] = np.exp(.4 * np.log(first_result.price_doc_louis) + .6 * np.log(first_result.price_doc_bruno))

result = first_result.merge(gunja_output, on="id", suffixes=['_follow', '_gunja'])
result["price_doc"] = np.exp(0.7 * np.log(result.price_doc_follow) + 0.3 * np.log(result.price_doc_gunja))
result["price_doc"] = result["price_doc"] * 1

result = result.merge(y_target, on="id", suffixes=['_1', '_2'])

print(utils.rmsle(result.price_doc_1, result.price_doc_2))

#gunja_output = gunja_output.merge(y_target, on="id", suffixes=['_1', '_2'])

#print(rmsle(gunja_output.price_doc_1, gunja_output.price_doc_2))
'''