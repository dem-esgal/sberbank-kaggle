import pandas as pd
import numpy as np
import statsmodels.api as sm

micro_humility_factor = 0.96     #    range from 0 (complete humility) to 1 (no humility)
macro_humility_factor = 1

macro = pd.read_csv('../input/macro.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Macro data monthly medians
macro["timestamp"] = pd.to_datetime(macro["timestamp"])
macro["year"] = macro["timestamp"].dt.year
macro["month"] = macro["timestamp"].dt.month
macro["yearmonth"] = 100 * macro.year + macro.month
macmeds = macro.groupby("yearmonth").median()

# Price data monthly medians
train["timestamp"] = pd.to_datetime(train["timestamp"])
train["year"] = train["timestamp"].dt.year
train["month"] = train["timestamp"].dt.month
train["yearmonth"] = 100 * train.year + train.month
prices = train[["yearmonth", "price_doc"]]
p = prices.groupby("yearmonth").median()

# Join monthly prices to macro data
df = macmeds.join(p)

import numpy.matlib as ml


def almonZmatrix(X, maxlag, maxdeg):
    """
    Creates the Z matrix corresponding to vector X.
    """
    n = len(X)
    Z = ml.zeros((len(X) - maxlag, maxdeg + 1))
    for t in range(maxlag, n):
        # Solve for Z[t][0].
        Z[t - maxlag, 0] = sum([X[t - lag] for lag in range(maxlag + 1)])
        for j in range(1, maxdeg + 1):
            s = 0.0
            for i in range(1, maxlag + 1):
                s += (i) ** j * X[t - i]
            Z[t - maxlag, j] = s
    return Z


y = df.price_doc.div(df.cpi).apply(np.log).loc[201108:201506]
lncpi = df.cpi.apply(np.log)
tblags = 5  # Number of lags used on PDL for Trade Balance
mrlags = 5  # Number of lags used on PDL for Mortgage Rate
cplags = 5  # Number of lags used on PDL for CPI
ztb = almonZmatrix(df.balance_trade.loc[201103:201506].as_matrix(), tblags, 1)
zmr = almonZmatrix(df.mortgage_rate.loc[201103:201506].as_matrix(), mrlags, 1)
zcp = almonZmatrix(lncpi.loc[201103:201506].as_matrix(), cplags, 1)
columns = ['tb0', 'tb1', 'mr0', 'mr1', 'cp0', 'cp1']
z = pd.DataFrame(np.concatenate((ztb, zmr, zcp), axis=1), y.index.values, columns)
X = sm.add_constant(z)

# Fit macro model
eq = sm.OLS(y, X)
fit = eq.fit()

# Predict with macro model
test_cpi = df.cpi.loc[201507:201605]
test_index = test_cpi.index
ztb_test = almonZmatrix(df.balance_trade.loc[201502:201605].as_matrix(), tblags, 1)
zmr_test = almonZmatrix(df.mortgage_rate.loc[201502:201605].as_matrix(), mrlags, 1)
zcp_test = almonZmatrix(lncpi.loc[201502:201605].as_matrix(), cplags, 1)
z_test = pd.DataFrame(np.concatenate((ztb_test, zmr_test, zcp_test), axis=1),
                      test_index, columns)
X_test = sm.add_constant(z_test)
pred_lnrp = fit.predict(X_test)
pred_p = np.exp(pred_lnrp) * test_cpi

# Merge with test cases and compute mean for macro prediction
test["timestamp"] = pd.to_datetime(test["timestamp"])
test["year"] = test["timestamp"].dt.year
test["month"] = test["timestamp"].dt.month
test["yearmonth"] = 100 * test.year + test.month
test_ids = test[["yearmonth", "id"]]
monthprices = pd.DataFrame({"yearmonth": pred_p.index.values, "monthprice": pred_p.values})
macro_mean = np.exp(test_ids.merge(monthprices, on="yearmonth").monthprice.apply(np.log).mean())
print(macro_mean)

naive_pred_lnrp = y.mean()
naive_pred_p = np.exp(naive_pred_lnrp) * test_cpi
monthnaive = pd.DataFrame({"yearmonth": pred_p.index.values, "monthprice": naive_pred_p.values})
macro_naive = np.exp(test_ids.merge(monthnaive, on="yearmonth").monthprice.apply(np.log).mean())
print(macro_naive)

macro_mean = macro_naive * (macro_mean / macro_naive) ** macro_humility_factor
print(macro_mean)
