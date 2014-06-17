from jug import TaskGenerator
import itertools
import pandas
import asset_ification as af


#def get_coeffs(xs, ys):
#    xs = xs - xs.mean()
#    ys = ys - ys.mean()
#    return xs.apply(lambda x: x.cov(ys)/x.var())

@TaskGenerator
def gen_models(xs, ys, i):
    best_d = {'x_vars': [], 'r2_adj': 0.0}
    models = itertools.combinations(xs.columns, i)
    for model in models:
        reg =  pandas.ols(x = xs[list(model)], y = ys, intercept = True)
        if reg.r2_adj > best_d['r2_adj']:
            best_d['x_vars'] = list(model)
            best_d['r2_adj'] = reg.r2_adj

    return best_d

store = pandas.HDFStore('../dat/ac.h5', 'r')
acs_df, not_acs = af.load_data(store)
ticker = store.get(not_acs[-1])['Adj Close']
store.close()
ind = af.clean_dates(acs_df, ticker)
xs = acs_df.loc[ind, :].pct_change().dropna()
ys = ticker[ind].pct_change().dropna()
best_models = map(lambda x: gen_models(xs, ys, x), xrange(1, xs.shape[1]))
