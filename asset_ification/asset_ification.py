#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: asset_ification.py

Created by Benjamin M. Gross

.. note:: General Convention for :mod:`asset_ification`

When passing around
HDFStore to and from functions, the store is left open if the function
does not complete.  For this reason (and until I figure out a better
fix) the general convention is that functions are passed
:class:`pandas.Series` of training data (tickers, and asset classes)
and :class:`str` of the paths that lead to stores, so if there is any
error with the function, the HDFStore can be closed within the exception

"""

import argparse
import datetime
import numpy
import pandas
import pandas.io.data
import os
from sklearn.svm import LinearSVC


def compare_knn_svm(ticker_list, store_path):
    """
    Yes, I'm officially losing my mind, but who isn't.  I mean, why
    **shouldn't** I compare the out of sample performance of a 338 
    dimensional svm with my knn algorithm... makes total sense, fuck off
    """

    #create categorical variables and load the data
    rsq_df = pandas.DataFrame.from_csv('../dat/rsq_matrix.csv')
    trained_series = pandas.Series.from_csv('../dat/trained_assets.csv', header = 0)
    acs = rsq_df.index.to_series().value_counts()
    ac_map = pandas.Series( numpy.arange(len(acs)), index = acs.index)
    y = numpy.array( map(lambda x: ac_map[str(x)], rsq_df.index))
    clf = LinearSVC()
    clf.fit(X = rsq_df.values, y = y)
    store = pandas.HDFStore(store_path, 'r')

    for ticker in ticker_list:

        print "Working on " + ticker + " "  + trained_series[ticker]
        series = store.get(ticker)['Adj Close']
        print "Estimate from Nearest Neighbor" + '\n'
        print lnchg_nearest_neighbors(series, store_path, trained_series)
        rsq_d = {}
        for dim in rsq_df.columns:
            x2 = store.get(dim)['Adj Close']
            ind = clean_dates(series, x2)
            rsq_d[dim] = adj_r2_uv(x = x2[ind].apply(numpy.log).diff().dropna(),
                                   y = series[ind].apply(numpy.log).diff().dropna())


        print "Estimate from SVM" + '\n'
        new_est = pandas.Series(rsq_d)
        print ac_map[ac_map == clf.predict(new_est)[0]].index

    if store.is_open:
        store.close()
    return None

def run_classification(trained_series, store_path):
    """
    See the classifications as they are run

    :ARGS:

        trained_series: :class:`pandas.Series` of tickers for the
        :class:`Index` and Asset Classes for ``.values``

        store_path: :class:`str` pointing to the HDFStore of trained
        data
    """
    etf_list = complete_etf_list()
    notin_trained = numpy.setdiff1d(etf_list.index.tolist(), 
                                    trained_series.index.tolist())
    #let 'er rip!
    for ticker in notin_trained:
        try:
            series = tickers_to_dict(ticker)['Adj Close']
            print lnchg_nearest_neighbors(series, store_path,
                                         trained_series)
        except:
            print "Didn't work for " + ticker

    return None

def model_accuracy_helper_fn(trained_series, store_path, calc_meth):
    """
    Helper function for both of the model accuracy functions

    :ARGS:

        trained_series: :class:`pandas.Series` of tickers for the
        :class:`Index` and Asset Classes for ``.values``

        store_path: :class:`str` pointing to the HDFStore of trained
        data
    """
    store = pandas.HDFStore(store_path, 'r')
    keys = map(lambda x: x.strip('/'),  store.keys())
    d = {}
    for i, key in enumerate(keys):
        print ('working on ' + key + '\n' + str(i + 1) + 
               ' of ' + str(len(keys)+ 1))
        try:
            series = store.get(key)['Adj Close']
            #exclude that ticker for testing
            not_key = trained_series.index != key
            d[key] = nn_helper_fn(series, store_path, 
                trained_series[not_key], calc_meth)
        except:
            print "Didn't work for " + key
    if store.is_open:
        store.close()
    return pandas.DataFrame(d).transpose()

def model_accuracy_crosstab(trained_series, store_path, calc_meth):
    """
    Calculate the model accuracy (returned as a confusion matrix) of 
    asset classes that are already known, provided in the 
    ``training_series``.

    :ARGS:

        store_path: :class:`string` to the HDFStore of asset 

        training_series: :class:`series` where Index is the tickers 
        and the values are the asset classes

    :RETURNS:

        :class:`pandas.DataFrame` of the confusion matrix
    """
    prob_df = model_accuracy_helper_fn(trained_series, store_path, calc_meth)
    algo_results = prob_df.apply(lambda x: x.argmax(), axis = 1)
    return pandas.crosstab(algo_results, trained_series)

def model_accuracy_prob_matrix(trained_series, store_path, calc_meth):
    """
    Return the trained tickers and their asset class probabilities

    
    :ARGS:

        trained_series: :class:`pandas.Series` of tickers for the
        :class:`Index` and Asset Classes for ``.values``

        store_path: :class:`str` pointing to the HDFStore of trained
        data
    """
    return model_accuracy_helper_fn(trained_series, store_path, calc_meth)

def plot_confusion_matrix(trained_series, store_path, calc_meth):
    """
    Plot the confusion matrix
    
    :ARGS:

        trained_series: :class:`pandas.Series` of tickers for the
        :class:`Index` and Asset Classes for ``.values``

        store_path: :class:`str` pointing to the HDFStore of trained
        data
    """
    cm = crosstab_model_accuracy(trained_series, store_path, calc_meth)
    cm_pct = cm.apply(lambda x: x/float(x.sum()), axis = 0)
    plt.imshow(cm_pct, interpolation = 'nearest')
    plt.xticks(numpy.arange(len(cm.index)), cm.index)
    plt.yticks(numpy.arange(len(cm.index)), cm.index)
    plt.colorbar()
    plt.title("Confusion Matrix for KNN Asset Classes", fontsize = 16)
    plt.show()

    
def lnchg_nearest_neighbors(price_series, store_path, trained_series):
    """
    Calculate the "nearest neighbors" on trained asset class data to 
    determine probabilities the series belongs to given asset classes
    
    :ARGS:

        series: :class:`pandas.Series` of prices
   

        store_path: :class:`string` of a store path or an actual store
        of the trained  asset class prices (each ticker should also 
        appear in the  ``.csv`` located at ``training_file_path``

        training_series: :class:`series` where Index is the tickers 
        and the values are the asset classes

    """
    return nn_helper_fn(price_series, store_path, trained_series, 'ln_chg')

def nrmdp_nearest_neighbor(price_series, store_path, trained_series):
    
    return nn_helper_fn(price_series, store_path, trained_series, 'nrmdp')

def nn_helper_fn(price_series, store_path, trained_series, calc_meth):
    """
    Helper Function to allow for different calculation values for the 
    nearest neighbor algorithm
    
    """
    fun_d = {'ln_chg':lambda x: x.apply(numpy.log).diff().dropna(),
             'nrmdp':lambda x: (x - x.mean())/x.std(),
             'na': lambda x: x,}

    ac_freq = trained_series.value_counts()
    #choose k = d + 1, where d is the number of unique asset classes
    k = len(ac_freq) + 1
    
    #if a path is provided, open the store, otherwise it IS a store

    store = pandas.HDFStore(store_path, 'r')

    prob_d = {'r2_adj': [], 'asset_class': [] }
    for ticker in trained_series.index:
        try:
            comp_asset = store.get(ticker)['Adj Close']
            ind = clean_dates(comp_asset, price_series)
            x = comp_asset[ind]
            y = price_series[ind]

            prob_d['r2_adj'].append(adj_r2_uv(
                fun_d[calc_meth](x), fun_d[calc_meth](y)))
            prob_d['asset_class'].append(trained_series[ticker])
        except:
            print "Did not work for ticker " + ticker

    prob_df = pandas.DataFrame(prob_d)
    prob_df.sort(columns = ['r2_adj'], ascending = False, inplace = True)
    store.close()
    
    #apply the weighting scheme
    df = prob_df.iloc[:k, :]
    wts = numpy.exp(df['r2_adj'])
    df['weight'] = wts/wts.sum()
    return df.groupby('asset_class').sum()['weight']
    
    
#    return prob_df['asset_class'][:k].value_counts()/float(k)

def clean_dates(arr_a, arr_b):
    """
    Return the intersection of two :class:`pandas` objects, either a
    :class:`pandas.Series` or a :class:`pandas.DataFrame`

    :ARGS:

        arr_a: :class:`pandas.DataFrame` or :class:`pandas.Series`
        arr_b: :class:`pandas.DataFrame` or :class:`pandas.Series`

    :RETURNS:

        :class:`pandas.DatetimeIndex` of the intersection of the two 
        :class:`pandas` objects
    """
    arr_a = arr_a.sort_index()
    arr_a.dropna(inplace = True)
    arr_b = arr_b.sort_index()
    arr_b.dropna(inplace = True)
    if arr_a.index.equals(arr_b.index) == False:
        return arr_a.index & arr_b.index
    else:
        return arr_a.index

def update_store_prices(store_path):
    """
    Update to the most recent prices for all keys of an existing store, 
    located at path ``path``.

    :ARGS:

        store_path: :class:`string` the location of the ``HDFStore`` file

    :RETURNS:

        :class:`NoneType` but updates the ``HDF5`` file, and prints to 
        screen which values would not update

    """
    reader = pandas.io.data.DataReader
    strftime = datetime.datetime.strftime
    
    today_str = strftime(datetime.datetime.today(), format = '%m/%d/%Y')
    try:
        store = pandas.HDFStore(path = store_path, mode = 'r+')
    except IOError:
        print  path + " is not a valid path to an HDFStore Object"
        return

    for key in store.keys():
        stored_data = store.get(key)
        last_stored_date = stored_data.dropna().index.max()
        today = datetime.datetime.date(datetime.datetime.today())
        if last_stored_date < pandas.Timestamp(today):
            try:
                tmp = reader(key.strip('/'), 'yahoo', start = strftime(
                    last_stored_date, format = '%m/%d/%Y'))

                #need to drop duplicates because there's 1 row of 
                #overlap
                tmp = stored_data.append(tmp)
                tmp["index"] = tmp.index
                tmp.drop_duplicates(cols = "index", inplace = True)
                tmp = tmp[tmp.columns[tmp.columns != "index"]]
                store.put(key, tmp)
            except IOError:
                print "could not update " + key

    store.close()
    return None

def setup_trained_hdfstore(trained_data, store_path):
    """
    The ``HDFStore`` doesn't work properly when it's compiled by different
    versions, so the appropriate thing to do is to setup the trained data
    locally (and not store the ``.h5`` file on GitHub).

    :ARGS:

        trained_data: :class:`pandas.Series` with tickers in the index and
        asset  classes for values 

        store_path: :class:`str` of where to create the ``HDFStore``
    """
    
    create_data_store(trained_data.index, store_path)
    return None

def create_data_store(ticker_list, store_path):
    """
    Creates the ETF store to run the training of the logistic 
    classificaiton tree

    :ARGS:
    
        ticker_list: iterable of tickers

        store_path: :class:`str` of path to ``HDFStore``
    """
    #check to make sure the store doesn't already exist
    if os.path.isfile(store_path):
        print "File " + store_path + " already exists"
        return
    
    store = pandas.HDFStore(store_path, 'w')
    success = 0
    for ticker in ticker_list:
        try:
            tmp = tickers_to_dict(ticker, 'yahoo', start = '01/01/2000')
            store.put(ticker, tmp)
            print ticker + " added to store"
            success += 1
        except:
            print "unable to add " + ticker + " to store"
    store.close()

    if success == 0: #none of it worked, delete the store
        print "Creation Failed"
        os.remove(path)
    return None

def first_valid_date(prices):
    """
    Helper function to determine the first valid date from a set of 
    different prices Can take either a :class:`dict` of 
    :class:`pandas.DataFrame`s where each key is a ticker's 'Open', 
    'High', 'Low', 'Close', 'Adj Close' or a single 
    :class:`pandas.DataFrame` where each column is a different ticker

    :ARGS:

        prices: either :class:`dictionary` or :class:`pandas.DataFrame`

    :RETURNS:

        :class:`pandas.Timestamp` 
   """
    iter_dict = { pandas.DataFrame: lambda x: x.columns,
                  dict: lambda x: x.keys() } 

    try:
        each_first = map(lambda x: prices[x].dropna().index.min(),
                         iter_dict[ type(prices) ](prices) )
        return max(each_first)
    except KeyError:
        print "prices must be a DataFrame or dictionary"
        return


def append_store_prices(ticker_list, store_path, start = '01/01/1990'):
    """
    Given an existing store located at ``path``, check to make sure
    the tickers in ``ticker_list`` are not already in the data
    set, and then insert the tickers into the store.

    :ARGS:

        ticker_list: :class:`list` of tickers to add to the
        :class:`pandas.HDStore`

        store_path: :class:`string` of the path to the     
        :class:`pandas.HDStore`

        start: :class:`string` of the date to begin the price data

    :RETURNS:

        :class:`NoneType` but appends the store and comments the
         successes ands failures
    """
    try:
        store = pandas.HDFStore(path = store_path,  mode = 'a')
    except IOError:
        print  path + " is not a valid path to an HDFStore Object"
        return
    store_keys = map(lambda x: x.strip('/'), store.keys())
    not_in_store = numpy.setdiff1d(ticker_list, store_keys )
    new_prices = tickers_to_dict(not_in_store, start = start)

    #attempt to add the new values to the store
    for val in new_prices.keys():
        try:
            store.put(val, new_prices[val])
            print val + " has been stored"
        except:
            print val + " couldn't store"
    store.close()
    return None

def complete_etf_list(write_path = None):
    """
    www.price-data.com has pages with all of the etf tickers
    available on it.  This function acesses each of those pages,
    stores the values to :class:`pandas.DataFrame` and saves to
    file

    :ARGS:
    
        write_path: :class:`string` of the path to save the :class:`DataFrame`
        of tickers as a ``.csv``  If ``None`` is provided, only the 
        :class:`pandas.DataFrame` will be returned
    """

    #the first three url's from price-data, the rest can be generated
    p_a = ("http://www.price-data.com/"
        "listing-of-exchange-traded-funds/")
    p_b = ("http://www.price-data.com/listing-of-exchange-traded-funds/"
        "listing-of-exchange-traded-funds-starting-with-b/")
    p_c = ("http://www.price-data.com/listing-of-exchange-traded-funds/"
        "list-of-exchange-traded-funds-starting-with-c/")
    p_k = ("http://www.price-data.com/listing-of-exchange-traded-funds/"
        "list-of-exchanged-traded-funds-starting-with-k/")

    base_url = ("http://www.price-data.com/listing-of-exchange-traded-funds/"
        "list-of-exchange-traded-funds-starting-with-")
    letters = 'defghijlmnopqrstuvwxyz'
    url_dict = dict(map(lambda x, y: [x, y], letters,
        map(lambda s: base_url + s, letters)))

    #the wonky ones
    url_dict['a'], url_dict['b'], url_dict['c'] = p_a, p_b, p_c
    url_dict['k'] = p_k

    #pull down the tables into a list

    d =[]
    for letter in url_dict.keys():
        try:
            d.extend(pandas.read_html(url_dict[letter], index_col = 0, 
                                       infer_types = False))
            print "succeeded for letter " + letter
        except:
            print "Did not succeed for letter " + letter

    agg_df = pandas.concat(d, axis = 0)
    agg_df.columns = ['Description']
    agg_df.index.name = 'Ticker'
    if path != None:
        agg_df.to_csv(path, encoding = 'utf-8')
    return agg_df

def adj_r2_uv(x, y):
    """
    Returns the adjusted R-Squared for multivariate regression
    """
    n = len(y)
    p = 1
    return 1 - (1 - r2_uv(x, y))*(n - 1)/(n - p - 1)    

def r2_uv(x, y):
    """
    Returns the R-Squared or `Coefficient of Determination
    <http://en.wikipedia.org/wiki/Coefficient_of_determination>`_ 
    for a univariate regression (does not adjust for more independent 
    variables
    
    .. seealso:: :meth:`r_squared_adjusted`

    :ARGS:

        series: :class`pandas.Series` of prices

        benchmark: :class`pandas.Series` of prices to regress 
        ``series`` against

    :RETURNS:

        float: of the coefficient of variation
    """
    def _r2_uv(x, y):   
        X = pandas.DataFrame({'ones':numpy.ones(len(x)), 'xs':x})
        beta = numpy.linalg.inv(X.transpose().dot(X)).dot(
            X.transpose().dot(y) )
        y_est = beta[0] + beta[1]*x
        ss_res = ((y_est - y)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        return 1 - ss_res/ss_tot

    if isinstance(x, pandas.DataFrame):
        return x.apply(lambda x: _r2_uv(y, x))
    else:
        return _r2_uv(x, y)

def r2_mv(x, y):   
    """
    Multivariate r-squared
    """
    ones = pandas.Series(numpy.ones(len(y)), name = 'ones')
    d = x.to_dict()
    d['ones'] = ones
    cols = ['ones']
    cols.extend(x.columns)
    X = pandas.DataFrame(d, columns = cols)
    beta = numpy.linalg.inv(X.transpose().dot(X)).dot(
        X.transpose().dot(y) )
    y_est = beta[0] + x.dot(beta[1:])
    ss_res = ((y_est - y)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    return 1 - ss_res/ss_tot
    
def adj_r2_mv(x, y):
    """
    Returns the adjusted R-Squared for multivariate regression
    """
    n = len(y)
    p = x.shape[1]
    return 1 - (1 - r2_mv(x, y))*(n - 1)/(n - p - 1)

def tickers_to_dict(ticker_list, api = 'yahoo', start = '01/01/1990'):
    """
    Utility function to return ticker data where the input is either a 
    ticker, or a list of tickers.

    :ARGS:

        ticker_list: :class:`list` in the case of multiple tickers or 
        :class:`str` in the case of one ticker

        api: :class:`string` identifying which api to call the data 
        from.  Either 'yahoo' or 'google'

        start: :class:`string` of the desired start date
                
    :RETURNS:

        :class:`dictionary` of (ticker, price_df) mappings or a
        :class:`pandas.DataFrame` when the ``ticker_list`` is 
        :class:`str`
    """
    def __get_data(ticker, api, start):
        reader = pandas.io.data.DataReader
        try:
            data = reader(ticker, api, start = start)
            print "worked for " + ticker
            return data
        except:
            print "failed for " + ticker
            return
    if isinstance(ticker_list, (str, unicode)):
        return __get_data(ticker_list, api = api, start = start)
    else:
        d = {}
        for ticker in ticker_list:
            d[ticker] = __get_data(ticker, api = api, start = start)
    return d
    
    
if __name__ == '__main__':
	
	usage = sys.argv[0] + "usage instructions"
	description = "describe the function"
	parser = argparse.ArgumentParser(description = description, usage = usage)
	parser.add_argument('name_1', nargs = 1, type = str, help = 'describe input 1')
	parser.add_argument('name_2', nargs = '+', type = int, help = "describe input 2")

	args = parser.parse_args()
	
	script_function(input_1 = args.name_1[0], input_2 = args.name_2)


