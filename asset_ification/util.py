#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: util.py

Created by Benjamin M. Gross

A series of utility functions for online and HDFStore to extract price
series and classify assets.  To be used with the asset_ification.py
module
"""

import argparse
import datetime
import numpy
import pandas
import pandas.io.data
import os

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

def adj_r2_mv(x, y):
    """
    Returns the adjusted R-Squared for multivariate regression
    """
    n = len(y)
    p = x.shape[1]
    return 1 - (1 - r2_mv(x, y))*(n - 1)/(n - p - 1)

def adj_r2_uv(x, y):
    """
    Returns the adjusted R-Squared for multivariate regression
    """
    n = len(y)
    p = 1
    return 1 - (1 - r2_uv(x, y))*(n - 1)/(n - p - 1)

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

def construct_training_set(trained_assets, store_path):
    """
    Take in a :class:`pandas.Series` of assets with respective asset
    class and construct a `.csv` that can be used to test different
    machine learning techniques

    :ARGS:

        trained_assets: :class:`pandas.Series` of the trained asset
        classes

        store_path: :class:`string` of the location of the HDFStore
        where the asset prices are located

    :RETURNS:

        :class:`pandas.DataFrame` of the r-squared values for each
        of the "core assets"
    """
    try:
        store = pandas.HDFStore(store_path, 'r')
    except IOError:
        print store_path + " is not a valid path to HDFStore"
        return

    keys = pandas.Index(map(lambda x: x.strip('/'), store.keys() ))
    
    #in-sample tickers
    col_tickers = numpy.random.choice(keys, int(len(keys)/3.))

    compl_ins = keys[~keys.isin(col_tickers)]

    #create the asset clas mappings
    ac_asint = pandas.Categorical.from_array(trained_assets)
    cat_dict = dict( zip(ac_asint, ac_asint.labels) )
    ac_df = pandas.DataFrame({'asset_class': trained_assets,
                           'ac_asint': ac_asint.labels},
                           index = trained_assets.index)
    
    #ticker_dict =  dict( (tick, []) for tick in col_tickers)
    p_fish = fish.ProgressFish(total = len(compl_ins))
    agg_d = {}
    os_d = {}
    for i, col in enumerate(col_tickers):
        p_fish.animate(amount = i)
        d = {}
        for ticker in compl_ins:
            xs = store.get(col)['Adj Close']
            ys = store.get(ticker)['Adj Close']
            ind = xs.index & ys.index
            rsq = pandas.ols(x = xs[ind].apply(numpy.log).diff(),
                           y = ys[ind].apply(numpy.log).diff()).r2_adj
            d[ticker] = rsq
        agg_d[col] = pandas.Series(d)

    store.close()
    ret_df = pandas.DataFrame(agg_d)
    ret_df['ac_asint'] = ac_df.loc[tmp.index, 'ac_asint']
    return ret_df


def data_to_in_out_sample(data, train_prop):
    """
    Split the data into training and testing data, wwhere training
    data = train_prop of the total length of data

    :ARGS:

        data: :class:`pandas.DataFrame` of the training data

        train_prop :class:`float` of the proportion of the data to be
        be used for training

    :RETURNS:

        train_xs, train_ys, test_xs, test_ys ... 
    """
    train_ind = numpy.random.choice(data.index, int(train_prop * len(data)),
                                    replace = False)
    test_ind = data.index[~data.index.isin(train_ind)]
    train_xs = data.loc[train_ind, :].copy()
    train_ys = train_xs['ac_asint'].copy()
    train_xs = train_xs.drop(labels = ['ac_asint'], axis = 1)
    test_xs = data.loc[test_ind, :].copy()
    test_ys = test_xs['ac_asint'].copy()
    test_xs = test_xs.drop(labels = ['ac_asint'], axis = 1)
    return train_xs, train_ys, test_xs, test_ys


def check_for_keys(ticker_list, store_path):
    """
    HDFStore function to determine which, if any of the :class:`list`
    `ticker_list` are inside of the store.  If all tickers are located
    in the store returns 1, otherwise returns 0 (provides a "check" to see if
    other functions can be run)
    """
    try:
        store = pandas.HDFStore(path = store_path, mode = 'r+')
    except IOError:
        print  store_path + " is not a valid path to an HDFStore Object"
        return

    if isinstance(ticker_list, pandas.Index):
        #pandas.Index is not sortable, so much tolist() it
        ticker_list = ticker_list.tolist()

    store_keys = map(lambda x: x.strip('/'), store.keys())
    not_in_store = numpy.setdiff1d(ticker_list, store_keys)
    store.close()

    #if len(not_in_store) == 0, all tickers are present
    if not len(not_in_store):
        print "All tickers in store"
        return 1
    else:
        for ticker in not_in_store:
            print "store does not contain " + ticker
        return 0


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

def log_returns(series):
    """
    Return the log_returns of a price series

    :ARGS:

        series: :class:`pandas.Series` of prices

    :RETURNS:

        :class:`pandas.Series` of log returns
    
    """
    return series.apply(numpy.log).diff()

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
        print  store_path + " is not a valid path to an HDFStore Object"
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


def price_data_etf_list(write_path = None):
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


