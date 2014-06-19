#!/usr/bin/env python
# encoding: utf-8
"""
Created by Benjamin M. Gross

"""

import argparse
import pandas
import numpy
import os
import pandas.io.data
import scipy.optimize as sopt

def run_classification():
    """
    See the classifications as they are run
    """
    etf_list = complete_etf_list()
    trained_list = pandas.Series.from_csv('../dat/trained_assets.csv')
    notin_trained = numpy.setdiff1d(etf_list.index.tolist(), 
                                    trained_list.index.tolist())
    #let 'er rip!
    for ticker in notin_trained:
        try:
            series = tickers_to_dict(ticker)['Adj Close']
            print find_nearest_neighbors(series, '../dat/trained_data.h5',
                                         trained_list)
        except:
            print "Didn't work for " + ticker

    return None
            
    
def find_nearest_neighbors(price_series, store_path, training_series):
    """
    Calculate the "nearest neighbors" on trained asset class data to 
    determine probabilities the series belongs to given asset classes
    
    :ARGS:

        series: :class:`pandas.Series` of prices
   

        store_path: :class:`string` leading to the store of trained  
        asset class prices (each ticker should also appear in the 
        ``.csv`` located at ``training_file_path``

        training_series: :class:`series` where Index is the tickers 
        and the values are the asset classes

    """
    ac_freq = training_series.value_counts()
    #choice of k, 80% of minimum unique trained instances for now
    k = int(round(0.8 * ac_freq.min() ))
    store = pandas.HDFStore(store_path, 'r')

    prob_d = {'r2_adj': [], 'asset_class': [] }
    for ticker in training_series.index:
        try:
            comp_asset = store.get(ticker)['Adj Close']
            ind = clean_dates(comp_asset, price_series)
            prob_d['r2_adj'].append(adj_r2_uv(
                comp_asset[ind].apply(numpy.log).diff().dropna(),
                price_series[ind].apply(numpy.log).diff().dropna()))
            prob_d['asset_class'].append(training_series[ticker])
        except:
            print "Did not work for ticker " + ticker

    prob_df = pandas.DataFrame(prob_d)
    prob_df.sort(columns = ['r2_adj'], ascending = False, inplace = True)
    return prob_df['asset_class'][:k].value_counts()/float(k)


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

def update_store_prices(path):
    """
    Update to the most recent prices for all keys of an existing store, 
    located at path ``path``.

    :ARGS:

        path: :class:`string` the location of the ``HDFStore`` file

    :RETURNS:

        :class:`NoneType` but updates the ``HDF5`` file, and prints to 
        screen which values would not update

    """
    reader = pandas.io.data.DataReader
    strftime = datetime.datetime.strftime
    
    today_str = strftime(datetime.datetime.today(), format = '%m/%d/%Y')
    try:
        store = pandas.HDFStore(path = path, mode = 'r+')
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
                store.put(key, stored_data.append(tmp).drop_duplicates())
            except IOError:
                print "could not update " + key

    store.close()
    return None

def create_data_store(path, ticker_list):
    """
    Creates the ETF store to run the training of the logistic 
    classificaiton tree
    """
    #check to make sure the store doesn't already exist
    if os.path.isfile(path):
        print "File " + path + " already exists"
        return
    
    store = pandas.HDFStore(path, 'w')
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


def append_store_prices(ticker_list, path, start = '01/01/1990'):
    """
    Given an existing store located at ``path``, check to make sure
    the tickers in ``ticker_list`` are not already in the data
    set, and then insert the tickers into the store.

    :ARGS:

        ticker_list: :class:`list` of tickers to add to the
        :class:`pandas.HDStore`

        path: :class:`string` of the path to the     
        :class:`pandas.HDStore`

        start: :class:`string` of the date to begin the price data

    :RETURNS:

        :class:`NoneType` but appends the store and comments the
         successes ands failures
    """
    try:
        store = pandas.HDFStore(path = path,  mode = 'a')
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

def complete_etf_list(path = None):
    """
    www.price-data.com has pages with all of the etf tickers
    available on it.  This function acesses each of those pages,
    stores the values to :class:`pandas.DataFrame` and saves to
    file

    :ARGS:
    
        path: :class:`string` of the path to save the :class:`DataFrame`
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


