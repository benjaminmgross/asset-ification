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

def best_fitting_weights(series, asset_class_prices):
    """
    Return the best fitting weights given a :class:`pandas.Series` of 
    asset prices and a :class:`pandas.DataFrame` of asset class prices.  
    Can be used with the :func:`clean_dates` function to ensure an 
    intersection of the two indexes is being passed to the function
    
    :ARGS:

        asset_prices: m x 1 :class:`pandas.TimeSeries` of asset_prices

        ac_prices: m x n :class:`pandas.DataFrame` asset class ("ac") 
        prices

    :RETURNS:

        :class:`pandas.TimeSeries` of nonnegative weights for each 
        asset such that the r_squared from the regression of 
        :math:`Y ~ Xw + e` is maximized

    """

    def _r_squared_adj(weights):
        """
        The Adjusted R-Squared that incorporates the number of 
        independent variates using the `Formula Found of Wikipedia
        <http://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2>_`
        """

        estimate = numpy.dot(ac_rets, weights)
        sse = ((estimate - series_rets)**2).sum()
        sst = ((series_rets - series_rets.mean())**2).sum()
        rsq = 1 - sse/sst
        p, n = weights.shape[0], ac_rets.shape[0]
        return rsq - (1 - rsq)*(float(p)/(n - p - 1))

    def _obj_fun(weights):
        """
        To maximize the r_squared_adj minimize the negative of r_squared
        """
        return -_r_squared_adj(weights)

    #linear price changes to create a weighted return
    ac_rets = asset_class_prices.pct_change()
    series_rets = series.pct_change()

    #de-mean the sample
    ac_rets = ac_rets.sub(ac_rets.mean() )
    series_rets = series_rets.sub( series_rets.mean() )

    num_assets = ac_rets.shape[1]
    guess = numpy.zeros(num_assets,)

    #ensure the boundaries of the function are (0, 1)
    ge_zero = [(0,1) for i in numpy.arange(num_assets)]

    #optimize to maximize r-squared, using the 'TNC' method 
    #(that uses the boundary functionality)
    opt = sopt.minimize(_obj_fun, x0 = guess, method = 'TNC', 
                        bounds = ge_zero)
    normed = opt.x*(1./numpy.sum(opt.x))

    return pandas.TimeSeries(normed, index = ac_rets.columns)

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
    
    store = pandas.HDF5tore(path, 'w')
    for ticker in ticker_list:
        try:
            tmp = web.DataReader(ticker, 'yahoo', start = '01/01/2000')
            store.put(ticker, tmp)
            print ticker + " added to store"
        except:
            print "unable to add " + ticker + " to store"
    store.close()

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

def class_dicts():
    """
    Given a list of asset classes and a list of tickers to train
    the asset classes, with a store located at ``path``, train the 
    logistic classification tree
    
    :ARGS:

        class_list: :class:`list` of asset class tickers `

        test_list: :class:`list` of tickers to train the algorithm
        with

   .. note:: Requires Existing HDF5Store

        Be sure to create an HDF5Store before using this function
        as it looks for asset prices in the HDF5Store located at 
        ``path``.
    """
    fi_dict = {'US Inflation Protected':'TIP', 'Foreign Treasuries':'BWX',
               'Foreign High Yield':'PCY','US Investment Grade':'LQD',
               'US High Yield':'HYG', 'US Treasuries ST':'SHY',
               'US Treasuries LT':'TLT', 'US Treasuries MT':'IEF'}

    us_eq_dict = {'U.S. Large Cap Growth':'JKE', 'U.S. Large Cap Value':'JKF',
                  'U.S. Mid Cap Growth':'JKH','U.S. Mid Cap Value':'JKI',
                  'U.S. Small Cap Growth':'JKK', 'U.S. Small Cap Value':'JKL'}

    for_eq_dict = {'Foreign Developed Small Cap':'SCZ',
                   'Foreign Developed Large Growth':'EFG',
                   'Foreign Developed Large Value':'EFV',
                   'Foreign Emerging Market':'EEM'}


    alt_dict = {'Commodities':'GSG', 'U.S. Real Estate':'IYR',
                'Foreign Real Estate':'WPS', 'U.S. Preferred Stock':'PFF'}
    return none

def gen_univariate_rsquared(path):

    store = pandas.HDFStore(path, 'r')
    
    acs = ['GSG','IYR','WPS','PFF','EEM','EFV','EFG','SCZ','JKL',
           'JKK','JKI','JKH','JKF','JKE','IEF','TLT','SHY','HYG',
           'LQD','PCY','BWX','TIP']

    acs_df = pandas.DataFrame(dict(map(lambda x, y: [x, y], acs, 
        map(lambda x: store.get(x)['Adj Close'], acs) ))).dropna()
    
    not_acs = numpy.setdiff1d(
        map(lambda x: x.strip('/'), store.keys() ), acs)

    rsq_d = {}
    for ticker in not_acs:
        tmp = store.get(ticker)['Adj Close']
        ind = acs_df.index & tmp.index
        rsq_d[ticker] =  r_squared(tmp[ind], acs_df.loc[ind, :])
        print "all univiariate r_squared generated for " + ticker
    
    ret_df = pandas.DataFrame.from_dict(rsq_d, orient = 'index')
    ret_df.columns = acs
    store.close()

    return ret_df

def gen_linprog_maxrsquard(path):

    store = pandas.HDFStore(path, 'r')
    
    acs = ['GSG','IYR','WPS','PFF','EEM','EFV','EFG','SCZ','JKL',
           'JKK','JKI','JKH','JKF','JKE','IEF','TLT','SHY','HYG',
           'LQD','PCY','BWX','TIP']

    acs_df = pandas.DataFrame(dict(map(lambda x, y: [x, y], acs, 
        map(lambda x: store.get(x)['Adj Close'], acs) ))).dropna()
    
    not_acs = numpy.setdiff1d(
        map(lambda x: x.strip('/'), store.keys() ), acs)

    rsq_d = {}
    for ticker in not_acs:
        tmp = store.get(ticker)['Adj Close']
        ind = acs_df.index & tmp.index
        rsq_d[ticker] = best_fitting_weights(tmp[ind], acs_df.loc[ind, :])
        print "max r_squared generated for " + ticker
    
    ret_df = pandas.DataFrame.from_dict(rsq_d, orient = 'index')
    ret_df.columns = acs
    
    return ret_df
    


def gen_etf_list(path):
    """
    www.price-data.com has pages with all of the etf tickers
    available on it.  This function accesses each of those pages,
    stores the values to :class:`pandas.DataFrame` and saves to
    file

    :ARGS:
    
        path: :class:`string` of the path to save the :class:`DataFrame`
        of tickers
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
    agg_df.to_csv(path, encoding = 'utf-8')
    return None

def adj_r2_uv(x, y):
    """
    Returns the adjusted R-Squared for multivariate regression
    """
    n = len(y)
    p = 1
    return 1 - (1 - r2_uv(x, y))*(n - 1)/(n - p - 1)    

def r2_uv(series, benchmark):
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

    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _r2_uv(series, x))
    else:
        return _r2_uv(series, benchmark)

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


