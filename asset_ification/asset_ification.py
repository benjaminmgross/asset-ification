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
import os
    

def knn_wt_inv_weighted(series, trained_series, n = None):
    """
    Training data is a m x n matrix with 'training_tickers' as columns
    and rows of r-squared for different tickers and asset_class is a
    n x 1 result of the asset class

    :ARGS:

        series: :class:`pandas.Series` or :class:`pandas.DataFrame` of
        r-squared values

        trained_series: :class:`pandas.Series` of the columns
        and their respective asset clasnses

        n: :class:`integer` of the number of highest r-squared assets
        to include when classifying a new asset

    :RETURNS:

        :class:`string` of the tickers that have been estimated
        based on the n closest neighbors
   """
    if not n:
        n = len(trained_series.unique()) + 1

    return __weighting_method_agg_fun(series, trained_series, n,
                                      calc_meth = 'x-inv-x')

def knn_inverse_weighted(series, trained_series, n = None):
    """
    Training data is a m x n matrix with 'training_tickers' as columns
    and rows of r-squared for different tickers and asset_class is a
    n x 1 result of the asset class

    :ARGS:

        series: :class:`pandas.Series` or :class:`pandas.DataFrame` of
        r-squared values

        trained_series: :class:`pandas.Series` of the columns
        and their respective asset clasnses

        n: :class:`integer` of the number of highest r-squared assets
        to include when classifying a new asset

    :RETURNS:

        :class:`string` of the tickers that have been estimated
        based on the n closest neighbors
   """
    if not n:
        n = len(trained_series.unique()) + 1

    return __weighting_method_agg_fun(series, trained_series, n,
                                      calc_meth = 'inv-x')

def knn_exp_weighted(series, trained_series, n = None):
    """
    Training data is a m x n matrix with 'training_tickers' as columns
    and rows of r-squared for different tickers and asset_class is a
    n x 1 result of the asset class

    :ARGS:

        series: :class:`pandas.Series` or :class:`pandas.DataFrame` of
        r-squared values

        trained_series: :class:`pandas.Series` of the columns
        and their respective asset clasnses

        n: :class:`integer` of the number of highest r-squared assets
        to include when classifying a new asset

    :RETURNS:

        :class:`string` of the tickers that have been estimated
        based on the n closest neighbors
   """
    if not n:
        n = len(trained_series.unique()) + 1

    return __weighting_method_agg_fun(series, trained_series, n,
                                      calc_meth = 'exp-x')

def __weighting_method_agg_fun(series, trained_series, n, calc_meth):
    """
    Generator function for the different calcuation methods to determine
    the asset class based on a Series or DataFrame of r-squared values

    :ARGS:

        series: :class:`pandas.Series` or :class:`pandas.DataFrame` of
        r-squared values

        trained_series: :class:`pandas.Series` of the columns
        and their respective asset classes

        n: :class:`integer` of the number of highest r-squared assets
        to include when classifying a new asset

        calc_meth: :class:`string` of either ['x-inv-x', 'inv-x', 'exp-x']
        to determine which calculation method is used

    :RETURNS:

        :class:`string` of the asset class  been estimated
        based on the n closest neighbors, or 'series' in the case when
        a :class:`DataFrame` has been provided instead of a :class:`Series`

    """
    def weighting_method_agg_fun(series, trained_series, n, calc_meth):
        weight_map = {'x-inv-x': lambda x: x.div(1. - x),
                      'inv-x': lambda x: 1./(1. - x),
                      'exp-x': lambda x: numpy.exp(x)
                      }

        key_map = trained_series[series.index]
        series = series.rename(index = key_map)
        wts = weight_map[calc_meth](series)
        wts = wts.sort(ascending = False, inplace = False)
        grp = wts[:n].groupby(wts[:n].index).sum()
        return grp.argmax()

    if isinstance(series, pandas.DataFrame):
        return series.apply(
            lambda x: weighting_method_agg_fun(x,
            trained_series, n, calc_meth), axis = 1)
    else:
        return weighting_method_agg_fun(series, trained_series, n, calc_meth)
    

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



