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

def scipt_function(arg_1, arg_2):
    return None

def calibrate_model(class_list, test_list):
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
    

    return None

def create_etf_store(path, ticker_list):
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
    p_a = ("http://www.price-data.com/
        listing-of-exchange-traded-funds/")
    p_b = ("http://www.price-data.com/listing-of-exchange-traded-funds/
        listing-of-exchange-traded-funds-starting-with-b/")
    p_c = ("http://www.price-data.com/listing-of-exchange-traded-funds/
        list-of-exchange-traded-funds-starting-with-c/")
    p_k = ("http://www.price-data.com/listing-of-exchange-traded-funds/
        list-of-exchanged-traded-funds-starting-with-k/")

    base_url = ("http://www.price-data.com/listing-of-exchange-traded-funds/
        list-of-exchange-traded-funds-starting-with-")
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
           f d.extend(pandas.read_html(url_dict[letter], index_col = 0, 
                                       infer_types = False))

            print "succeeded for letter " + letter
        except:
            print "Did not succeed for letter " + letter

    agg_df = pandas.concat(d, axis = 0)
    agg_df.columns = ['Description']
    agg_df.index.name = 'Ticker'
    agg_df.to_csv(path, encoding = 'utf-8')
    return None
    
    
    
if __name__ == '__main__':
	
	usage = sys.argv[0] + "usage instructions"
	description = "describe the function"
	parser = argparse.ArgumentParser(description = description, usage = usage)
	parser.add_argument('name_1', nargs = 1, type = str, help = 'describe input 1')
	parser.add_argument('name_2', nargs = '+', type = int, help = "describe input 2")

	args = parser.parse_args()
	
	script_function(input_1 = args.name_1[0], input_2 = args.name_2)
