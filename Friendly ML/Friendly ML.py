# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 18:19:20 2019

@author: jeson
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Importing the libraries
    import backend.backend as back
    
    # Request learning type
    learning_type = back.request_learning()
    
    # Request problem
    problem = back.request_problem(learning_type)
    
    # Request dataset/features/label. Encodes categorical variables.
    data = back.request_all_data(problem)
    
    # Request model
    model = back.request_model(problem, data)
    
    # Predict until user inputs 'quit'
    back.predict_until_quit(data, model, problem)