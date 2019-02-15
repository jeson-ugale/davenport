# Apriori

from time import time
from algorithms.association_rule.apyori import apriori as apyori

def apriori(dataset):
    start=time()
    
    # Data Preprocessing
    transactions = []
    transactions = [[str(dataset.values[i,j]) for j in range(0, dataset.shape[1])
    if str(dataset.values[i,j]) != 'nan'] for i in range(0, dataset.shape[0])]
    
    # Training Apriori on the dataset
    rules = apyori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
     
    # Putting the results in a list
    results = list(rules)
    results_list = []
    for i in range(0, len(results)):
        rule = str(results[i][0]).split('\'')
        rule = rule[1] + ' -> ' + rule[3]
        support = results[i][1]
        results_list.append((rule, support))
    
    # Return top 10 rules
    return {'name':'Apriori', 'rules':results_list, 'duration':(time()-start)/60}