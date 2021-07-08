import pandas as pd
testdata = pd.DataFrame( {'date':['2020-01-01','2020-01-01','2020-01-01',
                                  '2020-01-02','2020-01-02','2020-01-02', 
                                  '2020-01-03','2020-01-03','2020-01-03'],
                          'item_type':['car','dog','flower',
                                       'car','dog','flower',
                                       'car','dog','flower'
                                       ],
                          'item':['mitsubishi','chihuahua','iris',
                                  'mitsubishi','chihuahua','iris',
                                  'mitsubishi','chihuahua','iris',
                                 ],
                          'n_sold':[1,2,3,4,5,6,7,8,9]                            
                         }
                       )

print(testdata)
print( testdata.groupby(['item_type', 'item','date'])['n_sold'].sum().unstack('date').reset_index() )