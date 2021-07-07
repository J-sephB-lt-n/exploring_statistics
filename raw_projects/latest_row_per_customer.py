import pandas as pd
testdata = pd.DataFrame( {'date':['2020-01-01','2020-01-01','2020-01-01',
                                  '2020-01-02','2020-01-02','2020-01-02', 
                                  '2020-01-03','2020-01-03','2020-01-03',
                                  '2020-01-03','2020-01-03','2020-01-03'],
                           'customer_ID':[11,12,13,11,12,13,11,12,13,11,12,13],                                 
                          'n_sold':[1,2,3,4,5,6,7,8,9,10,11,12]                            
                         }
                       ).sample(12, replace=False).sort_values(['customer_ID','date'])

print(testdata)
print( 
        testdata.sort_values(by="date").drop_duplicates(subset=["customer_ID"], keep="last")
)

print( 
        testdata.sort_values(by="date").drop_duplicates(subset=["customer_ID"], keep="first")
)