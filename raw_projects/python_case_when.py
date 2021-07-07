import pandas as pd
import numpy as np 

testdata = pd.DataFrame( {'x':[0,0,1,1,6], 'y':[0,1,0,1,9]} )

testdata['cases'] = np.select( condlist = [ testdata.x == 0 & testdata['y']==0,
                                            testdata['x']==0 & testdata['y']==1,
                                            testdata['x']==1 & testdata['y']==0,
                                            testdata['x']==1 & testdata['y']==1
                                           ],
                             choicelist = [ 'zero',
                                            'one',
                                            'one',
                                            'two'
                                           ],
                                 default = 'other'
                            ) 
testdata                            
