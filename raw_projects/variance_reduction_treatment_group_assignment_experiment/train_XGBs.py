
"""
DROP TABLE IF EXISTS [feature_store_dev].[dbo].[temp_customer_list_for_experimental_group_assignment]
;

SELECT
            TOP 500000
            [PersonNo]
INTO 
            [feature_store_dev].[dbo].[temp_customer_list_for_experimental_group_assignment]
FROM 
            [data_staging].[extract].[HC_DW_dmtlistmodel_history] 
WHERE   
            -- latest data 
            [CycleDateKey] = 20190409     -- campaign starts '2019-07-09', so this leaves 3 months of response period to train model
AND
            [RFM_Segment] < 11
ORDER BY 
            NEWID()           -- random order
;
"""

"""
-- training response y:
DROP TABLE IF EXISTS [feature_store_dev].[dbo].[temp_response_y_for_experimental_group_assignment]
;
SELECT 
            customerlist.[PersonNo]
        ,   CASE WHEN nsv.[sum_OrdPrice] IS NOT NULL THEN [sum_OrdPrice] ELSE 0 END AS [sum_OrdPrice]
        ,   CASE WHEN nsv.[sum_OrdPrice] IS NOT NULL THEN 1 ELSE 0 END AS [ordered]
INTO
            [feature_store_dev].[dbo].[temp_response_y_for_experimental_group_assignment]
FROM 
            [feature_store_dev].[dbo].[temp_customer_list_for_experimental_group_assignment] customerlist
LEFT JOIN
            (
                SELECT 
                            [PersonNo]
                        ,   SUM(OrdPrice) AS sum_OrdPrice
                FROM
                            [data_staging].[extract].[HC_DW_factOrderTruncate] orders
                INNER JOIN
                            -- all orders except 'cancelled' and 'goods returned': 
                            (
                                SELECT 
                                        [OrdStatusKey]
                                FROM 
                                        [data_staging].[extract].[HC_DW_dimOrdStatus]
                                WHERE 
                                        [OrdStatusGrp] IN ('Intake','Pass','Despatch')
                            )
                            dispatch
                ON  
                            orders.[CurrOrdStatus] = dispatch.[OrdStatusKey]
                WHERE 
                            orders.[OrderDteKey] > 20190409     -- response period start date
                AND 
                            orders.[OrderDteKey] < 20190709     -- response period end date
                GROUP BY    
                            [PersonNo]
            )
            nsv 
ON 
            customerlist.[PersonNo] = nsv.[PersonNo]
;
"""

"""
-- training features X:
DROP TABLE IF EXISTS [feature_store_dev].[dbo].[temp_features_x_for_experimental_group_assignment]
; 
EXEC [feature_store_prod].[dbo].[Create_Features]
                                                            @period_start_date = '2019-01-09'
                                                        ,   @period_end_date = '2019-07-09'         -- campaign start date
                                                        ,   @customerlist_location = '[feature_store_dev].[dbo].[temp_customer_list_for_experimental_group_assignment]'
						                                ,   @desired_output_location = '[feature_store_dev].[dbo].[temp_features_x_for_experimental_group_assignment]'    -- provide desired location for final output of this script (e.g. 'feature_store_dev.dbo.CategoryPropensity_Training_Features') - will be created if it doesn't exist
										                ,   @append_to_existing = 0
                                                        ,   @feature_table_list = 'hc_people|hc_orders|email6|rfm_atb|prod_productline_hist|recent_productline_seq'   
;
"""

"""
USE [feature_store_dev]
EXEC sp_rename 'feature_store_dev.dbo.temp_features_x_for_experimental_group_assignment.PersonNo', 'PersonNo_X', 'COLUMN'
;
"""

import pickle
import pyodbc
import datetime
import sqlalchemy
import urllib
import pandas as pd 
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime

# define SQL connection:
dss_conn = pyodbc.connect("Driver={SQL Server};"
                            "Server=PWBHCDSCI01;"
                            "Database=feature_store_dev;"
                            "uid=;pwd=",autoCommit = True
                        )
csr = dss_conn.cursor()

# pull model training data into python ----------------------------------------------------------------------------------------------------
training_features_query = """   SELECT 
                                                *
                                FROM
                                                [feature_store_dev].[dbo].[temp_response_y_for_experimental_group_assignment] y
                                LEFT JOIN
                                                [feature_store_dev].[dbo].[temp_features_x_for_experimental_group_assignment] x
                                ON
                                                y.PersonNo = x.PersonNo_X
                        """
join_deletes = ['PersonNo']                  # columns that I want dropped after fetching the features data 
X_from_column_index = 5  
Y_columns_index = [0,2]
Z_columns_index = [2,5]

server = 'PWBHCDSCI01'
database = 'feature_store_dev'
params = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};SERVER="+server+";DATABASE="+database+";Trusted_connection=yes;")
engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params, fast_executemany=True)
train_data = pd.read_sql_query(training_features_query, engine)

# clean & prepare model training data -----------------------------------------------------------------------------------------------------
train_data = train_data.drop(join_deletes,axis=1)
train_data = train_data.fillna(0)          # replace NULL entries with 0
X = train_data.iloc[:,X_from_column_index:]
y = train_data.iloc[:,Y_columns_index[0]:Y_columns_index[1]]
z = train_data.iloc[:,Z_columns_index[0]:Z_columns_index[1]]
X_train, X_test, y_train, y_test, z_train, z_test = train_test_split( 
                                                                      X, 
                                                                      y,
                                                                      z,
                                                                      test_size = 0.05,           # leave 5% as test set
                                                                      random_state = 3141592
                                                                    )
# scale the features data:                                                                    
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)                                   # learn mean and sd of training data, to use for later scaling                                                               
X_train = standard_scaler.transform(X_train)                   # scale training data 
X_test = standard_scaler.transform(X_test)                     # scale test data 

# save the scaler:                                             https://www.thoughtco.com/using-pickle-to-save-objects-2813661
scaler_save_location = 'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//saved_scaler.obj'
pickle.dump( 
                standard_scaler,
                open(scaler_save_location, 'wb')
           )

# define regression model ----------------------------------------------------------------------------------------------------------------------------
# (spend prediction model)
xgb_regressor = XGBRegressor(
                    max_depth = 3,                    #3,int
                learning_rate = 0.1,              #0.1,float
                 n_estimators = 100,               #100,int
                       silent = True,                    #True,False
                    objective = "reg:squarederror",    #reg:linear, reg:logistic, binary:logistic, check documentation
                      booster = "gbtree",               #gbtree,gblinear,dart
                        gamma = 0,                        #0,float
             min_child_weight = 1,             #1,int
               max_delta_step = 0,               #0,int
                    subsample = 1,                    #1,float
             colsample_bytree = 1,             #1,float
            colsample_bylevel = 1,            #1,float
                    reg_alpha = 0,                    #0,float (L1 regularization)
                   reg_lambda = 1,                   #1,float (L2 regularization)
             scale_pos_weight = 1,             #1,float
                   base_score = 0.5,                 #0.5,float
                         seed = None,                      #None,int
                      missing = None,                   #None,float
                       n_jobs = 20,
                 random_state = 1969,
                      nthread = 20
                )

# train the regression model -------------------------------------------------------------------------------------------------------------------------
print( datetime.datetime.now() )         # time at training start 
xgb_regressor.fit( 
                   X_train,
                   y_train['sum_OrdPrice']
                )
print( datetime.datetime.now() )         # time at training end              

# save the trained regression (spend predicting) model:
spend_predicting_model_save_location = 'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//saved_spend_predicting_model.obj'
pickle.dump( 
                xgb_regressor,
                open(spend_predicting_model_save_location, 'wb')
           )

# define classification model ----------------------------------------------------------------------------------------------------------------------------
# (order propensity model)
xgb_classifier = XGBClassifier(
                                max_depth = 3,                    #3,int
                                learning_rate = 0.1,              #0.1,float
                                n_estimators = 100,               #100,int
                                silent = True,                    #True,False
                                objective = "binary:logistic",    #reg:linear, reg:logistic, binary:logistic, check documentation
                                booster = "gbtree",               #gbtree,gblinear,dart
                                gamma = 0,                        #0,float
                                min_child_weight = 1,             #1,int
                                max_delta_step = 0,               #0,int
                                subsample = 1,                    #1,float
                                colsample_bytree = 1,             #1,float
                                colsample_bylevel = 1,            #1,float
                                reg_alpha = 0,                    #0,float (L1 regularization)
                                reg_lambda = 1,                   #1,float (L2 regularization)
                                scale_pos_weight = 1,             #1,float
                                base_score = 0.5,                 #0.5,float
                                seed = None,                      #None,int
                                missing = None,                   #None,float
                                n_jobs = 20,
                                random_state = 0,
                                nthread = 20
                            )

# train the classifier model -------------------------------------------------------------------------------------------------------------------------
print( datetime.datetime.now() )         # time at training start 
xgb_classifier.fit( 
                   X_train,
                   y_train['ordered']
                )
print( datetime.datetime.now() )         # time at training end              

# save the trained classification (order prob predicting) model:
order_prob_model_save_location = 'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//saved_order_prob_model.obj'
pickle.dump( 
                xgb_classifier,
                open(order_prob_model_save_location, 'wb')
           )

           

