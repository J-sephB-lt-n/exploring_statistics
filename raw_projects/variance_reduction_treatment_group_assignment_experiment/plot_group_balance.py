

# user inputs --------------------------------------------------------------------------------------------------------------#
period_start_date = '2020-01-01'         # this date included in selection
period_end_date = '2021-04-10'           # this date exluded from selection
campaign_commence_date = '2021-03-10'
customer_query_SQL = """
        SELECT 
                    [PersonNo]
                ,   [test1_holdout0]              -- group assignment column must have this name and contain {0,1} values
        FROM 
                    [Analytics].[dbo].[all_HC_group_assignments]
        WHERE
                    [solution] = 'OrderProp RFM7-10 selection'
        AND
                    [campaign_commence_date] = '2021-03-10'
        --AND
        --            [group_desc] = 'cell_only'
"""
rolling_mean_window_size = 4
#---------------------------------------------------------------------------------------------------------------------------#

import pandas as pd
import sqlalchemy
import urllib 
from matplotlib import pyplot as plt

server = 'pwbhcdsci01'
database = 'Analytics'
params = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};SERVER="+server+";DATABASE="+database+";Trusted_connection=yes;")
sql_engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params, fast_executemany=True)

# get weekly total spend (OrdPrice) per group for selected customers:
order_data = pd.read_sql( 
                f"""

                    SELECT 
                                allperiods.[CalendarYearNo]
                            ,   allperiods.[WeekOfYear]
                            ,   allperiods.[test1_holdout0]
                            ,   CASE WHEN allspend.[sum_OrdPrice] IS NULL THEN 0 ELSE allspend.[sum_OrdPrice] END AS [sum_OrdPrice]
                            ,   CASE WHEN allspend.[n_customers_with_order] IS NULL THEN 0 ELSE allspend.[n_customers_with_order] END AS [n_customers_with_order]
                    FROM    
                            -- list all periods between period_start_date and period_end_date:
                            -- (for both test1_holdout0=1 and test1_holdout0=0) 
                            (
                                SELECT 
                                        DISTINCT 
                                            1 AS [test1_holdout0]
                                        ,   [CalendarYearNo]
                                        ,   [WeekOfYear]
                                FROM 
                                            [data_staging].[extract].[HC_DW_dim_date]
                                WHERE
                                            [Date] >= '{period_start_date}'
                                AND
                                            [Date] < '{period_end_date}'
                            UNION 
                                SELECT 
                                        DISTINCT 
                                            0 AS [test1_holdout0]
                                        ,   [CalendarYearNo]
                                        ,   [WeekOfYear]
                                FROM 
                                            [data_staging].[extract].[HC_DW_dim_date]
                                WHERE
                                            [Date] >= '{period_start_date}'
                                AND
                                            [Date] < '{period_end_date}'
                            )
                            allperiods
                    
                    LEFT JOIN   
                                -- total spend per treatment group per period: 
                            ( 
                                SELECT 
                                            dimdate.[CalendarYearNo]
                                        ,   dimdate.[WeekOfYear]
                                        ,   customerlist.[test1_holdout0]
                                        ,   SUM( orders.[OrdPrice] ) AS [sum_OrdPrice]
                                        ,   SUM( CASE WHEN orders.[OrdPrice]>0 THEN 1 ELSE 0 END ) AS [n_customers_with_order] 
                                FROM 
                                            [data_staging].[extract].[HC_DW_factOrderTruncate] orders 
                                INNER JOIN 
                                            
                                            -- restrict to selected weeks:
                                            (
                                                SELECT 
                                                            [DateKey]
                                                        ,   [CalendarYearNo]
                                                        ,   [WeekOfYear]
                                                FROM
                                                        [data_staging].[extract].[HC_DW_dim_date]
                                                WHERE
                                                        [Date] >= '{period_start_date}'
                                                AND
                                                        [Date] < '{period_end_date}'
                                            )
                                            dimdate
                                ON
                                            orders.[OrderDteKey] = dimdate.[DateKey]
                                INNER JOIN 
                                            -- restrict to only dispatched orders:
                                            (
                                                SELECT 
                                                            [OrdStatusKey]
                                                FROM 
                                                            [data_staging].[extract].[HC_DW_dimOrdStatus]
                                                WHERE 
                                                            [OrdStatus] = 'Despatched'
                                            )
                                            dimOrdStatus
                                ON
                                            orders.[CurrOrdStatus] = dimOrdStatus.[OrdStatusKey]   
                                INNER JOIN
                                            -- restrict to selected customers 
                                            (
                                                {customer_query_SQL}
                                            )   
                                            customerlist           
                                ON
                                            orders.[PersonNo] = customerlist.[PersonNo]  
                                    /*
                                    INNER JOIN
                                                --- restrict to only OUTBOUND orders (orders arising from dialler agent)
                                                (	
                                                    SELECT 
                                                                [OrdSourceKey]
                                                    FROM
                                                                [data_staging].[extract].[HC_DW_dimOrdSource]
                                                    WHERE
                                                                [OrdSourceGrp_new] = 'Outbound'
                                                )
                                                restrict_source
                                    ON
                                                orders.[OrdSourceKey] = restrict_source.[OrdSourceKey]
                                    */
                                GROUP BY                            
                                            dimdate.[CalendarYearNo]
                                        ,   dimdate.[WeekOfYear]
                                        ,   customerlist.[test1_holdout0]  
                            )
                            allspend
                    ON 
                            allperiods.[CalendarYearNo] = allspend.[CalendarYearNo] 
                    AND
                            allperiods.[WeekOfYear] = allspend.[WeekOfYear]
                    AND
                            allperiods.[test1_holdout0] = allspend.[test1_holdout0]
                    ORDER BY                            
                                allperiods.[CalendarYearNo]
                            ,   allperiods.[WeekOfYear]
                            ,   allperiods.[test1_holdout0]                            
                """,
                con = sql_engine
) 

campaign_commence_week = pd.read_sql(
    f"""
        SELECT  
                    [CalendarYearNo]
                ,   [WeekOfYear]
        FROM
                    [data_staging].[extract].[HC_DW_dim_date]
        WHERE 
                    [Date] = '{campaign_commence_date}'
    """,
    con = sql_engine
)

group_sizes = pd.read_sql( 
                f"""
                    SELECT 
                                [test1_holdout0], 
                                COUNT(DISTINCT PersonNo) AS [n_customers]
                    FROM 
                                (
                                {customer_query_SQL}     
                                )
                                customerlist
                    GROUP BY 
                                [test1_holdout0]
                """,
                con = sql_engine
)

data_for_plot = (
    order_data
    .merge( group_sizes,
            on = 'test1_holdout0'
          )
    .assign( avg_spend_per_customer = lambda df: df['sum_OrdPrice']/df['n_customers'] )
    .pivot( index = ['CalendarYearNo','WeekOfYear'], columns='test1_holdout0', values='avg_spend_per_customer')
    .reset_index()
    .assign( diff = lambda df: df[1]-df[0],                                              # weekly difference between avg spend in test vs. holdout
             yearweek = lambda df: 100*df['CalendarYearNo'] + df['WeekOfYear'],           # make a single column containing the week number
             percent_diff = lambda df: df[1]/df[0] - 1
        )
)
data_for_plot['rolling_mean_diff'] = data_for_plot['diff'].rolling(rolling_mean_window_size).mean()

campaign_commence_date_line_location = data_for_plot.loc[ (data_for_plot['CalendarYearNo']==campaign_commence_week['CalendarYearNo'].values[0]) & (data_for_plot['WeekOfYear']==campaign_commence_week['WeekOfYear'].values[0]) ].index[0]

# make barplot of difference per week:
ax = data_for_plot.plot(
                        kind = 'bar', 
                        x = 'yearweek',
                        y = 'diff',
                        figsize = (20,10),
                        title = 'Difference Between [Mean OrdPrice Test] and [Mean OrdPrice Holdout Group] : reported by week',
                        ylabel = '[Mean OrdPrice Test]-[Mean OrdPrice Holdout Group]',
                        xlabel = 'Year_Week' 
                 )
# draw in rolling mean difference:                 
plt.plot( 
            data_for_plot.index, 
            data_for_plot['rolling_mean_diff'].values,
            color = 'red',
            linewidth = 2.0
        )
# draw in vertical blue line at start of campaign:
plt.axvline( 
                x = campaign_commence_date_line_location,
                color = 'blue',
                label = 'campaign commence week'
            ) 

# response plot 
data_for_response_plot = (
    order_data
    .merge( group_sizes,
            on = 'test1_holdout0'
          )
    .assign( percent_responded = lambda df: 100 * df['n_customers_with_order']/df['n_customers'] )
    .pivot( index = ['CalendarYearNo','WeekOfYear'], columns='test1_holdout0', values='percent_responded')
    .reset_index()
    .assign( diff = lambda df: df[1]-df[0],                                              # weekly difference between avg spend in test vs. holdout
             yearweek = lambda df: 100*df['CalendarYearNo'] + df['WeekOfYear'],           # make a single column containing the week number
             percent_diff = lambda df: df[1]/df[0] - 1
        )
)
data_for_response_plot['rolling_mean_diff'] = data_for_response_plot['diff'].rolling(rolling_mean_window_size).mean()

# make barplot of response difference per week:
ax = data_for_response_plot.plot(
                        kind = 'bar', 
                        x = 'yearweek',
                        y = 'diff',
                        figsize = (20,10),
                        title = 'Difference Between % Responded in Test Group and % Responded in Holdout Group : reported by week',
                        ylabel = '%responded in test - %responded in holdout',
                        xlabel = 'Year_Week' 
                 )
# draw in rolling mean difference:                 
plt.plot( 
            data_for_response_plot.index, 
            data_for_response_plot['rolling_mean_diff'].values,
            color = 'red',
            linewidth = 2.0
        )
# draw in vertical blue line at start of campaign:
plt.axvline( 
                x = campaign_commence_date_line_location,
                color = 'blue',
                label = 'campaign commence week'
            ) 

# plot of ratio: 
#data_for_plot.plot.bar( 
#                        x = 'yearweek',
#                        y = 'percent_diff',
#                        figsize = (20,10),
#                        title = '% Difference Between [Mean OrdPrice Test] and [Mean OrdPrice Holdout Group] : reported by week',
#                        ylabel = '[Mean OrdPrice Test] / [Mean OrdPrice Holdout Group] -1',
#                        xlabel = 'Year_Week' 
#                 )
