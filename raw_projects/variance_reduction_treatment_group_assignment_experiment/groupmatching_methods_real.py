
import pandas as pd
import numpy as np
import urllib
import sqlalchemy
from matplotlib import pyplot as plt
import math            # for function exp()
import random
import pyodbc
import pickle
from sklearn.preprocessing import StandardScaler
import datetime

# set up logfile to record timings of the various processes:
timings_logfile = open( 
                'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//timings.log',
                'w+'
            )
timings_logfile.write( f'start: {datetime.datetime.now().strftime("%H:%M:%S")}\n' ) 

# set up connection to SQL server:
server = 'pwbhcdsci01'
database = 'data_staging'
params_data_staging = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};SERVER=pwbhcdsci01;DATABASE=data_staging;Trusted_connection=yes;")
sql_engine_data_staging = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params_data_staging, fast_executemany=True)
params_analytics = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};SERVER=pwbhcdsci01;DATABASE=Analytics;Trusted_connection=yes;")
sql_engine_analytics = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params_analytics, fast_executemany=True)
dss_conn = pyodbc.connect("Driver={SQL Server};"
                          "Server=PWBHCDSCI01;"
                          "Database=Analytics;"
                          "uid=;pwd=",autoCommit = True
)
csr = dss_conn.cursor()

# global user inputs ------------------------- #
gbl_inpt_control_group_proportion = 0.1
gbl_inpt_n_customers = 50000
gbl_start_date = '2019-01-01'
gbl_end_date = '2020-03-01'
gbl_campaign_start_date = '2019-07-09'
gbl_n_datasets = 10
gbl_plot_ymin = -15
gbl_plot_ymax = 15
# -------------------------------------------- #

# load order and spend prediction models:
spend_predict_xgb = pickle.load( open('E://Solutions//2101_ExperimentalGroupAssignment//experimentation//saved_spend_predicting_model.obj', 'rb') ) 
order_predict_xgb = pickle.load( open('E://Solutions//2101_ExperimentalGroupAssignment//experimentation//saved_order_prob_model.obj', 'rb') )  
data_scaler = pickle.load( open('E://Solutions//2101_ExperimentalGroupAssignment//experimentation//saved_scaler.obj', 'rb') ) 

def generate_random_dataset_example( start_date, end_date, campaign_start_date, n_customers ):
    dataset = pd.read_sql( 
                f"""
                        SELECT
                                    dmt.[PersonNo]
                                ,   all_periods.[CalendarYearNo]
                                ,   all_periods.[WeekOfYear]
                                ,   ROW_NUMBER() OVER ( PARTITION BY    dmt.[PersonNo]
                                                         ORDER BY       all_periods.[CalendarYearNo]
                                                                    ,   all_periods.[WeekOfYear]
                                                    ) 
                                                    AS [period]
                                ,   CASE WHEN campaign_indicator.[in_campaign] IS NULL THEN 0 ELSE 1 END AS [in_campaign]
                                ,   CASE WHEN all_orders.[sum_OrdPrice] IS NULL THEN 0 ELSE all_orders.[sum_OrdPrice] END AS [spend] 
                        FROM 
                                -- select {n_customers} random customers:
                                    (    
                                        SELECT 
                                                    TOP {n_customers} [PersonNo]
                                        FROM 
                                                    [data_staging].[extract].[HC_DW_dmtlistmodel_history]
                                        WHERE 
                                                    [CycleDateKey] = {campaign_start_date[0:4]+campaign_start_date[5:7]+'09'}
                                        AND 
                                                    [RFM_Segment] < 11 
                                        ORDER BY 
                                                    NEWID()       -- random 
                                    )
                                    dmt
                        LEFT JOIN
                                    -- make a row for every customer/week combination:
                                    (
                                        SELECT 
                                                DISTINCT 
                                                    -99 AS [PersonNo]
                                                ,   [CalendarYearNo]
                                                ,   [WeekOfYear]
                                        FROM
                                                    [data_staging].[extract].[HC_DW_dim_date]
                                        WHERE
                                                    [Date] >= '{start_date}'
                                        AND
                                                    [Date] < '{end_date}'
                                    )
                                    all_periods
                        ON
                                    dmt.[PersonNo] <> all_periods.[PersonNo]
                        LEFT JOIN
                                    -- add column indicating which weeks are "campaign weeks"
                                    -- ..to simplify, I make all weeks after the week containing the campaign_start_date are "campaign weeks"
                                    (
                                        SELECT 
                                                DISTINCT
                                                    1 AS [in_campaign]
                                                ,   [CalendarYearNo]
                                                ,   [WeekOfYear]
                                        FROM
                                                    [data_staging].[extract].[HC_DW_dim_date]
                                        WHERE
                                                    [Date] >= '{campaign_start_date}'                                                    
                                    )
                                    campaign_indicator
                        ON
                                    all_periods.[CalendarYearNo] = campaign_indicator.[CalendarYearNo]
                        AND
                                    all_periods.[WeekOfYear] = campaign_indicator.[WeekOfYear]
                        LEFT JOIN
                                    -- get sales per week for each customer:
                                    (
                                        SELECT 
                                                    orders.[PersonNo]
                                                ,   dimdate.[CalendarYearNo]
                                                ,   dimdate.[WeekOfYear]
                                                ,   SUM(OrdPrice) AS sum_OrdPrice
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
                                                                [Date] >= '{start_date}'
                                                        AND
                                                                [Date] < '{end_date}'
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
                                                                    [OrdStatusGrp] IN ('Intake','Pass','Despatch')
                                                    )
                                                    dimOrdStatus
                                        ON
                                                    orders.[CurrOrdStatus] = dimOrdStatus.[OrdStatusKey]
                                        GROUP BY 
                                                    orders.[PersonNo]
                                                ,   dimdate.[CalendarYearNo]
                                                ,   dimdate.[WeekOfYear]                                        
                                    )
                                    all_orders
                        ON
                                    dmt.[PersonNo] = all_orders.[PersonNo]
                        AND 
                                    all_periods.[CalendarYearNo] = all_orders.[CalendarYearNo]
                        AND 
                                    all_periods.[WeekOfYear] = all_orders.[WeekOfYear]
                """,
                con = sql_engine_data_staging
                )
    return dataset

class dataset_example:
    def __init__(self, dataset_id, rawdata, control_proportion):
        self.dataset_id = dataset_id
        self.control_proportion = control_proportion
        self.rawdata = rawdata
        self.customer_list = rawdata['PersonNo'].drop_duplicates().reset_index()['PersonNo'].values
        self.group_assignments = {}
        self.group_differences = {}
        self.model_predictions = { 
                                    'spend_predict_xgboost':[],
                                    'order_predict_xgboost':[]
                                }
        self.scaled_precampaign_features = []
        self.simple_rfm_features = []
        # initialise each algorithm with random group assignments:
        for alg in (
                    'random_assignment',
                    'best_random_assignment',
                    'simulated_annealing',
                    'stratify_on_recency',
                    'stratify_on_predicted_spend_xgboost',
                    'stratify_on_predicted_order_prob_xgboost'    
                ):
            self.group_assignments[alg] = pd.DataFrame( 
                                                        {
                                                            'PersonNo':self.customer_list,
                                                            'treatment_group':np.random.binomial(1, 1-control_proportion, size=len(self.customer_list) )
                                                        }
                                                       )
            # define group differences data structure:
            self.group_differences[alg] = {
                                           'per_period_diff':None,
                                           'overall_mean_diff':{ 
                                                                    # average per-period difference between groups
                                                                    'pre_campaign':None,
                                                                    'in_campaign':None
                                                                },
                                            'overall_sum_abs_period_diffs':{    
                                                                    # sum of absolute per-period differences between groups
                                                                    'pre_campaign':None,
                                                                    'in_campaign':None
                                                                }                                                                    
                                        } 
    def calc_group_diff( self, alg ):
        # this function summarises the differences between the treatment and control group (per period and overall) for the given algorithm
        calc_per_period_group_diff = (
            self.rawdata
            .merge( 
                    # add treatment/control group assignment of each person:
                    self.group_assignments[alg],
                    on = 'PersonNo'
                )
            .groupby( ['period','in_campaign','treatment_group'] )
            .agg( mean_spend_per_person = ('spend','mean')  )
            .reset_index() 
            .pivot( index=['period','in_campaign'], columns='treatment_group', values='mean_spend_per_person')
            .reset_index()
            .assign( diff = lambda x: x[1]-x[0] )
        )
        self.group_differences[alg]['per_period_diff'] = calc_per_period_group_diff
        self.group_differences[alg]['overall_mean_diff']['pre_campaign'] = calc_per_period_group_diff.query('in_campaign==0')['diff'].mean()
        self.group_differences[alg]['overall_mean_diff']['in_campaign'] = calc_per_period_group_diff.query('in_campaign==1')['diff'].mean()
        self.group_differences[alg]['overall_sum_abs_period_diffs']['pre_campaign'] = calc_per_period_group_diff.query('in_campaign==0')['diff'].abs().sum()

    def calc_group_diff_theoretical( self, theoretical_group_assignments ):
        calc_per_period_group_diff = (
            self.rawdata
            .merge( 
                    # add treatment/control group assignment of each person:
                    theoretical_group_assignments,
                    on = 'PersonNo'
                )
            .groupby( ['period','in_campaign','treatment_group'] )
            .agg( mean_spend_per_person = ('spend','mean')  )
            .reset_index() 
            .pivot( index=['period','in_campaign'], columns='treatment_group', values='mean_spend_per_person')
            .reset_index()
            .assign( diff = lambda x: x[1]-x[0] )
        )
        return {
                'pre_campaign_overall_mean_diff':calc_per_period_group_diff.query('in_campaign==0')['diff'].mean(), 
                'in_campaign_overall_mean_diff_in_campaign':calc_per_period_group_diff.query('in_campaign==1')['diff'].mean(),
                'pre_campaign_overall_sum_abs_period_diffs': calc_per_period_group_diff.query('in_campaign==0')['diff'].abs().sum()
                }
        
    def plot_group_differences_per_period( self, alg, custom_ylim=(None,None), export_path=None, plot_title='' ):
        data_for_plot = self.group_differences[alg]['per_period_diff']
        campaign_start_period = data_for_plot.query('in_campaign==1')['period'].min()
        data_for_plot['rolling_mean_diff'] = data_for_plot['diff'].rolling(4).mean()         # using window=4
        # make barplot of difference per period:
        ax = data_for_plot.plot(
                                kind = 'bar', 
                                x = 'period',
                                y = 'diff',
                                figsize = (20,10),
                                title = plot_title,
                                ylabel = 'y',
                                xlabel = 'Period',
                                ylim = custom_ylim
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
                        x = campaign_start_period-1,                # -1 because indexing on plot starts at 0
                        color = 'blue'
                    )
        # if an export_path is provided, then export the figure to a file:             
        if export_path is not None:
            plt.savefig(    
                            export_path,
                            format = 'jpg' 
                        )

    def trigger_create_sql_customerlist( self ):
        (
            self.
            rawdata[['PersonNo']]
            .drop_duplicates()
            .to_sql( 
                        f'temp_customerlist_dataset{self.dataset_id}',
                        if_exists = 'replace',
                        con = sql_engine_analytics,
                        index = False
                    )
        )

    def trigger_create_customer_features( self ):                                         
        csr.execute( '''EXEC [feature_store_prod].[dbo].[Create_Features] 
                            @period_start_date=?,                    -- period start date to use (note that some feature tables use only @period_end_date)
                            @period_end_date=?,                      -- period end date to use (note: used by many tables as the cycle end date)
                            @customerlist_location=?,                -- SQL location of customer list
						    @desired_output_location=?,              -- provide desired location for final output of this script (e.g. 'feature_store_dev.dbo.CategoryPropensity_Training_Features') - will be created if it doesn't exist
							@append_to_existing=?,                   -- 0=delete existing feature table; 1=append results to existing feature table,
                            @feature_table_list=?                    -- pipe-delimited list of feature tables to pull data from (see list below)                   
                    ''',
              ( 
                  gbl_start_date                                                                    # @period_start_date (can be same as period end date because doesn't matter for the tables that I'm using)
                , gbl_campaign_start_date                                                           # @period_end_date
                , f'[Analytics].[dbo].[temp_customerlist_dataset{self.dataset_id}]'                 # @customerlist_location
                , f'[Analytics].[dbo].[temp_precampaign_features_dataset{self.dataset_id}]'         # @desired_output_location
                , 0                                                                                 # @append_to_existing:  1=yes, 0=no
                , 'hc_people|hc_orders|email6|rfm_atb|prod_productline_hist|recent_productline_seq' # @feature_table_list
              )    
            )

    def fetch_customer_features( self ):
        print( 'fetching customer features from SQL' )
        self.raw_precampaign_features = pd.read_sql( 
                    f"""
                        SELECT 
                                    *
                        FROM 
                                    [Analytics].[dbo].[temp_precampaign_features_dataset{self.dataset_id}]
                    """,
                    con = sql_engine_analytics
        ).fillna(0)          # replace NaN values with 0
        print( 'scaling customer features' )
        self.scaled_precampaign_features = data_scaler.transform( self.raw_precampaign_features.iloc[:,3:] )

    def get_model_predictions( self, model='model_name_here' ):
        if model == 'spend_predict_xgboost':
            self.model_predictions['spend_predict_xgboost'] = pd.DataFrame(
                                                                    {
                                                                        'PersonNo':self.raw_precampaign_features['PersonNo'].values,
                                                                        'model_score': spend_predict_xgb.predict( self.scaled_precampaign_features )
                                                                    }
                                                                )
        elif model == 'order_predict_xgboost':
            self.model_predictions['order_predict_xgboost'] = pd.DataFrame(
                                                                    {
                                                                        'PersonNo':self.raw_precampaign_features['PersonNo'].values,
                                                                        'model_score': order_predict_xgb.predict_proba( self.scaled_precampaign_features )[:,1]
                                                                    }
                                                                )
        else:
            print( '<<< model incorrectly specified >>>' )

    def fetch_simple_rfm_features( self ):
        # still need to write this function
        self.simple_rfm_features = """
            SELECT 
                        customerlist.[PersonNo]
                    ,	COUNT( DISTINCT OrderDteKey ) AS [num_orders_total]
                    ,	CASE WHEN SUM(OrdPrice) IS NULL THEN 0 ELSE SUM(OrdPrice) END AS [sum_OrdPrice_total]
                    ,	MIN(n_days_ago) AS [n_days_ago_last_order]
            FROM
                        [Analytics].[dbo].[temp_customerlist_dataset0] customerlist
            LEFT JOIN
                        (
                            SELECT 
                                        *
                                    ,	DATEDIFF(DAY,Date,'2019-07-09') AS [n_days_ago]
                            FROM
                                        [data_staging].[extract].[HC_DW_factOrderTruncate] orders
                            INNER JOIN
                                    (
                                        SELECT 
                                                [DateKey]
                                            ,	[Date]
                                        FROM
                                                [data_staging].[extract].[HC_DW_dim_date]
                                        WHERE
                                                [DateKey] >= 20190101
                                        AND
                                                [DateKey] < 20190709
                                    )
                                    dimdate 
                            ON
                                    orders.[OrderDteKey] = dimdate.[DateKey]
                                        
                            INNER JOIN
                                    (
                                        SELECT 
                                                [OrdStatusKey]
                                        FROM 
                                                [data_staging].[extract].[HC_DW_dimOrdStatus]
                                        WHERE
                                                [OrdStatusGrp] IN ('Intake','Pass','Despatch')
                                    )
                                    dimOrdStatus
                            ON
                                    orders.[CurrOrdStatus] = dimOrdStatus.[OrdStatusKey]
                        )
                        orderdata
            ON
                        customerlist.[PersonNo] = orderdata.[PersonNo]
            GROUP BY
                        customerlist.[PersonNo]
            ;		
        """

all_datasets = []

timings_logfile.write( f'begin create datasets: {datetime.datetime.now().strftime("%H:%M:%S")}\n' ) 
for i in range(gbl_n_datasets):
    all_datasets.append( 
            dataset_example(
                                dataset_id = i,
                                rawdata = generate_random_dataset_example(  start_date = gbl_start_date,
                                                                            end_date = gbl_end_date,
                                                                            campaign_start_date = gbl_campaign_start_date,
                                                                            n_customers = gbl_inpt_n_customers
                                                                        ), 
                                control_proportion = gbl_inpt_control_group_proportion
                            )
    )
    # calculate initial group differences for certain algorithms:
    all_datasets[i].calc_group_diff(alg='random_assignment')    
    all_datasets[i].calc_group_diff(alg='best_random_assignment')    
    all_datasets[i].calc_group_diff(alg='simulated_annealing')    

    # trigger creation of customer list on SQL db for this dataset:
    all_datasets[i].trigger_create_sql_customerlist()

timings_logfile.write( f'complete create datasets: {datetime.datetime.now().strftime("%H:%M:%S")}\n' ) 

# generate pre-campaign customer features for each dataset:
for i in range(gbl_n_datasets):
    print( f'--generating pre-campaign customer features for dataset {i}--' )
    all_datasets[i].trigger_create_customer_features()
timings_logfile.write( f'complete create pre-campaign features: {datetime.datetime.now().strftime("%H:%M:%S")}\n' ) 

# fetch pre-campaign customer features for each dataset:
for i in range(gbl_n_datasets):
    print( f'--fetching pre-campaign customer features for dataset {i}--' )
    all_datasets[i].fetch_customer_features()    
timings_logfile.write( f'complete fetch pre-campaign features: {datetime.datetime.now().strftime("%H:%M:%S")}\n' ) 

# fetch model predictions for each dataset:
for i in range(gbl_n_datasets):
    print( f'--fetching model predictions for dataset {i}--' )
    all_datasets[i].get_model_predictions( model='spend_predict_xgboost' )   
    all_datasets[i].get_model_predictions( model='order_predict_xgboost' )   
timings_logfile.write( f'complete fetch model predictions: {datetime.datetime.now().strftime("%H:%M:%S")}\n' ) 

# here is an illustration of some of the attributes and functions of each dataset object:
# all_datasets[1].__dict__.keys()     # see what's stored in the dataset object
# all_datasets[1].group_differences['random_assignment']['per_period_diff']
# all_datasets[1].group_differences['best_random_assignment']['overall_mean_diff']['pre_campaign']
# all_datasets[1].plot_group_differences_per_period( alg='random_assignment' )
# all_datasets[1].calc_group_diff_theoretical( theoretical_group_assignments=pd.DataFrame( 
#                                                         {
#                                                             'PersonNo':all_datasets[1].customer_list,
#                                                             'treatment_group':np.random.binomial( n=1, p=1-gbl_inpt_control_group_proportion, size=gbl_inpt_n_customers )
#                                                         }
#                                                        ) 
#                                             )

# GROUP ASSIGNMENT METHOD:
# +--------------------------+
# | Purely Random Assignment | 
# +--------------------------+
for i in range(gbl_n_datasets):
    all_datasets[i].plot_group_differences_per_period( 
                                                    alg = 'random_assignment',
                                                    custom_ylim = (gbl_plot_ymin, gbl_plot_ymax),
                                                    export_path = f'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//dataset{i}_random_assignment.jpg',
                                                    plot_title = f'dataset {i}: alg=random_assignment' 
                                                )
timings_logfile.write( f'complete export random assignment plots: {datetime.datetime.now().strftime("%H:%M:%S")}\n' )                                                 

# GROUP ASSIGNMENT METHOD:
# +-----------------------------+
# | Stratify on predicted spend | 
# +-----------------------------+
for i in range(gbl_n_datasets):
    print( f'Group assignment by XG-Boost Predicted Spend: Dataset {i}' )
    treatment_to_control_ratio = int( round(1/gbl_inpt_control_group_proportion, 0) )
    predicted_spend_values = all_datasets[i].model_predictions['spend_predict_xgboost']
    predicted_spend_values = predicted_spend_values.sort_values('model_score', ascending=False).reset_index()
    predicted_spend_values['treatment_group'] = 1    # initially assign all to treatment
    predicted_spend_values.loc[ predicted_spend_values.index%treatment_to_control_ratio==0, 'treatment_group'] = 0   # assign every [treatment_to_control_ratio]th customer to control group
    all_datasets[i].group_assignments['stratify_on_predicted_spend_xgboost'] = predicted_spend_values[ ['PersonNo','treatment_group'] ]
    all_datasets[i].calc_group_diff( alg='stratify_on_predicted_spend_xgboost' )
    all_datasets[i].plot_group_differences_per_period( 
                                                        alg = 'stratify_on_predicted_spend_xgboost',
                                                        custom_ylim = (gbl_plot_ymin, gbl_plot_ymax),
                                                        export_path = f'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//dataset{i}_stratify_on_predicted_spend_xgboost.jpg',
                                                        plot_title = f'dataset {i}: alg=stratify_on_predicted_spend_xgboost' 
                                                    )
timings_logfile.write( f'complete stratify on predicted spend: {datetime.datetime.now().strftime("%H:%M:%S")}\n' )                                                                                                     

# GROUP ASSIGNMENT METHOD:
# +-----------------------------------------+
# | Stratify on predicted Order Probability | 
# +-----------------------------------------+
for i in range(gbl_n_datasets):
    print( f'Group assignment by XG-Boost Predicted Order Probability: Dataset {i}' )
    treatment_to_control_ratio = int( round(1/gbl_inpt_control_group_proportion, 0) )
    order_prob_values = all_datasets[i].model_predictions['order_predict_xgboost']
    order_prob_values = order_prob_values.sort_values('model_score', ascending=False).reset_index()
    order_prob_values['treatment_group'] = 1    # initially assign all to treatment
    order_prob_values.loc[ order_prob_values.index%treatment_to_control_ratio==0, 'treatment_group'] = 0   # assign every [treatment_to_control_ratio]th customer to control group
    all_datasets[i].group_assignments['stratify_on_predicted_order_prob_xgboost'] = order_prob_values[ ['PersonNo','treatment_group'] ]
    all_datasets[i].calc_group_diff( alg='stratify_on_predicted_order_prob_xgboost' )
    all_datasets[i].plot_group_differences_per_period( 
                                                        alg = 'stratify_on_predicted_order_prob_xgboost',
                                                        custom_ylim = (gbl_plot_ymin, gbl_plot_ymax),
                                                        export_path = f'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//dataset{i}_stratify_on_predicted_order_prob_xgboost.jpg',
                                                        plot_title = f'dataset {i}: alg=stratify_on_predicted_order_prob_xgboost' 
                                                    )
timings_logfile.write( f'complete stratify on order probability: {datetime.datetime.now().strftime("%H:%M:%S")}\n' )                                                                                                     

# GROUP ASSIGNMENT METHOD:
# +---------------------+
# | Stratify on Recency | 
# +---------------------+
for i in range(gbl_n_datasets):
    print( f'-- Stratify on Recency: dataset {i} --' )
    treatment_to_control_ratio = int( round(1/gbl_inpt_control_group_proportion, 0) )
    # generate treatment group assigments by stratifying on "days since last order", then "days since first inserted in HC database":
    all_datasets[i].group_assignments['stratify_on_recency'] = pd.read_sql(
        f"""
                SELECT
                            customerlist.[PersonNo]
                            -- ,   order_recency.[days_since_most_recent_order]
                            -- ,   days_in_hc_database.[days_on_HC_db]
                        ,   CASE WHEN ROW_NUMBER() OVER ( 
                                                           ORDER BY 
                                                                    order_recency.[days_since_most_recent_order]
                                                                ,   days_in_hc_database.[days_on_HC_db]   
                                                        ) 
                                                        % {treatment_to_control_ratio} = 0 THEN 0 ELSE 1 END AS [treatment_group]
                FROM 
                            [Analytics].[dbo].[temp_customerlist_dataset{i}] customerlist
                LEFT JOIN
                            -- get days since most recent order
                            (
                                SELECT 
                                            [PersonNo]
                                        ,   DATEDIFF( 
                                                        day, 
		                                                CAST(cast( MAX(OrderDteKey) as char(8)) AS DATETIME), 
					                                    GETDATE()
					                                )
					                                AS [days_since_most_recent_order]
                                FROM 
                                            [data_staging].[extract].[HC_DW_factOrderTruncate] 
                                WHERE
                                            -- not cancelled orders or goods returned
                                            [CurrOrdStatus] IN (  
                                                                    SELECT 
                                                                            [OrdStatusKey]
                                                                    FROM
                                                                            [data_staging].[extract].[HC_DW_dimOrdStatus]
                                                                    WHERE
                                                                            [OrdStatusGrp] IN ('Intake','Pass','Despatch')
                                            )                                            
                                GROUP BY
                                            [PersonNo]
                            )
                            order_recency
                ON
                            customerlist.[PersonNo] = order_recency.[PersonNo]
                LEFT JOIN 
                            -- get number of days since first in HC database
                            (
                                SELECT 
                                            [PersonNo]
                                        ,   DATEDIFF( 
                                                        day, 
                                                        CAST( cast( MIN(StartDteKey) as CHAR(8)) AS DATETIME ), 
                                                        GETDATE()
                                                    )
                                                    AS [days_on_HC_db]
                                FROM 
                                            [data_staging].[extract].[HC_DW_dimPerson]
                                GROUP BY
                                            [PersonNo]
                            ) 
                            days_in_hc_database
                ON
                            customerlist.[PersonNo] = days_in_hc_database.[PersonNo]
                ORDER BY
                            order_recency.[days_since_most_recent_order]
                        ,   days_in_hc_database.[days_on_HC_db]                            
                ;
        """,
        con = sql_engine_data_staging
    )
    # calculate group differences
    all_datasets[i].calc_group_diff( alg='stratify_on_recency' )
    all_datasets[i].plot_group_differences_per_period( 
                                                        alg = 'stratify_on_recency',
                                                        custom_ylim = (gbl_plot_ymin, gbl_plot_ymax),
                                                        export_path = f'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//dataset{i}_stratify_on_recency.jpg',
                                                        plot_title = f'dataset {i}: alg=stratify_on_recency' 
                                                    )    
timings_logfile.write( f'complete stratify on recency: {datetime.datetime.now().strftime("%H:%M:%S")}\n' )                                                 

# run 'best random assignment':
"""
:: Algorithm ::
user inputs: 
    n_iter   number of random starts to try

1.  assess the quality of r random start points, selecting the best one from among them.
"""
n_iter = 1000                                         # number of random group assignments to assess (user-defined parameter)
for dataset_i in range(gbl_n_datasets):
    # initialise best assignment:
    best_assignment = all_datasets[dataset_i].group_assignments['best_random_assignment']  
    current_best_obj_value = abs( all_datasets[dataset_i].group_differences['best_random_assignment']['overall_sum_abs_period_diffs']['pre_campaign'] )
    for iteration_j in range(n_iter):
        print( f'-- BEST RANDOM ASSIGNMENT dataset {dataset_i} iteration {iteration_j}' )
        # generate a random group assignment:
        random_group_assignment = pd.DataFrame( 
                                                {
                                                    'PersonNo':all_datasets[dataset_i].customer_list,
                                                    'treatment_group':np.random.binomial( n=1, p=1-gbl_inpt_control_group_proportion, size=gbl_inpt_n_customers )
                                                }
                                            )
        obj_value_this_random_group_assignment = abs( all_datasets[dataset_i].calc_group_diff_theoretical( random_group_assignment )['pre_campaign_overall_sum_abs_period_diffs'] )
        if obj_value_this_random_group_assignment < current_best_obj_value:
            print( f'new objective value ({obj_value_this_random_group_assignment}) better (lower) than previous best ({current_best_obj_value})')
            best_assignment = random_group_assignment
            current_best_obj_value = obj_value_this_random_group_assignment
        else:
            print( f'new objective value ({obj_value_this_random_group_assignment}) worse than previous best ({current_best_obj_value})')
    
    # save best assignment, and export results:
    all_datasets[dataset_i].group_assignments['best_random_assignment'] = best_assignment
    all_datasets[dataset_i].calc_group_diff( alg='best_random_assignment' )
    all_datasets[dataset_i].plot_group_differences_per_period( 
                                                        alg = 'best_random_assignment',
                                                        custom_ylim = (gbl_plot_ymin, gbl_plot_ymax),
                                                        export_path = f'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//dataset{dataset_i}_best_random_assignment.jpg',
                                                        plot_title = f'dataset {dataset_i}: alg=best_random_assignment' 
                                                    ) 
timings_logfile.write( f'complete best random assignment: {datetime.datetime.now().strftime("%H:%M:%S")}\n' )                                                                                                     


# run simulated annealing 
# define algorithm parameters:
n_iter = 500 
n_customer_changes_per_iter = 50
temp = 5             # starting temp
a = 0.98
## print out the temperature schedule:
#example_schedule = pd.DataFrame( { 'iteration':[i for i in range(r)] } )
#example_schedule['start_temp'] = temp
#example_schedule['start_a'] = a
#example_schedule['temp'] = example_schedule['start_temp'] * ( example_schedule['start_a'] ** example_schedule['iteration'] )
#example_schedule['prob_accept_worsen_by_0.01'] = [ math.exp( -0.01 / i ) for i in example_schedule['temp'].values ] 
#example_schedule['prob_accept_worsen_by_0.1'] = [ math.exp( -0.1 / i ) for i in example_schedule['temp'].values ] 
#example_schedule['prob_accept_worsen_by_1.0'] = [ math.exp( -1.0 / i ) for i in example_schedule['temp'].values ] 
#example_schedule['prob_accept_worsen_by_10.0'] = [ math.exp( -10.0 / i ) for i in example_schedule['temp'].values ] 
#example_schedule.index = example_schedule['iteration']
#ax = example_schedule[['prob_accept_worsen_by_0.01','prob_accept_worsen_by_0.1','prob_accept_worsen_by_1.0','prob_accept_worsen_by_10.0']].plot.line() 
#plt.show()

for dataset_i in range(gbl_n_datasets):
    # define (initial) algorithm parameters:
    n_iter = 500 
    n_customer_changes_per_iter = 50
    temp = 5             # starting temp
    a = 0.98
    best_assignment = all_datasets[dataset_i].group_assignments['simulated_annealing']  
    current_best_obj_value = abs( all_datasets[dataset_i].group_differences['simulated_annealing']['overall_sum_abs_period_diffs']['pre_campaign'] )
    #store_obj_value_per_iteration = []
    for iteration_j in range(n_iter):
        rand01 = random.random()
        print(f'iteration {iteration_j}     temp={temp}   obj={round(current_best_obj_value,4)}')
        alt_assignment = best_assignment.copy()
        indices_to_flip = random.sample( list(alt_assignment.index), n_customer_changes_per_iter )
        alt_assignment.loc[ indices_to_flip, 'treatment_group' ] = -1*alt_assignment.loc[indices_to_flip, 'treatment_group'] +1          # flip the indices
        obj_value_of_alt_assignment = all_datasets[dataset_i].calc_group_diff_theoretical( alt_assignment )['pre_campaign_overall_sum_abs_period_diffs']
        print(f'objective value of proposed assignment: {obj_value_of_alt_assignment}')
        change_in_obj = current_best_obj_value - obj_value_of_alt_assignment
        print(f'change in objective by switching to new proposed assignment: {change_in_obj}')
        if change_in_obj > 0:
            print('accept proposed change')
            best_assignment = alt_assignment
            current_best_obj_value = obj_value_of_alt_assignment
            #store_obj_value_per_iteration.append( current_best_obj_value )
        elif change_in_obj <= 0 and rand01 < math.exp(change_in_obj/temp):
            print(f'runif01 {rand01}  <=  change prob {math.exp(change_in_obj/temp)}  => accept suboptimal change')
            best_assignment = alt_assignment
            current_best_obj_value = obj_value_of_alt_assignment
            #store_obj_value_per_iteration.append( current_best_obj_value )        
        else:
            print(f'runif01 {rand01}  >  change prob {math.exp(change_in_obj/temp)}  => reject suboptimal change')
            #store_obj_value_per_iteration.append( current_best_obj_value )
            # reduce temperature:    
            temp = temp*a
            print('\n\n')
    # save final best assignment:
    all_datasets[dataset_i].group_assignments['simulated_annealing'] = best_assignment
    all_datasets[dataset_i].calc_group_diff( alg='simulated_annealing' )
    all_datasets[dataset_i].plot_group_differences_per_period( 
                                                    alg = 'simulated_annealing',
                                                    custom_ylim = (gbl_plot_ymin, gbl_plot_ymax),
                                                    export_path = f'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//dataset{dataset_i}_simulated_annealing.jpg',
                                                    plot_title = f'dataset {dataset_i}: alg=simulated_annealing' 
                                                ) 
timings_logfile.write( f'complete simulated annealing: {datetime.datetime.now().strftime("%H:%M:%S")}\n' )                                                                                                 

# close SQL connection
csr.close()
dss_conn.close()

timings_logfile.write( f'full process completed: {datetime.datetime.now().strftime("%H:%M:%S")}\n' )                                                 
timings_logfile.close()

# save results of experiment:
# I don't actually do this since it ends up being 7Gb for 10 customer datasets of 50k customers in each :o
#experiment_results_save_location = 'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//experiment_results.obj'
#pickle.dump( 
#                all_datasets,
#                open(experiment_results_save_location, 'wb')
#           )

# collect data across all datasets and algorithms:
assemble_results_for_plots = []
for dataset_i in range(gbl_n_datasets):
    for alg_j in (
                    'random_assignment',
                    'best_random_assignment',
                    'simulated_annealing',
                    'stratify_on_recency',
                    'stratify_on_predicted_spend_xgboost',
                    'stratify_on_predicted_order_prob_xgboost'
                ):
        data_for_plot_ij = all_datasets[dataset_i].group_differences[alg_j]['per_period_diff']
        data_for_plot_ij['dataset'] = dataset_i
        data_for_plot_ij['alg'] = alg_j
        assemble_results_for_plots.append( data_for_plot_ij )

data_for_plot = pd.concat( assemble_results_for_plots ) 
campaign_start_period = data_for_plot.query('in_campaign==1')['period'].min()

# export faceted lineplots per dataset:
for dataset_i in range(gbl_n_datasets):
    plotdata_dataset_i = data_for_plot.query( f'dataset == {dataset_i}' ).copy()
    #plotdata_dataset_i.index = plotdata_dataset_i['period']
    plotdata_dataset_i = plotdata_dataset_i[ ['period','alg','diff'] ].pivot(index='period',values='diff',columns='alg')
    # make lineplot of difference per period:
    ax = plotdata_dataset_i.plot(
                                kind = 'line', 
                                figsize = (20,10),
                                title = f'Dataset {dataset_i}',
                                ylabel = 'diff',
                                xlabel = 'Period',
                                ylim = (gbl_plot_ymin, gbl_plot_ymax),
                                subplots = True,
                                layout = (3,2)     # layout of subplots
                        )
    # draw in constant lines on each plot:                    
    for subplot_row_i in range(len(ax)):
        for subplot_col_j in range(len(ax[0])):
            ax[subplot_row_i][subplot_col_j].axvline( x = campaign_start_period, color = 'blue' )             # -1 because indexing on plot starts at 0
            ax[subplot_row_i][subplot_col_j].axhline( y = -5, color = 'grey' )         
            ax[subplot_row_i][subplot_col_j].axhline( y = 5, color = 'grey' )         
            ax[subplot_row_i][subplot_col_j].axhline( y = 0, color = 'grey', linestyle='--' )
    
    # export plot to file:
    plt.savefig(    
                    f'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//results_facetline_byweek_summary_dataset{dataset_i}.jpg',
                    format = 'jpg' 
                )

    # create cumulative sum of differences plots:


# look at in-campaign performance of methods across all datasets:
cross_data_alg_comparison = data_for_plot.query('in_campaign==1').copy()
cross_data_alg_comparison['absolute_diff'] = abs(cross_data_alg_comparison['diff'])
avg_absolute_diff_per_alg_per_dataset = (
    cross_data_alg_comparison
    .groupby( ['alg','dataset'] )
    .agg( mean_absolute_diff = ('absolute_diff', 'mean') )
    .reset_index()
)
ax = avg_absolute_diff_per_alg_per_dataset.plot(
                                                    kind = 'scatter', 
                                                    figsize = (20,10),
                                                    x = 'alg',
                                                    y = 'mean_absolute_diff',
                                                    title = 'Comparison of Group-Assignment Algorithms across all Experimental Datasets',
                                                    ylabel = 'Mean Absolute Difference Between Treatment & Holdout Groups',
                                                    xlabel = 'Group-Assignment Algorithm',
                                                    alpha = 0.5,
                                                    s = 100             # point size
                        )
plt.xticks(rotation=45)                        
plt.savefig(    
                'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//overall_summary_mean_absolute_diff.jpg',
                format = 'jpg' 
            )

# Kyle asked me have a look at cumulative sum of (not absolute) campaign differences per alg:
for dataset_i in range(gbl_n_datasets):
    plotdata_dataset_i = data_for_plot.query( f'dataset == {dataset_i} & in_campaign==1' ).copy()
    plotdata_dataset_i['cumulative_diff'] = plotdata_dataset_i.groupby('alg')['diff'].cumsum()
    #plotdata_dataset_i.index = plotdata_dataset_i['period']
    plotdata_dataset_i = plotdata_dataset_i[ ['period','alg','cumulative_diff'] ].pivot(index='period',values='cumulative_diff',columns='alg')
    # make lineplot of cumulative difference per campaign period:
    ax = plotdata_dataset_i.plot(
                                kind = 'line', 
                                figsize = (20,10),
                                title = f'Dataset {dataset_i}',
                                ylabel = 'cumulative_diff',
                                xlabel = 'Period',
                                ylim = (-50, 50),
                                subplots = True,
                                layout = (3,2)     # layout of subplots
                        )
    # draw in constant lines on each plot:                    
    for subplot_row_i in range(len(ax)):
        for subplot_col_j in range(len(ax[0])):
            ax[subplot_row_i][subplot_col_j].axhline( y = -5, color = 'grey' )         
            ax[subplot_row_i][subplot_col_j].axhline( y = 5, color = 'grey' )         
            ax[subplot_row_i][subplot_col_j].axhline( y = 0, color = 'grey', linestyle='--' )     
    plt.savefig(    
                f'E://Solutions//2101_ExperimentalGroupAssignment//experimentation//experiment_results//cumulative_difference_plot_dataset{dataset_i}.jpg',
                format = 'jpg' 
            )                       