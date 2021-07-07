
import pandas as pd
import numpy as np

def generate_random_dataset_example( n_periods, campaign_start_period, n_PersonNo ):
    random_dataset = pd.DataFrame( [ (PersonNo,period) for PersonNo in range(n_PersonNo) for period in range(n_periods) ] )
    random_dataset.columns = ['PersonNo','period']
    random_dataset['spend'] = np.random.uniform( low=10, high=100, size=len(random_dataset) ).round(2)
    random_dataset['eval_period'] = np.select( condlist=[random_dataset['period']<campaign_start_period], choicelist=['pre_campaign'], default='in_campaign' )
    return random_dataset

generate_random_dataset_example( n_periods=9, campaign_start_period=5, n_PersonNo=3 )

class dataset_example:
    def __init__(self, dataset_id, rawdata, control_proportion):
        self.dataset_id = dataset_id
        self.control_proportion = control_proportion
        self.rawdata = rawdata
        self.PersonNo_list = rawdata['PersonNo'].drop_duplicates().reset_index()['PersonNo'].values
        self.group_assignments = {}
        self.group_differences = {}
        # initialise each algorithm with random group assignments:
        for alg in ('random_assignment','simulated_annealing'):
            self.group_assignments[alg] = pd.DataFrame( 
                                                        {
                                                            'PersonNo':self.PersonNo_list,
                                                            'treatment_group':np.random.binomial(1, 1-control_proportion, size=len(self.PersonNo_list) )
                                                        }
                                                       )
            # define group differences data structure:
            self.group_differences[alg] = {
                                           'per_period_diff':None,
                                           'overall_mean_diff':{ 
                                                                    'pre_campaign':None,
                                                                    'in_campaign':None
                                                                }    
                                        } 
    def calc_group_diff( self, alg ):
        # this function summarises the differences between the treatment and control group (per period and overall) for the given algorithm
        per_period_group_diff = (
            self.rawdata
            .merge( 
                    # add treatment/control group assignment of each person:
                    self.group_assignments[alg],
                    on='PersonNo'
                )
            .groupby( ['period','eval_period','treatment_group'] )
            .agg( mean_spend_per_person = ('spend','mean')  )
            .reset_index() 
            .pivot( index=['period','eval_period'], columns='treatment_group', values='mean_spend_per_person')
            .reset_index()
            .assign( diff = lambda x: x[1]-x[0] )
        )
        self.group_differences[alg]['per_period_diff'] = per_period_group_diff
        self.group_differences[alg]['overall_mean_diff']['pre_campaign'] = per_period_group_diff.query('eval_period=="pre_campaign"')['diff'].mean()
        self.group_differences[alg]['overall_mean_diff']['in_campaign'] = per_period_group_diff.query('eval_period=="in_campaign"')['diff'].mean()    

    def plot_group_differences_per_period( self, output_path ):
        "still to do"

random_dataset = pd.DataFrame( [ (PersonNo,period) for PersonNo in range(n_PersonNo) for period in range(n_periods) ] )
random_dataset.columns = ['PersonNo','period']
random_dataset['spend'] = np.random.uniform( low=10, high=100, size=len(random_dataset) ).round(2)
random_dataset['eval_period'] = np.select( condlist=[random_dataset['period']<campaign_start_period], choicelist=['pre_campaign'], default='in_campaign' )

test = dataset_example( dataset_id          = 0, 
                        rawdata             = random_dataset,
                         control_proportion = 0.05
                    )                                            
test.calc_group_diff(alg='simulated_annealing')
test.group_differences['random_assignment']
test.group_differences['simulated_annealing']['per_period_diff']
test.group_differences['simulated_annealing']['overall_mean_diff']


# experiment stored in nested dictionary structure:
#
#
# { 
#   dataset_0:{ 
#               rawdata:[],
#               pre_campaign_data:[],
#               in_campaign_data:[],
#               treatment_control_assignments:{
#                                                random:[],
#                                                sim_anneal:[]
#                                             }
#               group_differences:{
#                                    random:{ 
#                                                 precampaign:{
#                                                               per_period:[],
#                                                               mean_overall:[]   
#                                                             },
#                                                 incampaign:{
#                                                               per_period:[],
#                                                               mean_overall:[]   
#                                                             } 
#                                            },
#                                    sim_anneal:{ 
#                                                 precampaign:{
#                                                               per_period:[],
#                                                               mean_overall:[]   
#                                                             },
#                                                 incampaign:{
#                                                               per_period:[],
#                                                               mean_overall:[]   
#                                                             } 
#                                               },
#                                  }
#         
#             }
#   ,
#   dataset_1:{ 
#               rawdata:[],
#               pre_campaign_data:[],
#               in_campaign_data:[],
#               treatment_control_assignments:{
#                                                random:[],
#                                                sim_anneal:[]
#                                             }
#               group_differences:{
#                                    random:{ 
#                                                 pre_campaign:{
#                                                               per_period:[],
#                                                               mean_overall:[]   
#                                                             },
#                                                 in_campaign:{
#                                                               per_period:[],
#                                                               mean_overall:[]   
#                                                             } 
#                                            },
#                                    sim_anneal:{ 
#                                                 pre_campaign:{
#                                                               per_period:[],
#                                                               mean_overall:[]   
#                                                             },
#                                                 in_campaign:{
#                                                               per_period:[],
#                                                               mean_overall:[]   
#                                                             } 
#                                               }
#                                  }
#         
#             }
#   ,
#   .
#   .
#   .
#
# }
#
#
#

all_data = {}

def recalc_group_differences( dataset, group_assignemnts ):
    per_period_group_diff = (
            dataset
            .merge( 
                    group_assignments,
                    on='PersonNo'
                )
            .groupby( ['period','eval_period','treatment_group'] )
            .agg( mean_spend_per_person = ('spend','mean')  )
            .reset_index() 
            .pivot( index=['period','eval_period'], columns='treatment_group', values='mean_spend_per_person')
            .reset_index()
            .assign( diff = lambda x: x[1]-x[0] )
            )
    

# generate random datasets and insert them into {all_data}:
for i in range(8):
    random_dataset = pd.DataFrame( [ (PersonNo,period) for PersonNo in range(n_PersonNo) for period in range(n_periods) ] )
    random_dataset.columns = ['PersonNo','period']
    random_dataset['spend'] = np.random.uniform( low=10, high=100, size=len(random_dataset) ).round(2)
    random_dataset['eval_period'] = np.select( condlist=[random_dataset['period']<campaign_start_period], choicelist=['pre_campaign'], default='in_campaign' )

    random_dataset_PersonNo_list = random_dataset.PersonNo.drop_duplicates().reset_index()['PersonNo'].values

    all_data[ f'dataset_{i}'] = {
                                    'rawdata':random_dataset,
                                    'pre_campaign_data':random_dataset.query('eval_period=="pre_campaign"'),
                                    'in_campaign_data':random_dataset.query('eval_period=="in_campaign"'),
                                    'treatment_control_assignments':{
                                                                      'random':pd.DataFrame( 
                                                                                            {
                                                                                                'PersonNo':random_dataset_PersonNo_list,
                                                                                                'treatment_group':np.random.binomial(1, 1-control_proportion, size=len(random_dataset_PersonNo_list) )
                                                                                            }
                                                                                        ),
                                                                      'sim_anneal':pd.DataFrame( 
                                                                                            {
                                                                                                'PersonNo':random_dataset_PersonNo_list,
                                                                                                'treatment_group':np.random.binomial(1, 1-control_proportion, size=len(random_dataset_PersonNo_list) )
                                                                                            }
                                                                                        )
                                                                    },
                                    'group_differences':{
                                                            'random':{ 
                                                                        'pre_campaign':{
                                                                                        'per_period':[],
                                                                                        'mean_diff':[]   
                                                                                    },
                                                                        'in_campaign':{
                                                                                        'per_period':[],
                                                                                        'mean_diff':[]   
                                                                                    } 
                                                                    },
                                                            'sim_anneal':{ 
                                                                            'pre_campaign':{
                                                                                            'per_period':[],
                                                                                            'mean_diff':[]   
                                                                                        },
                                                                            'in_campaign':{
                                                                                            'per_period':[],
                                                                                            'mean_diff':[]   
                                                                                        } 
                                                                        }
                                                        }                                                                    
                                }
    
    # calculate per-period pre_campaign group differences per period, per algorithm:
    for alg in ('random','sim_anneal'):
        all_data[ f'dataset_{i}']['group_differences'][alg]['pre_campaign']['per_period'] = ( 
            all_data[ f'dataset_{i}']['pre_campaign_data']
            .merge( all_data[f'dataset_{i}']['treatment_control_assignments'][alg], 
                    on='PersonNo'
                )
            .groupby( ['period','treatment_group'] )
            .agg( mean_spend_per_person = ('spend','mean')  )
            .reset_index()
            .pivot( index='period', columns='treatment_group', values='mean_spend_per_person')
            .reset_index()
            .assign( diff = lambda x: x[1]-x[0] )
            )
        # overall pre_campaign mean group difference per algorithm:    
        all_data[ f'dataset_{i}']['group_differences'][alg]['pre_campaign']['mean_diff'] = (
            all_data[ f'dataset_{i}']['group_differences'][alg]['pre_campaign']['per_period']['diff'].mean()
        )

    # calculate per-period in_campaign group differences, per algorithm:
    for alg in ('random','sim_anneal'):
        all_data[ f'dataset_{i}']['group_differences'][alg]['in_campaign']['per_period'] = ( 
            all_data[ f'dataset_{i}']['in_campaign_data']
            .merge( all_data[f'dataset_{i}']['treatment_control_assignments'][alg], 
                    on='PersonNo'
                )
            .groupby( ['period','treatment_group'] )
            .agg( mean_spend_per_person = ('spend','mean')  )
            .reset_index()
            .pivot( index='period', columns='treatment_group', values='mean_spend_per_person')
            .reset_index()
            .assign( diff = lambda x: x[1]-x[0] )
            )
        # overall in_campaign mean group difference per algorithm:    
        all_data[ f'dataset_{i}']['group_differences'][alg]['in_campaign']['mean_diff'] = (
            all_data[ f'dataset_{i}']['group_differences'][alg]['in_campaign']['per_period']['diff'].mean()
        )


all_data['dataset_2']['rawdata'].merge( all_data['dataset_2']['treatment_control_assignments']['sim_anneal'], on='PersonNo' ).to_clipboard()

all_data['dataset_2']['group_differences']['sim_anneal']['pre_campaign']
all_data['dataset_2']['group_differences']['sim_anneal']['in_campaign']













