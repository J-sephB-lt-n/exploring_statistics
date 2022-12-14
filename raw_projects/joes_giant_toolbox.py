
##################################
# last update:  2021/07/08 15:48 #
##################################

def list_available_joe_functions():
    print(
        """
        FUNCTION_NAME                                                    STATUS                USAGE
        faster_experimental_group_balancing_multiobjective_sim_annealing beta                  for balancing experimental groups based on multiple numeric features 
        fastDataExploration_basic_stats_per_column                       in development        outputs basic column-wise statistics for a given pandas dataframe
        generate_itemFeatures_from_quantiles_of_numeric_user_feature     ready to use          creates quantile-based item features by aggregating over a numeric user feature               
        """
    )




def faster_experimental_group_balancing_multiobjective_sim_annealing(
        input_data,                # input data 
        n_iterations,              # number of iterations to run algorithm
        n_switches_per_iteration,  # how many users will switch groups per iteration 
        init_temp,                 # starting temperature for the algorithm 
        cool_param,                # cooling parameter for the algorithm 
        verbose_nsteps,            # results will be printed every [verbose_nsteps] iterations
        time_colname,              # name of the column in [input_data] containing the time variable 
        userID_colname,            # name of the column in [input_data] containing user ID
        group_colname,             # name of the column in [input_data] containing treated/control group assignment       
        value_colnames             # names of the columns in [input_data] which customers are to be balanced on 

        # simulated annealing works as follows:
        #   calculate objective value -O- of initial solution
        #   -O- is current "best solution"
        #   for iteration i in 1:n_iterations...
        #         ..generate a random value -R- in [0,1]
        #         ..perturb/modify the current solution
        #         ..if the proposed modified solution is better than current "best solution" -O-, then proposed solution becomes "best solution" -O-
        #             ..else if proposed modified solution worse than -O- and -R- < exp{ amount_objective_is_worse/temp }, then proposed solution becomes "best solution" -O-
        #                                                                    ( note that amount_objective_is_worse < 0 )
        #             ..else proposed modified solution is rejected
        #         .. temp = temp * cool_param  
        # throughout, I also keep track of the 'global' best objective value found so far
        #         ..even when a suboptimal modification is accepted, the 'global' best solution will not change unless it is the best solution found so far   
    ):
    """
        this function requires {pandas},{numpy},{math},{random} and {collections} to be imported 

        input data must be in the following format:
        time        userID       group       value1     value2   ....as many value columns as required 
         x            x            x           x           x
         x            x            x           x           x
         x            x            x           x           x
        (the actual column names can be anything, but column order and number does matter)
        the group column can only contain the values 'treated' and 'control'

        # example usage:
        import pandas as pd
        pd.set_option('max_rows',100)
        import numpy as np
        import math
        import random
        import collections
        from matplotlib import pyplot as plt
        import seaborn as sns
        input_data = pd.DataFrame( 
                        {
                            'period':list(range(10))*50_000,         # 10 times periods * 50k customers
                            'customerID':sorted(list(range(50_000))*10),
                            'control_group':['treated']*450_000+['control']*50_000,
                            'spend':np.random.uniform(low=5, high=5000, size=500_000),
                            'recency':np.random.choice( range(1,9), size=500_000, replace=True ),
                        } 
                    )
        input_data.loc[ sorted(list(np.random.choice(input_data.index, size=250_000, replace=False))), 'spend' ] = 0          # make 50% of the spend values into 0            
        input_data['active'] = 1
        input_data.loc[ input_data['spend']==0, 'active' ] = 0
        # run the optimisation:
        optimised_groups = faster_experimental_group_balancing_multiobjective_sim_annealing(
                    input_data = input_data,                        # input data 
                    n_iterations = 2000,                            # number of iterations to run algorithm
                    n_switches_per_iteration = 50,                  # how many users will switch groups per iteration 
                    init_temp = 5,                                  # starting temperature for the algorithm 
                    cool_param = 0.98,                              # cooling parameter for the algorithm 
                    verbose_nsteps = 50,                            # results will be printed every [verbose_nsteps] iterations
                    time_colname = 'period',                        # name of the column in [input_data] containing the time variable 
                    userID_colname = 'customerID',                  # name of the column in [input_data] containing user ID
                    group_colname = 'control_group',                # name of the column in [input_data] containing treated/control group assignment       
                    value_colnames = ['spend','recency','active']   # names of the columns in [input_data] which customers are to be balanced on 
        )
        input_data = input_data.merge( optimised_groups, on='customerID' )

        # compare the groups before and after optimisation:        
        store_aggregated_DFs = []
        for value_i in ['spend','recency','active']:
            temp_df = input_data.copy()
            temp_df['value_col'] = temp_df[value_i] 
            summarise_unoptimised_groups = (
                temp_df
                .groupby( ['period', 'control_group'] )
                .agg( mean_value = ('value_col', 'mean') )
                .reset_index()
            )

            summarise_unoptimised_groups = summarise_unoptimised_groups.pivot( index='period', columns='control_group', values='mean_value' ).reset_index()
            summarise_unoptimised_groups['diff'] = summarise_unoptimised_groups['treated'] - summarise_unoptimised_groups['control']
            summarise_unoptimised_groups['value'] = value_i
            summarise_unoptimised_groups['optimised_group'] = 'UNoptimised'

            store_aggregated_DFs.append( summarise_unoptimised_groups[['period','value','diff','optimised_group']] )

            summarise_optimised_groups = (
                temp_df
                .groupby( ['period', 'optimised_control_group'] )
                .agg( mean_value = ('value_col', 'mean') )
                .reset_index()
            )
            summarise_optimised_groups = summarise_optimised_groups.pivot( index='period', columns='optimised_control_group', values='mean_value' ).reset_index()
            summarise_optimised_groups['diff'] = summarise_optimised_groups['treated'] - summarise_optimised_groups['control']
            summarise_optimised_groups['value'] = value_i
            summarise_optimised_groups['optimised_group'] = 'optimised'

            store_aggregated_DFs.append( summarise_optimised_groups[['period','value','diff','optimised_group']] )
        
        aggregated_results = pd.concat( store_aggregated_DFs, axis=0 ) 

        graph = sns.relplot(
                    data = aggregated_results,
                    x = 'period', 
                    y = 'diff',
                    col = 'value',      # this is the variable faceted on 
                    kind = 'line',  
                    hue = 'optimised_group',
                    style = 'optimised_group',
                    markers = True,
                    height = 5, 
                    aspect = 0.75, 
                    col_wrap = 2,                                    # make facet-grid 2 plots wide
                    facet_kws = dict( sharex=False, sharey=False)     # arguments passed to sns.FacetGrid
        )
        axes = graph.axes.flatten()
        for ax in axes:
            ax.axhline(0, ls='-', linewidth=1, color='black')
        plt.show()
    """
    print( """\n
            --------------------------------------------------------------------------------------------------
            -- joe function: balancing of experimental groups with multi-objective simulated annealing v1.2 --
            --------------------------------------------------------------------------------------------------
        """ 
    )
    print( '\to this function could made exponentially faster by optimising how the objective function is calculated' )
    print( '\to the approach is currently using a (potentially) suboptimal sum of multiple objective functions approach, which will be improved in a future version' )
    print( '\to currently the approach uses integer period and user ID columns. Function may not work yet for other formats' )
    print( '\to there is currently no handling of the function\'s package dependencies: "pandas", "numpy", "math", "random" and "collections"' )
    print( 
            """\nto do in future updates:
                    - make calc_obj() function return the vector of objective values (a more general approach)
                    - implement one of the methods discussed by Sarafini paper (as additional second method)
                    - implement plotting functionality to visualise the end result 
                    - incorporate inclusion of categorical variables, and static variables (have the same value in every period)
                    - ability to do multiple runs with different random starting states, and also grid search over different starting parameters
                    - package multi-objective simulated annealing as it's own general function rather. This function will then use that generated simulated annealing function.
                    - code a genetic optimisation function, which can then be called as an alternative to simulated annealing.  
            """
        )

    import pandas as pd
    import numpy as np
    import math
    import random
    import collections
    import copy 

    # scale each value column so that it sums to 99999:
    for col_j in value_colnames:
        input_data[col_j] = 100 * input_data[col_j] / input_data[col_j].sum()  

    input_data = input_data.sort_values( [userID_colname,time_colname] )
    input_data.index = input_data[userID_colname] 

    treated_users = input_data.loc[ input_data[group_colname]=='treated' ]
    control_users = input_data.loc[ input_data[group_colname]=='control' ]
    n_treated_users = treated_users[userID_colname].drop_duplicates().shape[0]
    n_control_users = control_users[userID_colname].drop_duplicates().shape[0]

    # create dictionary of aggregated values (used by objective function)
    agg_values = {}
    for group_i in ['treated','control']:
        agg_values[group_i] = {}
        for value_j in value_colnames:
            agg_values[group_i][value_j] = input_data.loc[ input_data[group_colname]==group_i ].groupby(time_colname).agg(mean_value=(value_j,'mean'))['mean_value'].values

    print( '\nencoding users to dictionary...')
    # create dictionary of customers:
    treated_users_dict = {}
    for userID_i in treated_users[userID_colname].drop_duplicates().values:
        treated_users_dict[userID_i] = {}
        for value_j in value_colnames:            
            treated_users_dict[userID_i][value_j] = treated_users.loc[userID_i][value_j].values
    
    control_users_dict = {}
    for userID_i in control_users[userID_colname].drop_duplicates().values:
        control_users_dict[userID_i] = {}
        for value_j in value_colnames:            
            control_users_dict[userID_i][value_j] = control_users.loc[userID_i][value_j].values
    print( '...done\n' )

    def calc_obj( agg_values_dict, method ):
        if method=='avg_max_spike':
            # for each values series, maximum absolute value of (treated-control) is extracted (value from period with largest absolute difference)
            # these maximum values are then averaged to get a single scalar objective value 
            values_vec = {}
            for value_j in value_colnames:
                values_vec[ value_j ] = max( abs( agg_values_dict['treated'][value_j] - agg_values_dict['control'][value_j] ) )
            obj_value_to_return = np.mean( list(values_vec.values()) )

        if method=='avg_avg':
            # for each values series, maximum absolute value of (treated-control) is extracted (value from period with largest absolute difference)
            # these maximum values are then averaged to get a single scalar objective value 
            values_vec = {}
            for value_j in value_colnames:
                values_vec[ value_j ] = np.mean( abs( agg_values_dict['treated'][value_j] - agg_values_dict['control'][value_j] ) )
            obj_value_to_return = np.mean( list(values_vec.values()) )

        return obj_value_to_return

    # generate a new potential solution:
    def generate_potential_solution():
        changed_agg_values = copy.deepcopy(agg_values)
        remember_switches = []
        treated_users_to_switch = random.sample( list(treated_users_dict.keys()), n_switches_per_iteration )
        control_users_to_switch = random.sample( list(control_users_dict.keys()), n_switches_per_iteration )
        
        for switch_i in range(n_switches_per_iteration):
            random_treated_user = treated_users_to_switch[ switch_i ]  
            random_control_user = control_users_to_switch[ switch_i ]
            remember_switches.append( (random_treated_user, random_control_user) )
            random_treated_user_values = treated_users_dict[random_treated_user]
            random_control_user_values = control_users_dict[random_control_user]
            for value_j in value_colnames:
                # remove random treated user effect on treated group averages:
                changed_agg_values['treated'][value_j] = changed_agg_values['treated'][value_j] - random_treated_user_values[value_j]/n_treated_users 
                # add random control user effect on treated group averages:
                changed_agg_values['treated'][value_j] = changed_agg_values['treated'][value_j] + random_control_user_values[value_j]/n_treated_users 
                # remove random control user effect on control group averages:
                changed_agg_values['control'][value_j] = changed_agg_values['control'][value_j] - random_control_user_values[value_j]/n_control_users 
                # add random treated user effect on control group averages:
                changed_agg_values['control'][value_j] = changed_agg_values['control'][value_j] + random_treated_user_values[value_j]/n_control_users    

        return { 'changed_agg_values':changed_agg_values, 'remember_switches':remember_switches }  

    def implement_suggested_solution( switches_to_implement ):
        for switch_i in switches_to_implement:
            treated_user_values = treated_users_dict[ switch_i[0] ]
            control_user_values = control_users_dict[ switch_i[1] ]
            for value_j in value_colnames:
                # remove random treated user effect on treated group averages:
                agg_values['treated'][value_j] = agg_values['treated'][value_j] - treated_user_values[value_j]/n_treated_users 
                # add random control user effect on treated group averages:
                agg_values['treated'][value_j] = agg_values['treated'][value_j] + control_user_values[value_j]/n_treated_users 
                # remove random control user effect on control group averages:
                agg_values['control'][value_j] = agg_values['control'][value_j] - control_user_values[value_j]/n_control_users 
                # add random treated user effect on control group averages:
                agg_values['control'][value_j] = agg_values['control'][value_j] + treated_user_values[value_j]/n_control_users  
            # move treated user to control group:
            treated_users_dict.pop( switch_i[0] )
            control_users_dict[ switch_i[0] ] = treated_user_values 
            # move control user to treated group:
            control_users_dict.pop( switch_i[1] )
            treated_users_dict[ switch_i[1] ] = control_user_values

    #calc_obj( agg_values , method='avg_max_spike' )
    #test = generate_potential_solution()
    #calc_obj( test['changed_agg_values'], method='avg_max_spike' )
    #implement_suggested_solution( test['remember_switches'] )

    temp = init_temp

    current_best_obj_value = calc_obj( agg_values , method='avg_max_spike' )
    obj_value_at_start = current_best_obj_value 

    global_best_group_assignment = { 'treated':copy.deepcopy(treated_users_dict), 'control':copy.deepcopy(control_users_dict) } 
    global_best_obj_value = current_best_obj_value     
    
    n_iterations_since_last_saw_improvement_in_objective = 0

    # keep a count of the decisions made in the last 50 iterations:
    count_last100_solutions = collections.deque( maxlen=100 )

    for iteration_i in range(n_iterations):
        rand01 = random.random()
        # generate alternate group assignments:
        alt_assignment = generate_potential_solution()
        
        obj_value_of_alt_assignment = calc_obj( alt_assignment['changed_agg_values'], method='avg_max_spike' )

        print_results_this_iteration = 0
        if iteration_i%verbose_nsteps == 0:
            print_results_this_iteration = 1

        change_in_obj = obj_value_of_alt_assignment - current_best_obj_value 
        
        if print_results_this_iteration==1:
            print( f'\n-- iteration {iteration_i} --')
            print( f'current best objective value: {current_best_obj_value}' )
            print( f'number of iterations since last saw improvement in objective function: {n_iterations_since_last_saw_improvement_in_objective}' )
            print(f'objective value of proposed assignment: {obj_value_of_alt_assignment}')
            print(f'change in objective by switching to new proposed assignment: {change_in_obj}')
            print(f'global best objective value found so far: {global_best_obj_value}')
        if change_in_obj < 0:                    # i.e. objective value has improved (decreased) 
            # accept superior solution:
            count_last100_solutions.append( 'ACCEPT_better' )
            n_iterations_since_last_saw_improvement_in_objective = 0
            implement_suggested_solution( alt_assignment['remember_switches'] )        # update [agg_values], [treated_users_dict] and [control_users_dict]
            current_best_obj_value = calc_obj( agg_values, method='avg_max_spike' )
            
            if current_best_obj_value < global_best_obj_value:
                # update global best solution:
                global_best_group_assignment = { 'treated':copy.deepcopy(treated_users_dict), 'control':copy.deepcopy(control_users_dict) } 
                global_best_obj_value = current_best_obj_value

            if print_results_this_iteration==1:
                print('objective improved: accept proposed change')
                if current_best_obj_value < global_best_obj_value:
                    print( f'global best objective value updated to {current_best_obj_value}' )
            #store_obj_value_per_iteration.append( current_best_obj_value )
        elif change_in_obj >= 0 and rand01 < math.exp( -change_in_obj/temp ):
            n_iterations_since_last_saw_improvement_in_objective += 1
            count_last100_solutions.append( 'ACCEPT_worse' )
            implement_suggested_solution( alt_assignment['remember_switches'] )        # update [agg_values], [treated_users_dict] and [control_users_dict]
            current_best_obj_value = calc_obj( agg_values, method='avg_max_spike' )
            
            if print_results_this_iteration==1:
                print(f'ACCEPT suboptimal change:  runif01 {rand01}  <  change prob {math.exp(-change_in_obj/temp)}')

        else:
            n_iterations_since_last_saw_improvement_in_objective += 1
            count_last100_solutions.append( 'REJECT_worse' )
            if print_results_this_iteration==1:
                print(f'REJECT suboptimal change:  runif01 {rand01}  <  change prob {math.exp(-change_in_obj/temp)}')
        
        # reduce temperature:    
        if print_results_this_iteration==1:
            print( f'temperature = {temp}' )
            print( f'count accept BETTER solution in last 100 iterations:  {count_last100_solutions.count("ACCEPT_better")}' )
            print( f'count accept WORSE solution in last 100 iterations:   {count_last100_solutions.count("ACCEPT_worse")}' )
            print( f'count reject WORSE solution in last 100 iterations:   {count_last100_solutions.count("REJECT_worse")}' )
        temp = temp*cool_param

    print(
       f"""
        \n
        objective value at start: {obj_value_at_start}
        objective value at end:   {global_best_obj_value}
        \n
        """
    )

    #global_best_group_assignment.rename( columns={group_colname:f'optimised_{group_colname}'}, inplace=True )
    optimised_groups_to_return = pd.DataFrame(
        {
        userID_colname:( list(global_best_group_assignment['treated'].keys()) + list(global_best_group_assignment['control'].keys()) ),
        f'optimised_{group_colname}':['treated']*len(global_best_group_assignment['treated']) + ['control']*len(global_best_group_assignment['control'])
        }
    ) 

    return optimised_groups_to_return







def experimental_group_balancing_multiobjective_sim_annealing(
        input_data,                # input data 
        n_iterations,              # number of iterations to run algorithm
        n_switches_per_iteration,  # how many users will switch groups per iteration 
        init_temp,                 # starting temperature for the algorithm 
        cool_param,                # cooling parameter for the algorithm 
        verbose_nsteps             # results will be printed every [verbose_nsteps] iterations

        # simulated annealing works as follows:
        #   calculate objective value -O- of initial solution
        #   -O- is current "best solution"
        #   for iteration i in 1:n_iterations...
        #         ..generate a random value -R- in [0,1]
        #         ..perturb/modify the current solution
        #         ..if the proposed modified solution is better than current "best solution" -O-, then proposed solution becomes "best solution" -O-
        #             ..else if proposed modified solution worse than -O- and -R- < exp{ amount_objective_is_worse/temp }, then proposed solution becomes "best solution" -O-
        #                                                                    ( note that amount_objective_is_worse < 0 )
        #             ..else proposed modified solution is rejected
        #         .. temp = temp * cool_param  
        # throughout, I also keep track of the 'global' best objective value found so far
        #         ..even when a suboptimal modification is accepted, the 'global' best solution will not change unless it is the best solution found so far   
    ):
    """
        this function requires {pandas},{numpy},{math},{random} and {collections} to be imported 

        input data must be in the following format:
        time        userID       group       value1     value2   ....as many value columns as required 
         x            x            x           x           x
         x            x            x           x           x
         x            x            x           x           x
        (the actual column names can be anything, but column order and number does matter)
        the group column can only contain the values 'treated' and 'control'

        example usage:
        import pandas as pd
        pd.set_option('max_rows',100)
        import numpy as np
        import math
        import random
        import collections
        from matplotlib import pyplot as plt
        import seaborn as sns
        input_data = pd.DataFrame( 
                        {
                            'period':list(range(10))*50,
                            'customerID':sorted(list(range(50))*10),
                            'control_group':['treated']*400+['control']*100,
                            'spend':np.random.uniform(low=5, high=5000, size=500),
                            'recency':np.random.choice( range(1,9), size=500, replace=True ),
                        } 
                    )
        input_data.loc[ sorted(list(np.random.choice(input_data.index, size=100, replace=False))), 'spend' ] = 0          # make 50% of the spend values into 0            
        input_data['active'] = 1
        input_data.loc[ input_data['spend']==0, 'active' ] = 0
        # run the optimisation:
        optimised_groups = experimental_group_balancing_multiobjective_sim_annealing(
                    input_data = input_data,             # input data 
                    n_iterations = 500,                # number of iterations to run algorithm
                    n_switches_per_iteration = 5,      # how many users will switch groups per iteration 
                    init_temp = 5,                     # starting temperature for the algorithm 
                    cool_param = 0.98,                 # cooling parameter for the algorithm 
                    verbose_nsteps = 10                # results will be printed every [verbose_nsteps] iterations
        )
        input_data = input_data.merge( optimised_groups, on='customerID' )

        # compare the groups before and after optimisation:        
        store_aggregated_DFs = []
        for value_i in ['spend','recency','active']:
            temp_df = input_data.copy()
            temp_df['value_col'] = temp_df[value_i] 
            summarise_unoptimised_groups = (
                temp_df
                .groupby( ['period', 'control_group'] )
                .agg( mean_value = ('value_col', 'mean') )
                .reset_index()
            )

            summarise_unoptimised_groups = summarise_unoptimised_groups.pivot( index='period', columns='control_group', values='mean_value' ).reset_index()
            summarise_unoptimised_groups['diff'] = summarise_unoptimised_groups['treated'] - summarise_unoptimised_groups['control']
            summarise_unoptimised_groups['value'] = value_i
            summarise_unoptimised_groups['optimised_group'] = 'UNoptimised'

            store_aggregated_DFs.append( summarise_unoptimised_groups[['period','value','diff','optimised_group']] )

            summarise_optimised_groups = (
                temp_df
                .groupby( ['period', 'optimised_control_group'] )
                .agg( mean_value = ('value_col', 'mean') )
                .reset_index()
            )
            summarise_optimised_groups = summarise_optimised_groups.pivot( index='period', columns='optimised_control_group', values='mean_value' ).reset_index()
            summarise_optimised_groups['diff'] = summarise_optimised_groups['treated'] - summarise_optimised_groups['control']
            summarise_optimised_groups['value'] = value_i
            summarise_optimised_groups['optimised_group'] = 'optimised'

            store_aggregated_DFs.append( summarise_optimised_groups[['period','value','diff','optimised_group']] )
        
        aggregated_results = pd.concat( store_aggregated_DFs, axis=0 ) 

        graph = sns.relplot(
                    data = aggregated_results,
                    x = 'period', 
                    y = 'diff',
                    col = 'value',      # this is the variable faceted on 
                    kind = 'line',  
                    hue = 'optimised_group',
                    style = 'optimised_group',
                    markers = True,
                    height = 5, 
                    aspect = 0.75, 
                    col_wrap = 2,                                    # make facet-grid 2 plots wide
                    facet_kws = dict( sharex=False, sharey=False)     # arguments passed to sns.FacetGrid
        )
        axes = graph.axes.flatten()
        for ax in axes:
            ax.axhline(0, ls='-', linewidth=1, color='black')
        plt.show()
    """
    print( """\n
            --------------------------------------------------------------------------------------------------
            -- joe function: balancing of experimental groups with multi-objective simulated annealing v1.2 --
            --------------------------------------------------------------------------------------------------
        """ 
    )
    print( '\to this function could made exponentially faster by optimising how the objective function is calculated' )
    print( '\to the approach is currently using a (potentially) suboptimal sum of multiple objective functions approach, which will be improved in a future version' )
    print( '\to currently the approach uses integer period and user ID columns. Function may not work yet for other formats' )
    print( '\to there is currently no handling of the function\'s package dependencies: "pandas", "numpy", "math", "random" and "collections"' )
    print( 
            """\nto do in future updates:
                    - make calc_obj() function return the vector of objective values (a more general approach)
                    - implement one of the methods discussed by Sarafini paper (as additional second method)
                    - implement plotting functionality to visualise the end result 
                    - incorporate inclusion of categorical variables, and static variables (have the same value in every period)
                    - ability to do multiple runs with different random starting states, and also grid search over different starting parameters
                    - package multi-objective simulated annealing as it's own general function rather. This function will then use that generated simulated annealing function.
                    - code a genetic optimisation function, which can then be called as an alternative to simulated annealing.  
            """
        )

    import pandas as pd
    import numpy as np
    import math
    import random
    import collections

    input_df = input_data.copy()

    time_colname = input_df.columns[0]
    userID_colname = input_df.columns[1]        
    group_colname = input_df.columns[2]       
    value_colnames = list(input_df.columns[3:])

    print(
        f"""
            time/period column:         {time_colname}
            user ID column:             {userID_colname}
            treated/control column:     {group_colname}
            balancing values column(s): {', '.join(value_colnames) }
        """
    
    )
    temp = init_temp

    # scale each value column so that it sums to 99999:
    for col_j in value_colnames:
        input_df[col_j] = 100 * input_df[col_j] / input_df[col_j].sum()  

    def calc_obj( group_assignments ):
        # this function calculates the objective function value for a given group assignment
        treated_group = input_df.merge( group_assignments.query(f'{group_colname}=="treated"')[userID_colname], on=userID_colname, how='inner' ).copy()
        control_group = input_df.merge( group_assignments.query(f'{group_colname}=="control"')[userID_colname], on=userID_colname, how='inner' ).copy()

        agg_treated_group = treated_group.groupby([time_colname]).agg(np.mean).reset_index()
        agg_control_group = control_group.groupby([time_colname]).agg(np.mean).reset_index()

        current_obj_value = 0

        for col_j in value_colnames:
            temp_merge_df = agg_treated_group[[time_colname,col_j]].merge( agg_control_group[[time_colname,col_j]], on=time_colname, how='left' ).fillna(0).copy()
            current_obj_value += sum( abs( temp_merge_df[f'{col_j}_x'] - temp_merge_df[f'{col_j}_y'] ) )

        return current_obj_value

    current_best_group_assignment = input_df[[userID_colname,group_colname]].drop_duplicates()
    current_best_obj_value = calc_obj( current_best_group_assignment )
    obj_value_at_start = current_best_obj_value 

    global_best_group_assignment = current_best_group_assignment 
    global_best_obj_value = current_best_obj_value     
    
    n_iterations_since_last_saw_improvement_in_objective = 0

    # keep a count of the decisions made in the last 50 iterations:
    count_last100_solutions = collections.deque( maxlen=100 )

    for iteration_i in range(n_iterations):
        rand01 = random.random()
        # generate alternate group assignments:
        alt_assignment = current_best_group_assignment.copy()
        
        for switch_i in range(n_switches_per_iteration):
            # select 1 random user from the treated group and 1 random user from the control group:
            random_user1_group = random_user2_group = 'init'
            while random_user1_group == random_user2_group:
                random_user1 = random.choice( current_best_group_assignment[userID_colname].values )
                random_user1_group = current_best_group_assignment.query(f'{userID_colname}=={random_user1}')[group_colname].values[0]
                random_user2 = random.choice( current_best_group_assignment[userID_colname].values )
                random_user2_group = current_best_group_assignment.query(f'{userID_colname}=={random_user2}')[group_colname].values[0]
            # switch the assignments of the 2 users:
            alt_assignment.loc[ alt_assignment[userID_colname]==random_user1, group_colname ] = random_user2_group
            alt_assignment.loc[ alt_assignment[userID_colname]==random_user2, group_colname ] = random_user1_group

        obj_value_of_alt_assignment = calc_obj( alt_assignment )

        print_results_this_iteration = 0
        if iteration_i%verbose_nsteps == 0:
            print_results_this_iteration = 1

        change_in_obj = obj_value_of_alt_assignment - current_best_obj_value 
        if print_results_this_iteration==1:
            print( f'\n-- iteration {iteration_i} --')
            print( f'current best objective value: {current_best_obj_value}' )
            print( f'number of iterations since last saw improvement in objective function: {n_iterations_since_last_saw_improvement_in_objective}' )
            print(f'objective value of proposed assignment: {obj_value_of_alt_assignment}')
            print(f'change in objective by switching to new proposed assignment: {change_in_obj}')
            print(f'global best objective value found so far: {global_best_obj_value}')
        if change_in_obj < 0:                    # i.e. objective value has improved (decreased) 
            n_iterations_since_last_saw_improvement_in_objective = 0
            current_best_group_assignment = alt_assignment
            current_best_obj_value = obj_value_of_alt_assignment
            count_last100_solutions.append( 'ACCEPT_better' )

            if current_best_obj_value < global_best_obj_value:
                global_best_group_assignment = current_best_group_assignment.copy()
                global_best_obj_value = current_best_obj_value

            if print_results_this_iteration==1:
                print('objective improved: accept proposed change')
                if current_best_obj_value < global_best_obj_value:
                    print( f'global best objective value updated to {current_best_obj_value}' )
            #store_obj_value_per_iteration.append( current_best_obj_value )
        elif change_in_obj >= 0 and rand01 < math.exp( -change_in_obj/temp ):
            n_iterations_since_last_saw_improvement_in_objective += 1
            count_last100_solutions.append( 'ACCEPT_worse' )
            if print_results_this_iteration==1:
                print(f'ACCEPT suboptimal change:  runif01 {rand01}  <  change prob {math.exp(-change_in_obj/temp)}')
            current_best_group_assignment = alt_assignment
            current_best_obj_value = obj_value_of_alt_assignment
        else:
            n_iterations_since_last_saw_improvement_in_objective += 1
            count_last100_solutions.append( 'REJECT_worse' )
            if print_results_this_iteration==1:
                print(f'REJECT suboptimal change:  runif01 {rand01}  <  change prob {math.exp(-change_in_obj/temp)}')
        
        # reduce temperature:    
        if print_results_this_iteration==1:
            print( f'temperature = {temp}' )
            print( f'count accept BETTER solution in last 100 iterations:  {count_last100_solutions.count("ACCEPT_better")}' )
            print( f'count accept WORSE solution in last 100 iterations:   {count_last100_solutions.count("ACCEPT_worse")}' )
            print( f'count reject WORSE solution in last 100 iterations:   {count_last100_solutions.count("REJECT_worse")}' )
        temp = temp*cool_param

    print(
       f"""
        \n
        objective value at start: {obj_value_at_start}
        objective value at end:   {global_best_obj_value}
        \n
        """
    )

    global_best_group_assignment.rename( columns={group_colname:f'optimised_{group_colname}'}, inplace=True )
    
    return global_best_group_assignment




def fastDataExploration_basic_stats_per_column( pandas_df ):
    """
        calculates different metrics depending on feature type:
                                     numeric     categorical        example output
            %null                       x             x                   12% 
            %nonzero                    x                                 5%
            n_unique_values             x             x                   12
            maximum value               x                                 1648.84
            minimum value               x                                 -12.9
            mean value                  x                                 17.46
            deciles                     x                                 
            proportion_each_category                  x                   '0.8; 0.15; 0.04; 0.05; 0.04; ...'                note: string concatenated at nchar=50          
    """

def generate_itemFeatures_from_quantiles_of_numeric_user_feature(
        input_data,
        n_quantiles
    ):
    """
        this function cuts the provided numeric user feature into quantile bins..
        ..then creates features which, for each item, describe the proportion of rows falling into each quantile
        i.e. the function creates features which describe the distribution of the user population consuming each item  
        e.g. function could be used to quantify if an item is consumed by users in an "older" age quantile, a middling 'user spend' quantile etc.  
        (example follows) 

        this function requires pandas to be imported 
        
        input data must have format:
        item_ID     numeric_user_feature_name     
            x               x
            x               x
            x               x
        (the actual column names can be anything, but column order and number does matter)

    example usage:
        import pandas as pd
        import numpy as np
        pd.set_option( 'max_columns', 999 )
        input_data = pd.DataFrame( {'itemID':[1]*25+[2]*35+[3]*40, 'user_age':np.random.choice(a=range(100),size=100,replace=True) } )
        generate_itemFeatures_from_quantiles_of_numeric_user_feature( input_data=input_data, n_quantiles=5 )
    """
    item_ID_colname = input_data.columns[0]
    user_feature_colname = input_data.columns[1]
    input_data['user_feature_quantile'] = user_feature_colname + '_quantile_' + pd.qcut( x=input_data[user_feature_colname], q=n_quantiles ).astype(str)  # assign each user featur value to a quantile
    # add total users per item column:
    input_data.merge(
        input_data.groupby(item_ID_colname).agg( n_users_this_item=(user_feature_colname,'count') ).reset_index(),
        how = 'left',
        on = item_ID_colname
    )
    item_features = (
        input_data
        .groupby( [item_ID_colname, 'user_feature_quantile'] )
        .agg( n_users = (user_feature_colname, 'count') )
        .reset_index()
        .merge(
                input_data.groupby(item_ID_colname).agg( n_users_this_item=(user_feature_colname,'count') ).reset_index(),
                how = 'left',
                on = item_ID_colname
                )   
    )
    item_features['proportion_users_this_item'] = item_features['n_users'] / item_features['n_users_this_item']
    return item_features.sort_values('user_feature_quantile').pivot( index=item_ID_colname, values='proportion_users_this_item', columns='user_feature_quantile' )


    