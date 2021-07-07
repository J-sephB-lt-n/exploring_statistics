
cat("#############################\n### LOADING JOE FUNCTIONS ###\n#############################\n")

#################
### STAR BARS ###
#################

star_bars_ftn <-
  function( values_vector,
            bar_length = 9,
            set_min = NULL
  ){
    
    if( anyNA(values_vector) ){ warning("WARNING: NA values in vector removed\n") }
    
    if( !is.numeric(values_vector) ){ 
      warning("WARNING: values_vector is not numeric\n") 
      return( rep(NA, length(values_vector)) )
    }
    
    # if there are no negative values in 'values_vector':
    if( min(values_vector, na.rm=TRUE) >= 0 )
    {
      if( is.null(set_min) ){ set_min <- min(values_vector, na.rm=TRUE) }
      
      bars_to_print <-
        ( values_vector - set_min ) /
        ( max(values_vector, na.rm=TRUE) - set_min )
      
      bars_to_print <- bars_to_print * bar_length
      bars_to_print <- round(bars_to_print)
      
      return(
        sapply( bars_to_print,
                function(x){ 
                    if( is.na(x) ){ "" } else{
                                                paste( rep( "*",
                                                           x
                                                      ),
                                                      collapse = ""
                                                    )    
                                              }
                  
                          }
        )
      )
    } else     # if the vector contains negative values
    {
      
      max_value <- max( abs(values_vector), na.rm=TRUE )
      
      bars_to_print <- abs(values_vector) / max_value
      bars_to_print <- bars_to_print * bar_length
      bars_to_print <- round(bars_to_print)
      
      return(
        sapply( 1:length(values_vector),
                function(i){ 
                  
                  if( is.na(values_vector[i]) ){ 
                      ""
                    } else
                  if( values_vector[i] < 0 )  # if this number is negative
                {
                  paste0( 
                    paste( rep( "_", bar_length-bars_to_print[i] ),
                           collapse = ""
                    ),
                    paste( rep( "*", bars_to_print[i] ),
                           collapse = ""
                    ),
                    "0",
                    paste( rep( "_", bar_length), collapse="")
                  )
                } else      # if this number is non-negative
                  paste0( 
                    paste( rep( "_", bar_length), collapse=""),
                    "0",
                    paste( rep("*", bars_to_print[i]), collapse = "" ),
                    paste( rep( "_", bar_length-bars_to_print[i] ),
                           collapse = ""
                    )
                  ) 
                }
        )
      )
    }
  }

if( "star_bars_ftn" %in% ls() ){ cat( "[star_bars_ftn], by Joe, loaded \n")  }

# # example usage of star_bars_ftn():
# library(tidyverse)
# noddy_data <- tibble( x1 = sample(70:100, size=10),
#                       x2 = sample(-100:100, size=10)
# )
# noddy_data %>%
#   mutate( x1_bars_v1 = star_bars_ftn(x1, bar_length=6),
#           x1_bars_v2 = star_bars_ftn(x1, bar_length=10),
#           x1_bars_v3 = star_bars_ftn(x1, bar_length=10, set_min=0),
#           x2_bars_v1 = star_bars_ftn(x2, bar_length=6),
#           x2_bars_v2 = star_bars_ftn(x2, bar_length=15)
#         ) %>%
#   select( x1, x1_bars_v1, x1_bars_v2, x1_bars_v3,
#           x2, x2_bars_v1, x2_bars_v2
#         ) %>%
#   View





#############################
### UNIVARIATE VIEW PLOTS ###
#############################

univariate_view_plots_ftn <- 
  function( data_source,
            export_images_folder_filepath = NULL, # e.g. "C:/Users/Joe/Desktop/",
            facet_column_name = NULL,
            
            numeric_remove_top_x_percent = 0,  # remove top x% from numeric columns
                                               # e.g. 0.05 for remove top 5%     
            
            # numerical (histogram) plots:
            numerical_confint_alpha = 0.1,
            
            # categorical plots:
            categorical_confint_alpha = 0.1,
            
            # ggplot parameters:
            geom_histogram__bins = 50,
            geom_histogram_facet_grid__scales = "free_y",
            geom_bar_facet_grid__scales = "fixed",

            # ggsave parameters:
            ggsave__device = "jpeg",
               ggsave__dpi = 100,
             ggsave__width = 10,
            ggsave__height = 8
          ){
    
    # load tidyverse package if it is not loaded:
    if( ! "tidyverse" %in% (.packages()) ){ library(tidyverse) } 
    if( ! "MultinomialCI" %in% (.packages()) ){ library(tidyverse) } 
    
    # define a function which, when given a categorical feature (as a column), 
    # returns multinomial confidence intervals with specified level of statistical significance (alpha):
    categorical_col_to_CI <- 
      function( categorical_col, chosen_alpha ){
      
        raw_values <- table( unlist(categorical_col) )
        store_category_names <- names(raw_values)
      
        get_confint <- MultinomialCI::multinomialCI(     x = raw_values,
                                                     alpha = chosen_alpha
                                                   ) 
        colnames(get_confint) <- c("lwr","upr")
        get_confint <- as_tibble( get_confint )
        
        return( 
                  bind_cols( tibble(category = store_category_names), get_confint )
        )
    }
    
    # force the data.frame into tibble() format: 
    data_source <- as_tibble(data_source)
    
    # create a list to store the plots created: 
    store_plots_list <- list()
    
    print( "creating plots" )
    
    # if there is no facet_column, then do this:
    if( is.null(facet_column_name) ){
      for( j in 1:ncol(data_source) ){     # for every column in the data

          col_to_plot <- data_source[,j]
          store_vbl_name <- names(col_to_plot)
          names(col_to_plot) <- "placeholder_colname"
        
          if( is.numeric(col_to_plot[[1]]) ){       # if column to plot is numeric, do this:
            
            # find and remove the top X% of the data (as specified by "numeric_remove_top_x_percent"):
            top_x_percent_quantile <- quantile(  col_to_plot$placeholder_colname, 
                                                 prob = 1-numeric_remove_top_x_percent,
                                                na.rm = TRUE
                                              )
            
            col_to_plot <- col_to_plot %>% filter( placeholder_colname <= top_x_percent_quantile )
            
            # calculate some sample statistics: 
            col_stats <- 
              col_to_plot %>% 
                summarise(                   n = n(),
                                             col_mean = mean(placeholder_colname, na.rm=TRUE),
                                             col_std_dev = sd(placeholder_colname, na.rm=TRUE),
                                             col_median = median(placeholder_colname, na.rm=TRUE)
                ) %>% 
                ungroup() %>% 
                # calculate the normal asymptotic standard error (sd) of the mean: 
                mutate( std_error_of_col_mean = col_std_dev / sqrt(n) ) %>% 
                # put the metrics into a clean text format to put onto the plot:
                mutate( aggs_combined = paste0( "mean: ",
                                                round( col_mean, digits=2 ),
                                                "\n",
                                                "Std error of mean: ",
                                                round( std_error_of_col_mean, digits=2 ),
                                                "\n",
                                                "Std Dev.: ",
                                                round( col_std_dev, digits=2 ),
                                                "\n",
                                                "Median: ",
                                                round( col_median, digits=2 )
                                              ) 
                      ) %>% 
                # calculate the normal asymptotic confidence intervals for the mean:
                mutate( CI_mean_lwr = col_mean - qnorm(1-numerical_confint_alpha/2, mean=0, sd=1) * std_error_of_col_mean ) %>% 
                mutate( CI_mean_upr = col_mean + qnorm(1-numerical_confint_alpha/2, mean=0, sd=1) * std_error_of_col_mean )
            
            # create the plot for this variable:
            store_plots_list[[ names(data_source)[j] ]] <-
                ggplot( data = col_to_plot,
                        aes( x = placeholder_colname )
                      ) +
                  geom_histogram( bins = geom_histogram__bins ) +
 
                  labs(        x = store_vbl_name,
                               title = paste0( "Dashed lines are ",
                                               round( (1-numerical_confint_alpha)*100, digits=1),
                                               "% asymptotic normal confidence intervals for the mean"      
                                             ),
                               subtitle = paste0( "note: removed top ",
                                                  round( numeric_remove_top_x_percent*100, digits=2 ),
                                                  "% of observations"
                                                 )  
                      ) +
              
                      # add the statistics to the plots:
                      geom_text( data = col_stats, 
                                 aes( label = aggs_combined,
                                      x = Inf, y = Inf,   # force text to top left of plot, regardless of units     
                                 ),
                                 hjust = 1, vjust = 1,
                                 colour = "red"
                      ) +
                      # add line at the mean:
                      geom_vline( data = col_stats,
                                  aes( xintercept = col_mean ),
                                  colour = "green",
                                  size = 1.5
                      ) + 
                      # add 100[1 - alpha]% confidence intervals (standard errors) for the mean:
                      geom_vline( data = col_stats,
                                  aes( xintercept = CI_mean_lwr ),
                                  colour = "green",
                                  size = 1,
                                  linetype = "dashed"
                      ) +
                      geom_vline( data = col_stats,
                                  aes( xintercept = CI_mean_upr ),
                                  colour = "green",
                                  size = 1,
                                  linetype = "dashed"
                      )
              
              
          } else {                # if column to plot is not numeric, then don't plot a histogram but rather a barplot:
            
             if( length(unique(col_to_plot[[1]])) > 20 ){  
               # if there are more than 20 unique values for this variable:
               # return error message
               store_plots_list[[ names(data_source)[j] ]] <-
                 "[error] did not plot - too many unique values in variable"
               
             } else {   # otherwise make the plot
             
             col_to_plot <- col_to_plot %>% mutate( placeholder_colname = as.character(placeholder_colname) )   
               
             get_categ_CI <- categorical_col_to_CI( categorical_col = col_to_plot,
                                                    chosen_alpha = categorical_confint_alpha
                                                  )
               
             store_plots_list[[ names(data_source)[j] ]] <-
                  col_to_plot %>% 
                      group_by( placeholder_colname ) %>% 
                      tally() %>%
                      mutate( proportion = n / sum(n) ) %>% 
                      left_join( get_categ_CI,
                                 by = c("placeholder_colname"="category") 
                               ) %>% 
                      ggplot( data = .,
                              aes(    x = placeholder_colname,
                                      y = proportion*100,
                                   fill = placeholder_colname,
                                   ymin = lwr*100,
                                   ymax = upr*100 
                                 )
                      ) +
                      geom_bar( stat="identity" ) +
                      labs( x = store_vbl_name,
                            y = "%",
                            title = paste0( "Error bars are ",
                                            round( 100*(1-categorical_confint_alpha), digits=1 ),
                                            "% simultaneous multinomial (Sison-Graz) confidence intervals"
                            )
                          ) +
                      theme( legend.position = "none" ) +
                      geom_errorbar() 
               
             }
          }
          
          print( names(data_source)[j] )          # print name of column just processed
      }
    }
    
    # if there is facet_column, then do this:
    if( !is.null(facet_column_name) ){
      
      data_to_plot <- data_source %>% select(-facet_column_name)   # remove the facet_column:
      
      for( j in 1:ncol(data_to_plot) ){     # for every (non-facet) column in the data
        
        col_to_plot <- data_to_plot[,j]     # store this column
        
        # add back the faceting column:
        col_to_plot <- bind_cols( data_source %>% select(facet_column_name)
                                  ,
                                  col_to_plot
                                )
        names(col_to_plot) <- c( "facet_variable", "placeholder_colname" )
        
        if( is.numeric(col_to_plot[[2]]) ){     # if the variable to plot is numeric, then do this
          
          # find top x% in each group:
          top_x_pct_by_group <- 
            col_to_plot %>% 
                group_by( facet_variable ) %>% 
                summarise( top_x_pct_quantile = quantile(placeholder_colname, probs=1-numeric_remove_top_x_percent)  ) %>% 
                ungroup()
          
          # remove top x% from each facet:
          col_to_plot <- 
            col_to_plot %>% 
                left_join( top_x_pct_by_group, by = "facet_variable" ) %>% 
                filter( placeholder_colname <= top_x_pct_quantile )
          
          # calculate some sample statistics: 
          col_stats <- 
            col_to_plot %>% 
              group_by( facet_variable ) %>% 
              summarise(                   n = n(),
                                    col_mean = mean(placeholder_colname, na.rm=TRUE),
                                 col_std_dev = sd(placeholder_colname, na.rm=TRUE),
                                  col_median = median(placeholder_colname, na.rm=TRUE)
                     ) %>% 
              ungroup() %>% 
              mutate( std_error_of_col_mean = col_std_dev / sqrt(n) ) %>% 
              ungroup() %>% 
              mutate( aggs_combined = paste0( "mean: ",
                                              round( col_mean, digits=2 ),
                                              "\n",
                                              "Std error of mean: ",
                                              round( std_error_of_col_mean, digits=2 ),
                                              "\n",
                                              "Std Dev.: ",
                                              round( col_std_dev, digits=2 ),
                                              "\n",
                                              "Median: ",
                                              round( col_median, digits=2 )
                                            ) 
                     ) %>% 
              mutate( CI_mean_lwr = col_mean - qnorm(1-numerical_confint_alpha/2, mean=0, sd=1) * std_error_of_col_mean ) %>% 
              mutate( CI_mean_upr = col_mean + qnorm(1-numerical_confint_alpha/2, mean=0, sd=1) * std_error_of_col_mean )
          
          store_plots_list[[ names(data_to_plot)[j] ]] <-
            
            ggplot( data = col_to_plot,
                    aes( x = placeholder_colname )
            ) +
            geom_histogram( bins = geom_histogram__bins ) +
            labs(        x = names(data_to_plot)[j],
                     title = paste0( "Dashed lines are ",
                                     round( (1-numerical_confint_alpha)*100, digits=1),
                                     "% asymptotic normal confidence intervals for the mean"      
                                   ),
                  subtitle = paste0( "note: removed top ",
                                     round( numeric_remove_top_x_percent*100, digits=2 ),
                                     "% of observations from each facet"
                                   )  
                ) +
            facet_grid( facet_variable ~ .,
                        scales = geom_histogram_facet_grid__scales 
                      ) +
            
            # add the statistics to the plots:
            geom_text( data = col_stats, 
                       aes( label = aggs_combined,
                            x = Inf, y = Inf,   # force text to top left of plot, regardless of units     
                          ),
                       hjust = 1, vjust = 1,
                       colour = "red"
            ) +
            # add line at the mean:
            geom_vline( data = col_stats,
                         aes( xintercept = col_mean ),
                         colour = "green",
                         size = 1.5
                      ) + 
            # add 100[1 - alpha]% confidence intervals (standard errors) for the mean:
            geom_vline( data = col_stats,
                        aes( xintercept = CI_mean_lwr ),
                        colour = "green",
                        size = 1,
                        linetype = "dashed"
            ) +
            geom_vline( data = col_stats,
                        aes( xintercept = CI_mean_upr ),
                        colour = "green",
                        size = 1,
                        linetype = "dashed"
            )
            
          
        } else {  # if the variable to plot is NOT numeric, then do this  
          
          if( length(unique(col_to_plot[[2]])) > 20 ){  
            # if there are more than 20 unique values for this variable:
            # return error message
            store_plots_list[[ names(data_to_plot)[j] ]] <-
              "[error] did not plot - too many unique values in variable"
          
            } else {   # otherwise make the plot
            
            col_to_plot <- col_to_plot %>% mutate( placeholder_colname = as.character(placeholder_colname) ) 
              
            # generate the confidence intervals:
            get_CI <- 
              split( x = col_to_plot,
                     f = col_to_plot$facet_variable
                   )
            
            get_CI_stage2 <- 
              lapply(   X = get_CI,
                      FUN = function(x){ categorical_col_to_CI( categorical_col = x[,2],
                                                                chosen_alpha = categorical_confint_alpha
                                                              ) 
                                       }
                    )
            
            final_categ_CI <- tibble(facet_variable = "", category = "", lwr=0.69, upr=0.420) %>% slice(0)
            
            for( i in 1:length(get_CI_stage2) ){
              
                final_categ_CI <- 
                    bind_rows( final_categ_CI
                               ,
                               get_CI_stage2[[i]] %>% mutate( facet_variable = names(get_CI_stage2)[i] )
                             )
            }
            
            final_categ_CI <- 
                final_categ_CI %>% 
                  rename( placeholder_colname = category
                        )
            
            store_plots_list[[ names(data_to_plot)[j] ]] <-
              
              ggplot( data = col_to_plot %>% 
                                group_by( facet_variable, placeholder_colname ) %>% 
                                tally() %>% 
                                group_by( facet_variable ) %>% 
                                mutate( facet_group_total = sum(n) ) %>% 
                                ungroup() %>% 
                                mutate( proportion = n / facet_group_total ) %>% 
                                
                                # add on the Confidence Intervals:
                                left_join( final_categ_CI,
                                           by = c("facet_variable", "placeholder_colname")
                                         )
                        
                        
                        ,
                      aes(       x = placeholder_colname,
                                 y = proportion, 
                              fill = placeholder_colname,
                              ymin = lwr,
                              ymax = upr
                         )
                    ) +
              geom_bar( stat = "identity" ) + 
              geom_text( aes( label = paste0( n, " (", round(proportion*100, digits=2), "%)"),
                              angle = 90,
                                  y = 0        # start writing at bottom of bar
                            ),
                         hjust = "left"
                       ) +
              geom_errorbar( colour = "red" ) +
              labs(     x = names(data_to_plot)[j],
                    title = paste0( "Error bars are ",
                                    round( 100*(1-categorical_confint_alpha), digits=1 ),
                                    "% simultaneous multinomial (Sison-Graz) confidence intervals"
                                  )
                    
                  ) +
              theme( legend.position = "none",
                     axis.text.x = element_text( angle=90 )
                     ) +
              facet_wrap( . ~ facet_variable,
                          scales = geom_bar_facet_grid__scales  
                        ) 
              
            }
        }
              # print column name that has just been plotted
              print( names(data_to_plot)[j] )
        }
    }
    
    if( is.null(export_images_folder_filepath) ){     # if no filepath has been provided, do this:
      
      return( store_plots_list ) 
      
    } else {    # if a filepath has been provided, do this:
        
        # if the specified folder doesn't exist, then create it:
        if( !dir.exists(export_images_folder_filepath) ){ 
                dir.create(export_images_folder_filepath)
                print( paste0("created directory  ", export_images_folder_filepath) )
          }
      
        for( k in 1:length(store_plots_list) ){      # iterate through all plots created
         
          if( is.ggplot(store_plots_list[[k]]) ){    # if this plot has correctly made a ggplot
            
            full_filename <- paste0( "univ_plot_",
                                     names(store_plots_list)[k],
                                     case_when( ggsave__device=="jpeg" ~ ".jpg",
                                                 ggsave__device=="png" ~ ".png",
                                                 ggsave__device=="bmp" ~ ".bmp",
                                                 ggsave__device=="pdf" ~ ".pdf",
                                                 TRUE ~ ""
                                                # note: need to implement other ggsave formats!   
                                               )
                                    )
              
            ggsave(     plot = store_plots_list[[k]],
                    filename = paste0( export_images_folder_filepath, full_filename ),
                      device = ggsave__device,
                         dpi = ggsave__dpi,
                       width = ggsave__width,
                      height = ggsave__height
                  ) 
            
            print( paste0("exported plot  ", full_filename) )
            
          }
          
        }
        
    } 
    
}
 
if( "univariate_view_plots_ftn" %in% ls() ){ cat( "[univariate_view_plots_ftn] loaded \n")  }

# # example usage of univariate_view_plots_ftn():
# library(tidyverse)
# noddy_data <- tibble(           x0 = sample( paste("segment",1:3,sep="_"),
#                                              size = 100,
#                                              replace = TRUE
#                                            ),
#                         x1_numeric = sample(-30:100, size=100),
#                       x2_character = sample( c("ghana", "zambia", "nigeria"),
#                                              size = 100,
#                                              replace = TRUE
#                                             ) %>%
#                                       as.character(),
#                          x3_factor = sample( c("green", "yellow", "red", "blue"),
#                                           size=100,
#                                           replace=TRUE
#                                         ) %>%
#                                      as_factor(),
#                         x4_logical = sample( c(TRUE,FALSE),
#                                            size = 100,
#                                            replace = TRUE
#                                          )
# 
#                     )
# 
# noddy_data %>% head()
# 
# test <- univariate_view_plots_ftn(          data_source = noddy_data[,-1],
#                                    geom_histogram__bins = 10
#                                  )
# names(test)
# test$x1_numeric
# test$x2_character
# test$x3_factor
# test$x4_logical
# 
# test2 <- univariate_view_plots_ftn(          data_source = noddy_data,
#                                        facet_column_name = "x0",
#                                     geom_histogram__bins = 10
#                                   )
# names(test)
# test2$x1_numeric
# test2$x2_character
# test2$x3_factor
# 
# # create the plots and export them:
# univariate_view_plots_ftn(          data_source = noddy_data,
#                                     facet_column_name = "x0",
#                                     geom_histogram__bins = 10,
#                                     export_images_folder_filepath = "C:/Users/Joe/Desktop/"
#                           )





##################################################
### AVERAGE RESPONSE DECOMPOSITION SCATTERPLOT ###
##################################################
# avg_response_decomp_scatterplot <- 
#   function( data_source,
#             response_vbl_name,
#             remove_NA = FALSE,
#             n_bins = 6,                # number of bins to use when categorizing numeric variables 
#             bin_method = "uniform"    # alternative is "quantile"
#   ){
#     
#     # work out all pair combinations of the axis (x,y) variables 
#     # turn all axis (x,y) variables 
#    
#     # load tidyverse package if it is not loaded:
#     if( ! "tidyverse" %in% (.packages()) ){ library(tidyverse) } 
#     
#     # if ti isn't already, turn data_source into tibble() format: 
#     data_source <- as_tibble(data_source)
#       
#     store_plots_list <- list()
#     
#     axis_col_combinations <- 
#       combn( x = setdiff( colnames(data_source), response_vbl_name ),
#              m = 2, 
#              simplify = FALSE 
#            )
#     
#     # [do a step here where numeric variables are binned/categorized]
#     
#     for( i in 1:length(axis_col_combinations) ){
#       
#       df_plot_info <- data_source %>% 
#                         select(axis_col_combinations[[i]], response_vbl_name )
#       
#       colnames( df_plot_info ) <- c( "x", "y", "spend" )
#       
#       df_plot_info <-
#         df_plot_info %>%
#         group_by( x, y ) %>%
#         summarise( n_customers = n(),
#                    total_spend = sum(spend)
#         ) %>%
#         ungroup() %>%
#         
#         mutate( n_customers_scaled = scale(n_customers),
#                 total_spend_scaled = scale(total_spend)
#         )
#       
#       plot_title <- ""
#       plot_subtitle <- ""
#       
#       if( remove_NA == TRUE ){
#         df_plot_info <-
#           df_plot_info %>%
#           filter( !is.na(x) & !is.na(y) )
#         
#         plot_subtitle <- paste0( plot_subtitle, "Category NA values omitted (axis vbls)" )
#       }
#       
#       # create the plot: 
#       store_plots_list[[ paste(axis_col_combinations[[i]], collapse="_") ]] <- 
#         ggplot( data = df_plot_info,
#                 aes( y = y,
#                      x = x
#                 )
#               ) +
#           geom_point( aes( size = scale(total_spend/n_customers) ),
#                       colour = "black",
#                       shape = 1
#         #             ) +
#         #   geom_point( aes( size = total_fsp_scaled),
#         #                    alpha = 0.5,
#         #                    colour = "red",
#         #               position = position_nudge(y=0.1, x=0.1)
#         #             ) +
#         # scale_size_continuous( range=c(0, tweak_point_size) ) +
#         # geom_point( aes( size = n_customers_scaled ),
#         #             alpha = 0.5,
#         #             colour = "blue",
#         #             position = position_nudge(y=-0.1, x=-0.1)
#         # ) +
#         # geom_text( aes(label = paste0( "R", round(total_fsp/1e6L), "m" ) ),
#         #            size=3,
#         #            colour="red",
#         #            position = position_nudge(y=0.2,x=0.1)#,
#         #            #angle = -30
#         # ) +
#         # geom_text( aes(label = paste0( round(n_customers/1000), "k") ),
#         #            size = 3,
#         #            colour = "blue",
#         #            position = position_nudge(y=-0.3,x=-0.1) #,
#         #            #angle = -30
#         # ) +
#         # geom_text( aes( label = paste0( "R", round(total_fsp/n_customers) ) ),
#         #            #size=5,
#         #            colour="black"
#         # ) +
#         # labs( subtitle = plot_subtitle
#         # ) +
#         # theme( legend.position = "none",
#         #        axis.text.x = element_text(angle=90)
#         # ) +
#         # xlab( colname1 ) +
#         # ylab( colname2 )
#       
#       
#     }
#     
#     
#   }
# 
# # # example usage of avg_response_decomp_scatterplot():
# library(tidyverse)
# noddy_data <- tibble( y = sample(-30:100, size=100),
#                       x1 = sample( c("a","b","c"), size=100, replace=TRUE ),
#                       x2 = sample( c("W","X","Y","Z"), size=100, replace=TRUE ),
#                       x3 = sample( c("zero", "one"), size=100, replace=TRUE ) #,
#                       #x4 = rnorm( 100 )
#                     )
# 
# 
# 
# # Bivariate View
# make_bivariate_value_bubble_plot <- function( colname1, colname2, remove_NA=FALSE, tweak_point_size=20 ){
# 
#   colname1 <- "specify_colname 1"
#   colname2 <- "specify_colname 2"
#   df_plot_info <- expand.grid( colname1 = c("a","b","c","d"),
#                                colname2 = c("w","x","y","z") 
#                               ) %>% 
#                     as_tibble() %>% 
#                     mutate( sum_fsp = rexp(16, rate=0.005) )
# 
#   colnames( df_plot_info ) <- c( "x", "y", "fsp" )
# 
#   plot_title <- ""
#   plot_subtitle <- "At 'Final Sales Price'. Period 2017/07/01 - 2018/06/30. "
# 
#   df_plot_info <-
#     df_plot_info %>%
#     group_by( x, y ) %>%
#     summarise( n_customers = n(),
#                total_fsp = sum(fsp)
#     ) %>%
#     ungroup() %>%
# 
#     mutate( n_customers_scaled = scale(n_customers),
#             total_fsp_scaled = scale(total_fsp)
#     )
# 
#   if( remove_NA == TRUE ){
#     df_plot_info <-
#       df_plot_info %>%
#       filter( !is.na(x) & !is.na(y) )
# 
#     plot_subtitle <- paste0( plot_subtitle, "Category NA values omitted." )
#   }
# 
#      ggplot( data = df_plot_info,
#              aes( y = y,
#                  x = x
#             )
#     ) +
#       geom_point( aes( size = scale(total_fsp/n_customers) ),
#                   colour = "black",
#                   shape = 1
#       ) +
#       geom_point( aes( size = total_fsp_scaled),
#                   alpha = 0.5,
#                   colour = "red",
#                   position = position_nudge(y=0.1, x=0.1)
#       ) +
#       scale_size_continuous( range=c(0, tweak_point_size) ) +
#       geom_point( aes( size = n_customers_scaled ),
#                   alpha = 0.5,
#                   colour = "blue",
#                   position = position_nudge(y=-0.1, x=-0.1)
#       ) +
#       geom_text( aes(label = paste0( "R", round(total_fsp/1e6L), "m" ) ),
#                  size=3,
#                  colour="red",
#                  position = position_nudge(y=0.2,x=0.1)#,
#                  #angle = -30
#       ) +
#       geom_text( aes(label = paste0( round(n_customers/1000), "k") ),
#                  size = 3,
#                  colour = "blue",
#                  position = position_nudge(y=-0.3,x=-0.1) #,
#                  #angle = -30
#       ) +
#       geom_text( aes( label = paste0( "R", round(total_fsp/n_customers) ) ),
#                  #size=5,
#                  colour="black"
#       ) +
#       labs( subtitle = plot_subtitle
#       ) +
#       theme( legend.position = "none",
#              axis.text.x = element_text(angle=90)
#       ) +
#       xlab( colname1 ) +
#       ylab( colname2 )
#   
# 
# }
# 
# if( "avg_response_decomp_scatterplot" %in% ls() ){ cat( "[avg_response_decomp_scatterplot] loaded \n")  }
# 







data_rank_transform <- 
  function( data_source,
            n_quantiles_for_continuous_vbls = 10,  # number of bins to break continuous vbls into
            weight_for_1hot_vbls = NULL,              # weight value for 1-hot encoded categorical variables 
                                                      # default is round(n_quantiles_for_continuous_vbls/3)
            zero_inflated_percent_definition = 0.05,  # if proportion of 0 in data is more than this, then
                                                      # numerical variable is considered zero-inflated
                                                      # this causes an extra category (0) to be included
            colnames_to_ignore = NULL    # columns to return a is, without processing
           ){
  
    # what this function does: 
    #   (1) deal with NAs in the data
    #   (2) turn numeric variables into ranked factor 1,2,3,4,5,...,[n_quantiles_for_continuous_vbls]
    #       one-hot-encode factor variables and logicals into {0,1}

    # load required packages:
    if( ! "tidyverse" %in% (.packages()) ){ library(tidyverse) } 
    if( ! "mltools" %in% (.packages()) ){ library(mltools) } 
    if( ! "data.table" %in% (.packages()) ){ library(data.table) } 
    
    if( is.null(weight_for_1hot_vbls) ){ weight_for_1hot_vbls <- round(n_quantiles_for_continuous_vbls/3) }
    
    cat( paste0( "using weight ",
                 weight_for_1hot_vbls,
                 " for categorical variables \n",
                 "numeric variables split in ",
                 n_quantiles_for_continuous_vbls,
                 " quantiles (bins) \n",
                 "numeric variables containing more than ",
                 round(zero_inflated_percent_definition*100), "%",
                 " zeroes are considered 'zero-inflated' \n"
               )
    )
    
    data_source <- as_tibble(data_source)
    
    # (1) deal with NAs in the data:
    if( anyNA(data_source) ){ return( "warning: NAs present in data (to be dealt with in later version)\n" ) }
    
    #   (2) turn numeric variables into ranked factor 1,2,3,4,5,...,[n_quantiles_for_continuous_vbls]
    #       one-hot-encode factor variables and logicals into {0,1}
    clean_data_to_return <- data_source
      
    for( col_j in colnames(data_source) ){   
         # work through each feature (column) one at a time
         extract_col <- data_source[,col_j] 
         
         cat( paste0( "processing column ", colnames(extract_col)[1], "\n" ) )
         
         if( colnames(extract_col)[1] %in% colnames_to_ignore ){ 
           cat( paste0( "[", colnames(extract_col)[1], "] returned as is \n" ) )
            }
         
         else if( is.numeric(extract_col[[1]]) ){   # if column is numeric
           
           raw_numeric_vector <- extract_col[[1]]     # extract the values from the column
           
           if( length( unique(raw_numeric_vector) ) > 2 &   # variable has more than 2 unique values
               # ..and there are a large proportion of 0 values:
               ( sum(raw_numeric_vector==0) / length(raw_numeric_vector) ) > zero_inflated_percent_definition 
             ){
             raw_numeric_vector[ raw_numeric_vector==0 ] <- NA
             warning( paste0( "numeric variable [", 
                              colnames(extract_col)[[1]],
                              "] is zero-inflated \n",
                              " included additional category '0'\n"
                            ) 
                    )
           }
           
           # for calculating quantiles:
           # if variable has 2 or fewer unique values, return it as is
           #    else try break variable into specified number of quantiles (bins)
           #    else if breaks are not unique:
           #        try half the number of bins
           #        else try using 2 bins
           #        else return error message
           
           if( length( unique(raw_numeric_vector) ) <= 2 ){   # if only 2 unique values in col_j:
             warning( paste0( "column [",
                              colnames(extract_col)[[1]],
                              "] has 2 or fewer unique values - returning variable as is\n" 
                             )
                    )
             quantile_vector <- raw_numeric_vector 
           }
           
           else if( 
                sum(              # if breaks are unique
                  duplicated( 
                    quantile(     x = raw_numeric_vector,
                              probs = seq(0,1,length.out=(n_quantiles_for_continuous_vbls+1)),
                                 na.rm = TRUE     # don't consider the 0 values
                            )
                    )
                  ) == 0
           ){
             
             quantile_vector <- cut(      x = raw_numeric_vector,
                                     breaks = quantile(     x = raw_numeric_vector,
                                                        probs = seq(0,1,length.out=(n_quantiles_for_continuous_vbls+1)),
                                                        na.rm = TRUE     # don't consider the 0 values
                                                      ),
                                     labels = FALSE,          # return integer codes rather than (a,b] format labels
                             include.lowest = TRUE
                                   )
             
             # print to screen the quantiles used:
             quantile(     x = raw_numeric_vector,
                           probs = seq(0,1,length.out=(n_quantiles_for_continuous_vbls+1)),
                           na.rm = TRUE     # don't consider the 0 values
             ) %>% 
               print()
             
           } else if( 
                      sum(              # if breaks are unique using half the number of bins
                        duplicated( 
                          quantile(     x = raw_numeric_vector,
                                    probs = seq(0,
                                                1,
                                                length.out = round( (n_quantiles_for_continuous_vbls+1)/2 )
                                               ),
                                     na.rm = TRUE     # don't consider the 0 values
                                  )
                                 )
                         ) == 0
           ){
             warning( paste0( "column [",
                              colnames(extract_col)[[1]],
                              "]: breaks are not unique - using half the number of bins\n" 
                             )
             )
             quantile_vector <- cut(      x = raw_numeric_vector,
                                          breaks = quantile(     x = raw_numeric_vector,
                                                                 probs = seq(0,
                                                                             1,
                                                                             length.out = round( (n_quantiles_for_continuous_vbls+1)/2 )
                                                                 ),
                                                                 na.rm = TRUE     # don't consider the 0 values
                                          ),
                                          labels = FALSE,          # return integer codes rather than (a,b] format labels
                                          include.lowest = TRUE
                                   )
             # print to screen the quantiles used:
             quantile(     x = raw_numeric_vector,
                           probs = seq(0,
                                       1,
                                       length.out = round( (n_quantiles_for_continuous_vbls+1)/2 )
                           ),
                           na.rm = TRUE     # don't consider the 0 values
             ) %>% 
               print()
             
             
           } else if( 
             sum(              # if breaks are unique using only 2 bins
               duplicated( 
                 quantile(     x = raw_numeric_vector,
                               probs = c(0,0.5,1),
                               na.rm = TRUE     # don't consider the 0 values
                         )
               )
             ) == 0
           ){
             paste0( "column [",
                     colnames(extract_col)[[1]],
                     "]: breaks are not unique - using 2 bins\n" 
             )
             quantile_vector <- cut(      x = raw_numeric_vector,
                                          breaks = quantile(     x = raw_numeric_vector,
                                                                 probs = c(0,0.5,1),
                                                                 na.rm = TRUE     # don't consider the 0 values
                                          ),
                                          labels = FALSE,          # return integer codes rather than (a,b] format labels
                                          include.lowest = TRUE
                                    )
             
             # print to screen the quantiles used:
             quantile(     x = raw_numeric_vector,
                           probs = c(0,0.5,1),
                           na.rm = TRUE     # don't consider the 0 values
             ) %>% 
               print()
             
           } else{ 
             paste0( "column [",
                     colnames(extract_col)[[1]],
                     "]: number of breaks is not unique \n" 
             )
             quantile_vector <- rep(0, length(quantile_vector) ) 
           }

           quantile_vector[ is.na(quantile_vector) ] <- 0   
           #reverse_quantile_vector <- abs( quantile_vector - (max(quantile_vector)+1) )
           
           clean_data_to_return[,col_j] <- quantile_vector    # replace the previous column with the new quantile column
                                     
         } else if( class(extract_col[[1]]) %in% c("character","factor","logical") ) {
         
           # convert the column to factor class:
           col_j_factor <- factor(extract_col[[1]])
           col_j_factor_tbl <- tibble( placeholder_colname = col_j_factor )
           colnames(col_j_factor_tbl) <- names(extract_col)[1] 
           
           col_j_1hot_encode <- 
              mltools::one_hot(       dt = as.data.table(col_j_factor_tbl),
                                dropCols = TRUE
                              ) %>% 
              as_tibble() %>% 
              mutate_all( function(x){ x * weight_for_1hot_vbls } )
           
           clean_data_to_return <- 
             bind_cols( clean_data_to_return %>% select(-col_j ),   # throw away original raw variable
                        col_j_1hot_encode                           # add 1hot-encoded variable
                      )

         } else{
           clean_data_to_return[,col_j] <- "ERROR: column not numeric, character, factor or logical"
         }
    }
    
    return( clean_data_to_return )
}


if( "data_rank_transform" %in% ls() ){ cat( "[data_rank_transform] loaded \n note: function can be made a lot more efficient by bulk converting columns of same type (future update) \n")  }

# example usage of data_rank_transform():
# library(tidyverse)
# set.seed(420)
# noddy_data <- tibble( x1 = sample(-30:100, size=100),
#                       x2 = sample( c("ghana", "zambia", "nigeria"),
#                                              size = 100,
#                                              replace = TRUE
#                                             ) %>%
#                                       as.character(),
#                       x3 = sample( c("green", "yellow", "red", "blue"),
#                                           size=100,
#                                           replace=TRUE
#                                         ) %>%
#                                      as_factor(),
#                       x4 = sample(       x = 0:1,
#                                       size = 100,
#                                    replace = TRUE
#                                  ) * rnorm( n = 100,
#                                           mean = 5,
#                                             sd = 10
#                                         ),
#                       x5 = sample( c(TRUE,FALSE),
#                                            size = 100,
#                                            replace = TRUE
#                                  ),
#                       x6 = rep("error", 100 ), 
#                       x7 = sample( c(1:10), size=100, replace=TRUE, prob=c(2,rep(1,9)) )
#                     )
# noddy_data
# data_rank_transform(noddy_data, colnames_to_ignore = c("x6") ) %>% View()
# 

quick_and_dirty_k_means <- 
  function( data_source,
            k_values_to_try
  ){
    
    # data_rank_transform() the dataset
    # perform k-means clustering on all values of k specified in {k_values_to_try}
    # for each k-means model export:
    #   - cluster assignment per sample/row/unit in data
    #   - plot of silhouette values
    #   - summary of variable distributions per cluster
    
  }

if( "quick_and_dirty_k_means" %in% ls() ){ cat( "[quick_and_dirty_k_means] loaded \n")  }

# # example usage of quick_and_dirty_k_means():
# library(tidyverse)
# noddy_data <- tibble(           x0 = sample( paste("segment",1:3,sep="_"),
#                                              size = 100,
#                                              replace = TRUE
#                                            ),
#                         x1_numeric = sample(-30:100, size=100),
#                       x2_character = sample( c("ghana", "zambia", "nigeria"),
#                                              size = 100,
#                                              replace = TRUE
#                                             ) %>%
#                                       as.character(),
#                          x3_factor = sample( c("green", "yellow", "red", "blue"),
#                                           size=100,
#                                           replace=TRUE
#                                         ) %>%
#                                      as_factor(),
#                         x4_logical = sample( c(TRUE,FALSE),
#                                            size = 100,
#                                            replace = TRUE
#                                          )
# 
#                     )

#######################################
### SIMPLE REVENUE UPLIFT SIMULATOR ###
#######################################
simple_revenue_uplift_simulator <- 
  function( features_x,                    # tibble containing features data - 1 row per unit/sample
            response_y,                    # single column containing outcome/response for each unit/sample in X
            proportion_control_grp = 0.1,   # proportion of observations to assign to control group
            uplift_effect_min = -0.025,     # min size of linear uplift coefficient (as % of column mean)
            uplift_effect_max = 0.1        # max size of linear uplift coefficient (as % of column mean)
             
          ){
    
    if( ! "tidyverse" %in% (.packages()) ){ library(tidyverse) } 
    
    features_x <- as_tibble( features_x )
    response_y <- as_tibble( response_y )
    
    generate_linear_uplift_coefs <- 
      sapply( colnames(features_x),
              function(col_j){ col_avg <- mean(features_x[[col_j]])
                                  runif(   n = 1, 
                                         min = uplift_effect_min*abs(col_avg),
                                         max = uplift_effect_max*abs(col_avg)
                                       )
                             }
            )
    
    uplift_row_i <- 
      apply(      X = features_x,
             MARGIN = 1,   # apply to each row of X
                FUN = function(row_i) sum(row_i * generate_linear_uplift_coefs)   # apply this function to each row
           )
    
    treatment_group_assignments <- 
      sample(           0:1, 
                 size = nrow(features_x),
              replace = TRUE,
                 prob = c(proportion_control_grp, 1-proportion_control_grp)
            )
     
    # give people in control group 0 uplift:
    uplift_row_i[ treatment_group_assignments==0 ] <- 0
    
    return( list( linear_uplift_coefs = generate_linear_uplift_coefs,
                     data_with_uplift = bind_cols( features_x, 
                                                   treatment_grp = treatment_group_assignments,
                                                   response_y, 
                                                   deterministic_uplift = uplift_row_i 
                                                  ) 
                )
          )
}


if( "simple_revenue_uplift_simulator" %in% ls() ){ cat( "[simple_revenue_uplift_simulator] loaded \n")  }

# # example usage of simple_revenue_uplift_simulator():
# library(tidyverse)
# X <- tibble( x1 = rnorm(1000, mean=100, sd=40),
#              x2 = rnorm(1000, mean=100, sd=40),
#              x3 = rnorm(1000, mean=100, sd=40),
#              x4 = rnorm(1000, mean=100, sd=40)
#             )
# Y <- tibble( y = sample(0:100, size=1000, replace=TRUE) )
# 
# simple_revenue_uplift_simulator(             features_X = X,
#                                              response_y = Y,
#                                  proportion_control_grp = 0.1,   # i.e. 10% go to control group
#                                       uplift_effect_min = -2,
#                                       uplift_effect_max = 5
#                                )

#######################################
######### REVENUE QINI PLOT ###########
#######################################

revenue_QINI_plot <- 
    function( data_source,
              outcome_y_colname = "y",
              treatment_grp_colname = "treatment_grp",
              model_colnames = NULL,
              n_quantiles = 10,
              n_random_targeting_models_to_add = 0      # for benchmarking
            ){
    
    if( ! "tidyverse" %in% (.packages()) ){ library(tidyverse) }   
    
    data_source <- as_tibble(data_source)    
      
    y_vec <- data_source[[outcome_y_colname]]
    treatment_grp_vec <- data_source[[treatment_grp_colname]]
    
    store_model_results <- list()
    
    # add random targeting_models:
    if( n_random_targeting_models_to_add > 0 ){
        
        for( i in 1:n_random_targeting_models_to_add ){
          
          data_source <- 
            bind_cols( data_source,
                       tibble( temp_colname = rnorm( n=nrow(data_source) ) )           
                     )
          
          random_model_name <- paste0("random_targeting", i)
          
          # rename the column:
          colnames(data_source)[ncol(data_source)] <- random_model_name
          model_colnames <- c( model_colnames, random_model_name )
        }
          
    }
    
    uplift_scores_matrix <- data_source[,model_colnames]
    
    for( model_i in model_colnames ){      # for each model:

      uplift_score_vec_model_i <- uplift_scores_matrix[[model_i]]
      
      # break uplift_score into quantiles:
      uplift_quantile_labels_model_i <-
        cut(               x = uplift_score_vec_model_i,
                      breaks = quantile( x = uplift_score_vec_model_i,
                                         probs = seq(0, 1, length.out=n_quantiles+1 )
                                       ),
              include.lowest = TRUE
            )

      uplift_quantile_ranks_model_i <-
        cut(              x = uplift_score_vec_model_i,
                     breaks = quantile( x = uplift_score_vec_model_i,
                                        probs = seq(0, 1, length.out=n_quantiles+1 )
                                      ),
             include.lowest = TRUE,
                     labels = FALSE
           )

       store_model_results[[model_i]] <-
          tibble(           model_name = model_i,
                         treatment_grp = treatment_grp_vec,
                          uplift_score = uplift_score_vec_model_i,
                       uplift_quantile = uplift_quantile_labels_model_i,
                  uplift_quantile_rank = uplift_quantile_ranks_model_i,
                                     y = y_vec
                ) %>%
         mutate_if( is.factor, as.character ) %>%     # turn factor columns to character
         group_by( model_name, uplift_quantile, uplift_quantile_rank ) %>%
         summarise(         y_total_in_qtl = sum(y),
                    y_total_in_qtl_treated = sum( as.numeric(treatment_grp==1) * y ),
                    y_total_in_qtl_control = sum( as.numeric(treatment_grp==0) * y ),
                            n_total_in_qtl = n(),
                          n_treated_in_qtl = sum( as.numeric(treatment_grp==1) ),
                          n_control_in_qtl = sum( as.numeric(treatment_grp==0) )
                  ) %>%
         ungroup() %>%
         arrange( desc(uplift_quantile_rank) ) %>%
         mutate( proportion_targeted = row_number() / n_quantiles ) %>%
         mutate(         cumsum_n = cumsum(n_total_in_qtl),
                 cumsum_n_treated = cumsum(n_treated_in_qtl),
                 cumsum_n_control = cumsum(n_control_in_qtl),
                 cumsum_y_treated = cumsum(y_total_in_qtl_treated),
                 cumsum_y_control = cumsum(y_total_in_qtl_control)
               ) %>%
         mutate( cum_avg_treated = cumsum_y_treated / cumsum_n_treated,
                 cum_avg_control = cumsum_y_control / cumsum_n_control
               ) %>%
         mutate( cum_avg_uplift_per_person = cum_avg_treated - cum_avg_control ) %>%

         # add a row corresponding to 0 customers targeted:
         bind_rows( tibble(                model_name = model_i,
                                  proportion_targeted = 0,
                            cum_avg_uplift_per_person = 0
                          )
                  )

    }
     
    all_models_table <-
      store_model_results %>%
        purrr::reduce( ., bind_rows )

    # # # data for random targeting line:
    # # random_targeting_line_data <-
    # #   tibble(                model_name = rep( "random_targeting", 2),
    # #                 proportion_targeted = c(0,1),
    # #           cum_avg_uplift_per_person = c( 0,
    # #                                          mean( y_vec_model_i[ treatment_grp_vec==1 ] )-
    # #                                          mean( y_vec_model_i[ treatment_grp_vec==0 ] )
    # #                                        )
    # #         )
     
    # create plot:
    revenue_QINI_plot <-
      all_models_table %>% 
        filter( !grepl("random_targeting", model_name) ) %>%   # remove the random models:
          
      ggplot( data = .,
              aes(      x = proportion_targeted,
                        y = cum_avg_uplift_per_person,
                    group = model_name,
                   colour = model_name
                  )
            ) +
      geom_point( size=3 ) +
      geom_line() +
      scale_x_continuous( breaks = seq(0,1,0.1) ) +
      
      labs( title = "Revenue QINI plot",
            subtitle = paste0( n_random_targeting_models_to_add, " random targeting (benchmark) models added")
            ) +
      geom_hline( yintercept = 0 )

    # add the random models to the plot:
    if( n_random_targeting_models_to_add > 0 ){
      
      revenue_QINI_plot <- 
        revenue_QINI_plot + 
          # add the random models:
          geom_line( data = all_models_table %>% filter( grepl("random_targeting", model_name) ), 
                     colour = "white",
                     size = 1.1
      ) 
      
    }
    
    return( list( cumulative_results_table = all_models_table,
                         revenue_QINI_plot = revenue_QINI_plot
                )
          )
  }

if( "revenue_QINI_plot" %in% ls() ){ cat( "[revenue_QINI_plot] loaded \n") }

# # example usage of revenue_QINI_plot():
# set.seed(1460)
# noddy_data <- tibble( customer_spend = sample(0:100, size=1000, replace=TRUE),
#                       experimental_group = sample( 0:1, size=1000, prob=c(0.2, 0.8), replace=TRUE )
#                     ) %>%
#                 mutate( model1 = rnorm( n(), 50, 25 ),
#                         model2 = rnorm( n(), 0, 100 ),
#                         model3 = rnorm( n(), 20,20 )
#                       )
# 
# run_revenue_QINI_ftn <-
#   revenue_QINI_plot(           data_source = noddy_data,
#                          outcome_y_colname = "customer_spend",
#                      treatment_grp_colname = "experimental_group",
#                               model_colnames = c("model1","model2","model3"),
#                               n_quantiles = 10,
#                       n_random_targeting_models_to_add = 2
#                     )
# 
# run_revenue_QINI_ftn$cumulative_results_table %>% View
# run_revenue_QINI_ftn$revenue_QINI_plot

# # error checking:
# model2_only <- noddy_data %>% select(customer_spend, experimental_group, model2)
# top_70_pct <- model2_only %>% top_n( n = nrow(.)*0.6, wt=model2 )
# top_70_pct %>% count(experimental_group) 
# top_70_pct %>% group_by(experimental_group) %>% summarise( sum_spend = sum(customer_spend) )
# top_70_pct %>% group_by(experimental_group) %>% summarise( avg_spend = mean(customer_spend) )
