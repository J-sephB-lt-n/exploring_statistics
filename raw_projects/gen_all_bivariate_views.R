
gen_all_bivariate_views <- function( df ){
  
  #   column 1 | column 2   | Bivariate Plot
  # -----------|------------|-------------------
  # continuous | continuous | scatterplot with marginal histograms
  # continuous | categorical | raincloud plot
  # categorical | categorical | Various contingency tables plotted as barplots
  
  library(tidyverse)
  
  # store a list of all 2-column combinations:
  store_all_vbl_combinations <- 
    combn( x = names(df),
           m = 2,
           simplify = FALSE
         )
  
  # create a list in which to store the exported plots:
  plot_list <- list()
  
  # make the plots:
  for( i in 1:length(store_all_vbl_combinations) ){
    
     # only keep the 2 variables for plot i: 
      plotdata <- df %>% select( store_all_vbl_combinations[[i]] )
      
      # remove rows contain NA values:
      
    
  }
  
  
  
  
  
}