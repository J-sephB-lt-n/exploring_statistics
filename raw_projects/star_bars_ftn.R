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

# if( "star_bars_ftn" %in% ls() ){ cat( "[star_bars_ftn], by Joe, loaded \n")  }

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


