from math import floor, copysign
import pandas as pd

def joe_star_bars_ftn_python(vec, set_min='default', bar_length=9):
    
    negative_values_present = min(vec)<0
    
    if negative_values_present == False:
        if set_min == 'default':
            set_min = min(vec)
        star_width = (max(vec)-set_min)/bar_length
        calc_bar_lengths = [ floor(vec[i]/star_width) for i in range(len(vec)) ]
        star_bar_vec = [ 'x'*i for i in calc_bar_lengths ]
    else:
        star_width = max( [abs(i) for i in vec] ) / bar_length
        calc_bar_lengths = [ copysign(1,vec[i])*floor( abs(vec[i])/star_width ) for i in range(len(vec)) ]
        #return(calc_bar_lengths)
        star_bar_vec = [ '.'*int(bar_length-abs(i))+'x'*int(abs(i))+'0'+'.'*bar_length if i<0 else '.'*bar_length+'0'+'x'*int(i)+'.'*int(bar_length-i) for i in calc_bar_lengths ]
    
    return star_bar_vec

    # example use:
    # star_bars_ftn_python( vec=[1,55,34,64,99,12,15,23], bar_length=20, set_min=0 )
    # star_bars_ftn_python( vec=[1,-55,34,64,-99,-12,15,23], bar_length=20 )