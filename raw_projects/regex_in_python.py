import pandas as pd
import re

# remove domain from email:
original_string = 'email address, phone number: johndoe@yahoo.com 0829313556'
extract_email_domain = re.search( pattern = r'@(\w+)\.',
                                  string = original_string
                                )
print( extract_email_domain[0] )
print( extract_email_domain[1] )

# remove all characters that are not numbers from a string:
original_string = 'words 12 must 34 go 56 but 78 numbers 910 can 1112 stay 1314'
remove_non_numbers = re.sub( pattern = r'\D',      # what to replace (\D means all non-numbers)
                                repl = '',            # what to replace it with 
                              string = original_string
                           )
print(remove_non_numbers)                           


 