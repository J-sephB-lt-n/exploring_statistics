
# reference: https://stackoverflow.com/questions/29353096/add-a-sequence-number-to-each-element-in-a-group-using-python
# note that this will assign number in the order that is present in the data

df['sequence']=df.groupby('patient').cumcount()
