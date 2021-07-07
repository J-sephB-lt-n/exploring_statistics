
import pandas as pd

table_column_data = pd.read_csv('C:/Users/jbolton/Desktop/temp_joe_adspend_table_columns_all.csv')[['TABLE_NAME','COLUMN_NAME']]
table_column_data['TABLE_NAME'].drop_duplicates().values