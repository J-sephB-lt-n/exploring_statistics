from openpyxl import load_workbook

# initiate workbook
writer = pd.ExcelWriter('days_numbers.xlsx', engine='openpyxl')

# Write each dataframe to a different worksheet.
active_cust.to_excel(writer, sheet_name='cumulative_active_customers', index=False)
all_data.to_excel(writer, sheet_name='customer_level_data', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()