import os
from google.cloud import bigquery
import pandas as pd
import google.auth

# set environment variable:
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/Joseph.Boltman/Documents/2020/google_key/mr-d-food-app-7cd669657b90.json'

# Initialize a BigQuery client
client = bigquery.Client()

# Define query
query_job = client.query("""
                            SELECT event_name, count(*) as count
                              FROM `mr-d-food-app.analytics_153657482.events_*`
                             WHERE (_TABLE_SUFFIX = '20200308') 
                             GROUP BY event_name
                             ORDER BY count desc
                        """)

# get query result:
results = query_job.result()  # Waits for job to complete.                        

# convert results to dataframe:
print( results.to_dataframe() )
