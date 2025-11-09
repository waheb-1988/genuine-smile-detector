"""Minimal SQL agent utilities for BigQuery."""
from google.cloud import bigquery

def average_satisfaction_per_month(project_id: str, full_table: str):
    client = bigquery.Client(project=project_id)
    sql = f'''
    SELECT
      FORMAT_DATE('%Y-%m', event_date) AS month,
      AVG(satisfaction) AS avg_satisfaction
    FROM `{full_table}`
    GROUP BY month
    ORDER BY month
    '''
    return client.query(sql).result().to_dataframe()
