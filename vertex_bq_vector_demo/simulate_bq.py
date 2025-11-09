"""Create a demo BigQuery dataset/table and insert sample rows."""
from datetime import date
from typing import Optional
from google.cloud import bigquery

def create_demo_table(project_id: str, dataset: str = "demo_satisfaction", table: str = "satisfaction_survey") -> str:
    client = bigquery.Client(project=project_id)
    ds_ref = bigquery.Dataset(f"{project_id}.{dataset}")
    try:
        client.create_dataset(ds_ref)
        print("Created dataset:", dataset)
    except Exception as e:
        if "Already Exists" in str(e):
            print("Dataset exists:", dataset)
        else:
            raise

    full = f"{project_id}.{dataset}.{table}"
    schema = [
        bigquery.SchemaField("event_date", "DATE"),
        bigquery.SchemaField("region", "STRING"),
        bigquery.SchemaField("satisfaction", "INT64"),
        bigquery.SchemaField("comments", "STRING"),
    ]
    tbl = bigquery.Table(full, schema=schema)
    try:
        client.create_table(tbl)
        print("Created table:", full)
    except Exception as e:
        if "Already Exists" in str(e):
            print("Table exists:", full)
        else:
            raise

    rows = [
        {"event_date": date(2025, 8,  2), "region":"EMEA", "satisfaction":4, "comments":"good"},
        {"event_date": date(2025, 8, 15), "region":"EMEA", "satisfaction":3, "comments":"ok"},
        {"event_date": date(2025, 9,  1), "region":"AMER", "satisfaction":5, "comments":"great"},
        {"event_date": date(2025, 9, 21), "region":"APAC", "satisfaction":2, "comments":"slow support"},
        {"event_date": date(2025,10, 5), "region":"AMER", "satisfaction":4, "comments":"helpful"},
        {"event_date": date(2025,10,18), "region":"EMEA", "satisfaction":1, "comments":"bad"},
    ]
    client.insert_rows_json(full, rows)
    print("Inserted sample rows.")
    return full
