from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
import json
import pandas as pd
from datasets import load_dataset

load_dotenv()

NEO4J_URL =  os.getenv("NEO4J_URL", "")
NEO4J_USER_NAME =  os.getenv("NEO4J_USER_NAME", "")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD", "")

graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER_NAME, password=NEO4J_PASSWORD)
schema = graph.schema
graph._driver.close()

# Path to JSONL file
jsonl_file_path = 'datasets/nl2cypher_30.jsonl'

# List to hold all entries
entries = []

# Open and read the JSONL file
with open(jsonl_file_path, 'r') as file:
    for line in file:
        # Parse each line as JSON
        data = json.loads(line)
        
        # Format the system prompt with the schema
        formatted_entry = f"""<s>[INST] <<SYS>>
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema: 
{schema}
<</SYS>>

{data['nl']}  [/INST] {data['cypher']} </s>"""
        
        entries.append({"text": formatted_entry})

df = pd.DataFrame(entries)

# Save to Parquet file
parquet_file_path = 'datasets/nl2cypher_30.parquet'
df.to_parquet(parquet_file_path)

dataset = load_dataset('parquet', data_files=parquet_file_path)

# Print the first item to verify
print(dataset['train'][0])

