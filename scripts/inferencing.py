from src.inferencer import Inferencer
from src.enums import LlmModels
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os

config_path = 'config/exp_1.yaml'
fine_tuned_model='fine_tuned_models/exp_1-13b-Instruct-hf'

inferencer = Inferencer(base_model=str(LlmModels.LLAMA_2_13B_INSTRUCT), fine_tuned_model=fine_tuned_model, config_path=config_path)
text_generation_pipeline = inferencer.get_pipeline()


load_dotenv()
NEO4J_URL =  os.getenv("NEO4J_URL", "")
NEO4J_USER_NAME =  os.getenv("NEO4J_USER_NAME", "")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD", "")

question = "List all the providers whose locality is Germany?"

# Get graph schema
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER_NAME, password=NEO4J_PASSWORD)
schema = graph.schema
graph._driver.close()

prompt = f"""<s>[INST] <<SYS>>
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema: 
{schema}
<</SYS>>

{question}  [/INST]"""

result = text_generation_pipeline(prompt)
print("LLM OUTPUT:",result[0]['generated_text'])

