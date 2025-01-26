from src.inferencer import Inferencer
from src.enums import LlmModels
from langchain_community.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import GraphCypherQAChain
from dotenv import load_dotenv
import os

config_path = 'config/exp_1.yaml'
fine_tuned_model='fine_tuned_models/exp_1-13b-Instruct-hf'

inferencer = Inferencer(base_model=str(LlmModels.LLAMA_2_13B_INSTRUCT), fine_tuned_model=fine_tuned_model, config_path=config_path)
text_generation_pipeline = inferencer.get_pipeline()

# Load environment variables
load_dotenv()
NEO4J_URL =  os.getenv("NEO4J_URL", "")
NEO4J_USER_NAME =  os.getenv("NEO4J_USER_NAME", "")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD", "")

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

CYPHER_GENERATION_TEMPLATE = """<s>[INST] <<SYS>>
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema: 
{schema}
<</SYS>>

{question}  [/INST]"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# Get graph schema
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER_NAME, password=NEO4J_PASSWORD)

chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
)

print(chain.run("what are the services provided by 'Fraunhofer-Institut für Angewandte Informationstechnik FIT"))