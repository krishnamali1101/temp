
Yes, running pipeline.run(documents=[Document.example()]) can add duplicate documents if the same document is processed multiple times without any deduplication strategy in place. To avoid adding duplicate documents, you can use the docstore_strategy parameter in the IngestionPipeline to handle duplicates.

According to the LlamaIndex documentation, you can set the docstore_strategy to UPSERTS, which checks if a document is already in the docstore based on its ID and hash. If the document is new or has been updated, it will be processed; otherwise, it will be skipped.

Here's an example of how to configure the pipeline to avoid duplicates:

python

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    docstore=SimpleDocumentStore(),
    docstore_strategy='upserts',  # Use upserts to handle duplicates
)
nodes = pipeline.run(documents=[Document.example()])
Would you like more detail on setting up the pipeline or handling duplicates?
==============
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined set of "summarize" templates
summarization_templates = [
    "Summarize this document.",
    "Provide a summary of this document.",
    "Can you summarize the entire document?",
    "Give an overview of this document.",
    "What is this document about?",
]

# Embed the summarization templates
template_embeddings = model.encode(summarization_templates)

def calculate_cosine_similarity(query_embedding, template_embeddings):
    """
    Calculate cosine similarity between query embedding and template embeddings.
    """
    similarity_scores = np.dot(template_embeddings, query_embedding) / (
        np.linalg.norm(template_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    return similarity_scores

def determine_top_k(query, summarization_templates, template_embeddings, low_k=5, mid_k=10, high_k=20, threshold=0.8):
    """
    Determine similarity_top_k dynamically based on query type and similarity scores.

    Recommendations for 100-page documents:
    1. Specific Queries (e.g., "What is the conclusion in section 4?"):
       - Use low_k (5–10) to focus on the most relevant chunks.
    2. Broad Queries (e.g., "Summarize this document"):
       - Use high_k (15–20 or more) to gather enough data for a comprehensive response.
    3. Adaptive Approach:
       - Short queries: Lower similarity_top_k (5–10).
       - Longer queries: Higher similarity_top_k (15–30).
    """
    # Embed the query
    query_embedding = model.encode([query])[0]
    
    # Calculate cosine similarity with summarization templates
    similarity_scores = calculate_cosine_similarity(query_embedding, template_embeddings)
    
    # Get the maximum similarity score
    max_similarity = max(similarity_scores)

    # Determine top_k based on similarity and query length
    if max_similarity >= threshold:  # Broad query
        top_k = high_k
    elif len(query.split()) > 10:  # Long exploratory query
        top_k = mid_k
    else:  # Specific or focused query
        top_k = low_k
    
    return top_k, max_similarity

# Example queries
query1 = "Summarize this document."
query2 = "Explain the content of page 4."
query3 = "What is discussed in the results section?"

# Determine top_k for each query
top_k_query1, similarity1 = determine_top_k(query1, summarization_templates, template_embeddings)
top_k_query2, similarity2 = determine_top_k(query2, summarization_templates, template_embeddings)
top_k_query3, similarity3 = determine_top_k(query3, summarization_templates, template_embeddings)

# Output results
print(f"Query 1: {query1}, Top K: {top_k_query1}, Similarity: {similarity1:.2f}")
print(f"Query 2: {query2}, Top K: {top_k_query2}, Similarity: {similarity2:.2f}")
print(f"Query 3: {query3}, Top K: {top_k_query3}, Similarity: {similarity3:.2f}")

# Recommendations for response_mode
def choose_response_mode(query):
    """
    Choose response_mode based on query type:
    1. Use 'tree_summarizer' for broad summaries (hierarchical summarization for large datasets).
    2. Use 'compact' or 'simple' for faster, direct results for specific answers.
    """
    if "summarize" in query.lower() or "overview" in query.lower():
        return "tree_summarizer"
    else:
        return "compact"

# Example usage with query engine
response_mode_query1 = choose_response_mode(query1)
response_mode_query2 = choose_response_mode(query2)

print(f"Response Mode for Query 1: {response_mode_query1}")
print(f"Response Mode for Query 2: {response_mode_query2}")

# Integrating with query engine
query_engine = index.as_query_engine(
    response_mode=response_mode_query1,
    similarity_top_k=top_k_query1  # Dynamic top_k based on the query
)

# Example response
response = query_engine.query(query1)
print("Response:", response)


===========


def update_requirements_ordered_complete(old_file, new_file, output_file):
    # Read old requirements into a list and a dictionary
    with open(old_file, 'r') as f:
        old_reqs = [line.strip() for line in f if "==" in line]
        old_reqs_dict = {line.split("==")[0]: line.strip() for line in old_reqs}
    
    # Read new requirements into a dictionary
    with open(new_file, 'r') as f:
        new_reqs_dict = {line.split("==")[0]: line.strip() for line in f if "==" in line}
    
    # Generate updated requirements
    updated_reqs = []
    for line in old_reqs:
        lib_name = line.split("==")[0]
        if lib_name in new_reqs_dict:
            updated_reqs.append(new_reqs_dict[lib_name])  # Use version from new_req.txt
        else:
            updated_reqs.append(line)  # Keep the version from old_req.txt
    
    # Write the updated requirements to the output file
    with open(output_file, 'w') as f:
        f.write("\n".join(updated_reqs))
    
    print(f"Updated requirements saved to {output_file}")

# Usage
update_requirements_ordered_complete('old_req.txt', 'new_req.txt', 'updated_req.txt')


# Usage
update_requirements_ordered_complete('old_req.txt', 'new_req.txt', 'updated_req.txt')




%%writefile requirements.txt
llama-index
llama-index-llms-huggingface
llama-index-embeddings-fastembed
fastembed
Unstructured[md]
chromadb
llama-index-vector-stores-chroma
llama-index-llms-groq
einops
accelerate
sentence-transformers
llama-index-llms-mistralai
llama-index-llms-openai


from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,SummaryIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool,QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters,FilterCondition
from typing import List,Optional

import  nest_asyncio
nest_asyncio.apply()


documents = SimpleDirectoryReader(input_files = ['./data/self_rag_arxiv.pdf']).load_data()
print(len(documents))
print(f"Document Metadata: {documents[0].metadata}")

splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Length of nodes : {len(nodes)}")
print(f"get the content for node 0 :{nodes[0].get_content(metadata_mode='all')}")

# Instantiate the vectorstore

import chromadb
db = chromadb.PersistentClient(path="./chroma_db_mistral")
chroma_collection = db.get_or_create_collection("multidocument-agent")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Instantiate the embedding model

from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
#
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
#
Settings.embed_model = embed_model
#
Settings.chunk_size = 1024
#


#Instantiate the LLM

from llama_index.llms.mistralai import MistralAI
os.environ["MISTRAL_API_KEY"] = userdata.get("MISTRAL_API_KEY")
llm = MistralAI(model="mistral-large-latest")



#instantiate Vectorstore
name = "BERT_arxiv"
vector_index = VectorStoreIndex(nodes,storage_context=storage_context)
vector_index.storage_context.vector_store.persist(persist_path="/content/chroma_db")
#
# Define Vectorstore Autoretrieval tool
def vector_query(query:str,page_numbers:Optional[List[str]]=None)->str:
  '''
  perform vector search over index on
  query(str): query string needs to be embedded
  page_numbers(List[str]): list of page numbers to be retrieved,
                          leave blank if we want to perform a vector search over all pages
  '''
  page_numbers = page_numbers or []
  metadata_dict = [{"key":'page_label',"value":p} for p in page_numbers]
  #
  query_engine = vector_index.as_query_engine(similarity_top_k =2,
                                              filters = MetadataFilters.from_dicts(metadata_dict,
                                                                                    condition=FilterCondition.OR)
                                              )
  #
  response = query_engine.query(query)
  return response
#
#llamiondex FunctionTool wraps any python function we feed it
vector_query_tool = FunctionTool.from_defaults(name=f"vector_tool_{name}",
                                              fn=vector_query)
# Prepare Summary Tool
summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize",
                                                      se_async=True,)
summary_query_tool = QueryEngineTool.from_defaults(name=f"summary_tool_{name}",
                                                    query_engine=summary_query_engine,
                                                  description=("Use ONLY IF you want to get a holistic summary of the documents."
                                              "DO NOT USE if you have specified questions over the documents."))
Test the LLM

response = llm.predict_and_call([vector_query_tool],
                                "Summarize the content in page number 2",
                                verbose=True)
######################RESPONSE###########################
=== Calling Function ===
Calling function: vector_tool_BERT_arxiv with args: {"query": "summarize content", "page_numbers": ["2"]}
=== Function Output ===
The content discusses the use of RAG models for knowledge-intensive generation tasks, such as MS-MARCO and Jeopardy question generation, showing that the models produce more factual, specific, and diverse responses compared to a BART baseline. The models also perform well in FEVER fact verification, achieving results close to state-of-the-art pipeline models. Additionally, the models demonstrate the ability to update their knowledge as the world changes by replacing the non-parametric memory.
Helper function to generate Vectorstore Tool and Summary tool for all the documents
def get_doc_tools(file_path:str,name:str)->str:
  '''
  get vector query and sumnmary query tools from a document
  '''
  #load documents
  documents = SimpleDirectoryReader(input_files = [file_path]).load_data()
  print(f"length of nodes")
  splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
  nodes = splitter.get_nodes_from_documents(documents)
  print(f"Length of nodes : {len(nodes)}")
  #instantiate Vectorstore
  vector_index = VectorStoreIndex(nodes,storage_context=storage_context)
  vector_index.storage_context.vector_store.persist(persist_path="/content/chroma_db")
  #
  # Define Vectorstore Autoretrieval tool
  def vector_query(query:str,page_numbers:Optional[List[str]]=None)->str:
    '''
    perform vector search over index on
    query(str): query string needs to be embedded
    page_numbers(List[str]): list of page numbers to be retrieved,
                            leave blank if we want to perform a vector search over all pages
    '''
    page_numbers = page_numbers or []
    metadata_dict = [{"key":'page_label',"value":p} for p in page_numbers]
    #
    query_engine = vector_index.as_query_engine(similarity_top_k =2,
                                                filters = MetadataFilters.from_dicts(metadata_dict,
                                                                                     condition=FilterCondition.OR)
                                                )
    #
    response = query_engine.query(query)
    return response
  #
  #llamiondex FunctionTool wraps any python function we feed it
  vector_query_tool = FunctionTool.from_defaults(name=f"vector_tool_{name}",
                                                fn=vector_query)
  # Prepare Summary Tool
  summary_index = SummaryIndex(nodes)
  summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize",
                                                       se_async=True,)
  summary_query_tool = QueryEngineTool.from_defaults(name=f"summary_tool_{name}",
                                                     query_engine=summary_query_engine,
                                                    description=("Use ONLY IF you want to get a holistic summary of the documents."
                                                "DO NOT USE if you have specified questions over the documents."))
  return vector_query_tool,summary_query_tool

Prepare a input list with specified document names

import os
root_path = "/content/data"
file_name = []
file_path = []
for files in os.listdir(root_path):
  if file.endswith(".pdf"):
    file_name.append(files.split(".")[0])
    file_path.append(os.path.join(root_path,file))
#
print(file_name)
print(file_path)

################################RESPONSE###############################
['self_rag_arxiv', 'crag_arxiv', 'RAG_arxiv', '', 'BERT_arxiv']
['/content/data/BERT_arxiv.pdf',
 '/content/data/BERT_arxiv.pdf',
 '/content/data/BERT_arxiv.pdf',
 '/content/data/BERT_arxiv.pdf',
 '/content/data/BERT_arxiv.pdf']



papers_to_tools_dict = {}
for name,filename in zip(file_name,file_path):
  vector_query_tool,summary_query_tool = get_doc_tools(filename,name)
  papers_to_tools_dict[name] = [vector_query_tool,summary_query_tool]



Get the tools into a flat list

initial_tools = [t for f in file_name for t in papers_to_tools_dict[f]]
initial_tools


from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
#
obj_index = ObjectIndex.from_objects(initial_tools,index_cls=VectorStoreIndex)
#



obj_retriever = obj_index.as_retriever(similarity_top_k=2)
tools = obj_retriever.retrieve("compare and contrast the papers self rag and corrective rag")
#
print(tools[0].metadata)
print(tools[1].metadata)

###################################RESPONSE###########################
ToolMetadata(description='Use ONLY IF you want to get a holistic summary of the documents.DO NOT USE if you have specified questions over the documents.', name='summary_tool_self_rag_arxiv', fn_schema=<class 'llama_index.core.tools.types.DefaultToolFnSchema'>, return_direct=False)

ToolMetadata(description='vector_tool_self_rag_arxiv(query: str, page_numbers: Optional[List[str]] = None) -> str\n\n    perform vector search over index on\n    query(str): query string needs to be embedded\n    page_numbers(List[str]): list of page numbers to be retrieved,\n                            leave blank if we want to perform a vector search over all pages\n    ', name='vector_tool_self_rag_arxiv', fn_schema=<class 'pydantic.v1.main.vector_tool_self_rag_arxiv'>, return_direct=False)



Setup the RAG Agent
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
#
agent_worker = FunctionCallingAgentWorker.from_tools(tool_retriever=obj_retriever,
                                                     llm=llm,
                                                     system_prompt="""You are an agent designed to answer queries over a set of given papers.
                                                     Please always use the tools provided to answer a question.Do not rely on prior knowledge.""",
                                                     verbose=True)
agent = AgentRunner(agent_worker)




Ask Query 1

#
response = agent.query("Compare and contrast self rag and crag.")
print(str(response))

##############################RESPONSE###################################
Added user message to memory: Compare and contrast self rag and crag.
=== LLM Response ===
Sure, I'd be happy to help you understand the differences between Self RAG and CRAG, based on the functions provided to me.

Self RAG (Retrieval-Augmented Generation) is a method where the model generates a holistic summary of the documents provided as input. It's important to note that this method should only be used if you want a general summary of the documents, and not if you have specific questions over the documents.

On the other hand, CRAG (Contrastive Retrieval-Augmented Generation) is also a method for generating a holistic summary of the documents. The key difference between CRAG and Self RAG is not explicitly clear from the functions provided. However, the name suggests that CRAG might use a contrastive approach in its retrieval process, which could potentially lead to a summary that highlights the differences and similarities between the documents more effectively.

Again, it's crucial to remember that both of these methods should only be used for a holistic summary, and not for answering specific questions over the documents.




Ask Query 2

response = agent.query("Summarize the paper corrective RAG.")
print(str(response))
###############################RESPONSE#######################
Added user message to memory: Summarize the paper corrective RAG.
=== Calling Function ===
Calling function: summary_tool_RAG_arxiv with args: {"input": "corrective RAG"}
=== Function Output ===
The corrective RAG approach is a method used to address issues or errors in a system by categorizing them into three levels: Red, Amber, and Green. Red signifies critical problems that need immediate attention, Amber indicates issues that require monitoring or action in the near future, and Green represents no significant concerns. This approach helps prioritize and manage corrective actions effectively based on the severity of the identified issues.
=== LLM Response ===
The corrective RAG approach categorizes issues into Red, Amber, and Green levels to prioritize and manage corrective actions effectively based on severity. Red signifies critical problems needing immediate attention, Amber requires monitoring or action soon, and Green indicates no significant concerns.
assistant: The corrective RAG approach categorizes issues into Red, Amber, and Green levels to prioritize and manage corrective actions effectively based on severity. Red signifies critical problems needing immediate attention, Amber requires monitoring or action soon, and Green indicates no significant concerns.
