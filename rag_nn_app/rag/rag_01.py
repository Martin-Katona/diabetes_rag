import os
import shutil

import fitz
from collections import defaultdict

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate



# CHROMA_PATH = "chroma"
# DATA_PATH = "data"

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths for Chroma and data directories relative to the script directory
CHROMA_PATH = os.path.join(script_dir, "chroma")
DATA_PATH = os.path.join(script_dir, "data")
#DATA_PATH = os.path.join(script_dir, "data-test") #for testing




def populate_database(emb_model,reset=False):
    """
    Main function to manage the database and document processing.

    :param reset: If True, clears the database before processing new documents.
    """
    if reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(emb_model,chunks)

    print("Process completed successfully.")


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    
    # Add metadata to each document
    for doc in documents:
        pdf_path = doc.metadata.get('source')
        if pdf_path:
            # Extract metadata using PyMuPDF
            pdf_document = fitz.open(pdf_path)
            metadata = pdf_document.metadata
            doc.metadata['title'] = metadata.get('title', 'Unknown Title')
            doc.metadata['author'] = metadata.get('author', 'Unknown Author')
            doc.metadata['link'] = metadata.get('keywords', 'Unknown Link')
    
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(emb_model,chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(emb_model)
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print(" No new documents to add")


def calculate_chunk_ids(chunks):

    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks



def clear_database():
    # Check if the database path exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Delete the directory and its contents
        print(f"Clearing Chroma database at '{CHROMA_PATH}'")
        
        if not os.path.exists(CHROMA_PATH):
            print(f" The Chroma database has been successfully deleted.")
        else:
            print(f" The Chroma database was not deleted.")

        
    else:
        print(f" Chroma database at '{CHROMA_PATH}' does not exist.")
    
    

def get_embedding_function(emb_model):
    embeddings = OllamaEmbeddings(model=emb_model)
    return embeddings





PROMPT_TEMPLATE = """
Role: You are a helper tool for extracting information and presenting them.

Provide a direct and concise answer to the question based on the retrieved context:

{context}

---
Answer the question based on the above context in 2-3 sentences: {question}


"""

REPHRASE_PROMPT_TEMPLATE = """
You are a helpful assistant. Rewrite the following text so that it is clear, concise, and self-contained. Ensure the rephrased version stands on its own without referring to any external context, source, or prior text.
TEXT: {text}
"""


def query_rag_nested(query_text: str, model, emb_model,fusion=False):

    """
    Executes a RAG pipeline with Reciprocal Rank Fusion (RRF) for improved ranking.

    Args:
        query_text (str): The input query text.
        model: The language model used for generating responses.
        emb_model: The embedding model used for similarity search.

    Returns:
        tuple: A tuple containing the rephrased response and formatted sources.
    """

    if fusion:

        # Prepare the database
        embedding_function = get_embedding_function(emb_model)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        if not db.get()["ids"]:
            print("Chroma database is empty.")
            return "No documents found.", ""

        llm = OllamaLLM(model=model)

        # Step 1: Generate Multiple Queries
        generated_queries = generate_queries(query_text, llm, num_queries=3)
        print("Generated Queries:", generated_queries)

        # Step 2: Retrieve Results for Each Query
        all_results = []
        result_sets = []  # To store ranked lists of document IDs
        doc_metadata = {}  # To store metadata for each document

        for q in generated_queries:
            results = db.similarity_search_with_score(q, k=3)
            ranked_docs = []
            for doc, _score in results:
                ranked_docs.append(doc.metadata["id"])
                doc_metadata[doc.metadata["id"]] = doc  # Store metadata by document ID
            result_sets.append(ranked_docs)
            all_results.extend(results)

        # Step 3: Apply Reciprocal Rank Fusion
        fused_doc_ids = reciprocal_rank_fusion(result_sets)

        # Step 4: Fuse Retrieved Documents
        unique_documents = [doc_metadata[doc_id] for doc_id in fused_doc_ids]

        # Step 5: Generate Context and Response
        context_text = "\n\n---\n\n".join([doc.page_content for doc in unique_documents])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        response_text = llm.invoke(prompt)

        # Step 6: Rephrase the Response
        rephrase_prompt_template = ChatPromptTemplate.from_template(REPHRASE_PROMPT_TEMPLATE)
        rephrase_prompt = rephrase_prompt_template.format(text=response_text)
        rephrased_response_text = llm.invoke(rephrase_prompt)

        # Step 7: Format Sources
        sources = [
            {
                "id": doc.metadata.get("id", "No ID"),
                "title": doc.metadata.get("title", "No Title"),
                "author": doc.metadata.get("author", "No Author"),
                "source": doc.metadata.get("source", "No Source"),
                "page": doc.metadata.get("page", "Unknown Page"),
                "link": doc.metadata.get("link", "No Link")
            }
            for doc in unique_documents
        ]

        grouped_sources = defaultdict(lambda: {"id": "", "title": "", "author": "", "pages": set(), "link": ""})
        
        for source in sources:
            source_path = source["source"]
            grouped_sources[source_path]["id"] = source["id"]
            grouped_sources[source_path]["title"] = source["title"]
            grouped_sources[source_path]["author"] = source["author"]
            grouped_sources[source_path]["link"] = source["link"]
            grouped_sources[source_path]["pages"].add(str(source["page"]))

        formatted_sources = "\n".join([
            f"  - Title: {group['title']}\n"
            f"  - Author: {group['author']}\n"
            f"  - Pages: {', '.join(sorted(group['pages'], key=int))}\n"
            f"  - Link: {group['link']}\n"
            for source_path, group in grouped_sources.items()
        ])

        return rephrased_response_text, formatted_sources

    else:

        # Prepare the DB.
        embedding_function = get_embedding_function(emb_model)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # print("Number of documents in database:", len(db.get()["ids"]))
        collection = db.get()

        if not collection["ids"]:
            print("Chroma database is empty.")
        #else:
        #    print(f"✅ Chroma database contains {len(collection['ids'])} documents.")
        
        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = OllamaLLM(model=model)
        response_text = model.invoke(prompt)

        # Rephrase the response
        rephrase_prompt_template = ChatPromptTemplate.from_template(REPHRASE_PROMPT_TEMPLATE)
        rephrase_prompt = rephrase_prompt_template.format(text=response_text)
        rephrased_response_text = model.invoke(rephrase_prompt)

        # Now get the metadata such as title, author, etc.
        sources = [
            {
                "id": doc.metadata.get("id", "No ID"),
                "title": doc.metadata.get("title", "No Title"),
                "author": doc.metadata.get("author", "No Author"),
                "source": doc.metadata.get("source", "No Source"),
                "page": doc.metadata.get("page", "Unknown Page"),
                "link": doc.metadata.get("link", "No Link")  # Assuming keywords store the document link
            }
            for doc, _score in results
        ]

        # Group sources by document and collect pages
        grouped_sources = defaultdict(lambda: {"id": "","title": "", "author": "", "pages": set(), "link": ""})
        
        for source in sources:
            source_path = source["source"]
            grouped_sources[source_path]["id"] = source["id"]
            grouped_sources[source_path]["title"] = source["title"]
            grouped_sources[source_path]["author"] = source["author"]
            grouped_sources[source_path]["link"] = source["link"]  # Store the document link
            grouped_sources[source_path]["pages"].add(str(source["page"]))  # Store pages as strings

        # Format the sources in a nice, readable format
        formatted_sources = "\n".join([
            # f"Source: {source_path}\n"
            #f"Source:\n"
            #f"  - ID: {group['id']}\n"
            f"  - Title: {group['title']}\n"
            f"  - Author: {group['author']}\n"
            f"  - Pages: {', '.join(sorted(group['pages'], key=int))}\n"
            f"  - Link: {group['link']}\n"  # Adding the document link
            for source_path, group in grouped_sources.items()
        ])

        return rephrased_response_text, formatted_sources




#___________________________________________________________________________________________________________________

#FUSION RAG


### ADD QUERY GENERATION FUNCTION
def generate_queries(original_query, model, num_queries=3):
    """
    Generates multiple related queries from the original query using a language model.

    Args:
        original_query (str): The original query text.
        model: The language model used to generate the queries.
        num_queries (int, optional): The number of related queries to generate. Defaults to 3.

    Returns:
        list: A list of generated query strings.
    """
    # Construct a prompt for the language model to generate related queries
    prompt_template = ChatPromptTemplate.from_template(
        """You are generating questions that are well optimized for retrieval. \
Your goal is to generate {num_queries} questions, that will find the answer to: {question}
You must adhere to the following guidelines:
* Each question should be highly specific to address a particular angle of the main topic
* Each question should be distinct but still related to the original question
* Each question should be able to retrieve different context"""
    )
    prompt = prompt_template.format(num_queries=num_queries, question=original_query)
    # Generate the related queries using the language model
    generated_queries = model.invoke(prompt)

    # Split the generated queries into a list
    related_queries = generated_queries.strip().split("\n")

    return related_queries




def reciprocal_rank_fusion(result_sets, k=60):
    """
    Applies Reciprocal Rank Fusion (RRF) to combine multiple ranked result sets.

    Args:
        result_sets (list of lists): A list where each element is a ranked list of documents (result set).
        k (int): A constant used in the RRF formula (default is 60).

    Returns:
        list: A list of documents sorted by their combined RRF scores.
    """
    rrf_scores = defaultdict(float)

    for result_set in result_sets:
        for rank, doc in enumerate(result_set, start=1):
            rrf_scores[doc] += 1 / (k + rank)

    # Sort documents by their RRF scores in descending order
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs]

'''
def query_rag_nested_fusion(query_text: str, model, emb_model):
    """
    Executes a RAG pipeline with Reciprocal Rank Fusion (RRF) for improved ranking.

    Args:
        query_text (str): The input query text.
        model: The language model used for generating responses.
        emb_model: The embedding model used for similarity search.

    Returns:
        tuple: A tuple containing the rephrased response and formatted sources.
    """
    # Prepare the database
    embedding_function = get_embedding_function(emb_model)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    if not db.get()["ids"]:
        print(" Chroma database is empty.")
        return "No documents found.", ""

    llm = OllamaLLM(model=model)

    # Step 1: Generate Multiple Queries
    generated_queries = generate_queries(query_text, llm, num_queries=3)
    print("Generated Queries:", generated_queries)

    # Step 2: Retrieve Results for Each Query
    all_results = []
    result_sets = []  # To store ranked lists of document IDs
    doc_metadata = {}  # To store metadata for each document

    for q in generated_queries:
        results = db.similarity_search_with_score(q, k=3)
        ranked_docs = []
        for doc, _score in results:
            ranked_docs.append(doc.metadata["id"])
            doc_metadata[doc.metadata["id"]] = doc  # Store metadata by document ID
        result_sets.append(ranked_docs)
        all_results.extend(results)

    # Step 3: Apply Reciprocal Rank Fusion
    fused_doc_ids = reciprocal_rank_fusion(result_sets)

    # Step 4: Fuse Retrieved Documents
    unique_documents = [doc_metadata[doc_id] for doc_id in fused_doc_ids]

    # Step 5: Generate Context and Response
    context_text = "\n\n---\n\n".join([doc.page_content for doc in unique_documents])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = llm.invoke(prompt)

    # Step 6: Rephrase the Response
    rephrase_prompt_template = ChatPromptTemplate.from_template(REPHRASE_PROMPT_TEMPLATE)
    rephrase_prompt = rephrase_prompt_template.format(text=response_text)
    rephrased_response_text = llm.invoke(rephrase_prompt)

    # Step 7: Format Sources
    sources = [
        {
            "id": doc.metadata.get("id", "No ID"),
            "title": doc.metadata.get("title", "No Title"),
            "author": doc.metadata.get("author", "No Author"),
            "source": doc.metadata.get("source", "No Source"),
            "page": doc.metadata.get("page", "Unknown Page"),
            "link": doc.metadata.get("link", "No Link")
        }
        for doc in unique_documents
    ]

    grouped_sources = defaultdict(lambda: {"id": "", "title": "", "author": "", "pages": set(), "link": ""})
    
    for source in sources:
        source_path = source["source"]
        grouped_sources[source_path]["id"] = source["id"]
        grouped_sources[source_path]["title"] = source["title"]
        grouped_sources[source_path]["author"] = source["author"]
        grouped_sources[source_path]["link"] = source["link"]
        grouped_sources[source_path]["pages"].add(str(source["page"]))

    formatted_sources = "\n".join([
        f"  - Title: {group['title']}\n"
        f"  - Author: {group['author']}\n"
        f"  - Pages: {', '.join(sorted(group['pages'], key=int))}\n"
        f"  - Link: {group['link']}\n"
        for source_path, group in grouped_sources.items()
    ])

    return rephrased_response_text, formatted_sources

'''

#___________________________________________________________________________________________________________________



def process_query(query_text, model, emb_model,fusion=False):
    response, sources = query_rag_nested(query_text, model=model, emb_model=emb_model, fusion=fusion)
    return response, sources

def create_response_and_sources(queries, process_query_func, model, emb_model, fusion=False):
    """
    Creates two dictionaries: one for responses and one for sources,
    using reference numbers as keys.

    Args:
        queries: A list of query strings.
        process_query_func: A function to process each query and return response and sources.
        model: The model to use for query processing.
        emb_model: The embedding model to use.

    Returns:
        A tuple containing:
            - responses_dict: A dictionary mapping reference numbers to response strings.
            - sources_dict: A dictionary mapping reference numbers to lists of source dictionaries.
    """
    responses_dict = {}
    sources_dict = {}
    ref_counter = 1

    for query_text in queries:
        response, sources = process_query_func(query_text, model, emb_model)
        responses_dict[f"[{ref_counter}]"] = response
        sources_dict[f"[{ref_counter}]"] = sources
        ref_counter += 1

    return responses_dict, sources_dict


def print_formatted_output(responses_dict, sources_dict):
    """
    Prints the responses and sources dictionaries in a concise, formatted style.
    Consolidates sources based on titles, combines pages (even if the source appears multiple times),
    and appends reference numbers to the responses.

    Args:
        responses_dict: A dictionary containing responses with keys as reference numbers.
        sources_dict: A dictionary containing sources with keys as reference numbers.
    """

    # Consolidate sources based on titles and combine pages
    new_sources_dict = {}
    source_counter = 1  # Counter for assigning unique reference numbers

    for ref, sources in sources_dict.items():
        source_list = sources.split("\n\n")  # Split individual sources by double newlines

        for source in source_list:
            if source.strip():  # Ensure the source is not empty
                # Extract relevant fields from the source
                lines = source.strip().split("\n")
                title_line = next((line for line in lines if "Title:" in line), None)
                pages_line = next((line for line in lines if "Pages:" in line), None)

                if title_line:
                    title = title_line.split(": ", 1)[1].strip()
                else:
                    title = "Unknown Title"

                if pages_line:
                    pages_str = pages_line.split(": ", 1)[1].strip()
                    pages = set(pages_str.split(", "))  # Use a set to combine pages
                else:
                    pages = set()

                # Check if this title already exists in the new_sources_dict
                if title not in new_sources_dict:
                    new_sources_dict[title] = {
                        "ref": f"[{source_counter}]",
                        "content": source.strip(),
                        "pages": pages,
                    }
                    source_counter += 1
                else:
                    # Combine pages if the title already exists
                    new_sources_dict[title]["pages"].update(pages)

    # Update the content of each source to reflect combined pages
    for title, details in new_sources_dict.items():
        incremented_pages = {str(int(page) + 1) if page.isdigit() else page for page in details["pages"]}
        combined_pages = ", ".join(sorted(incremented_pages, key=lambda x: int(x) if x.isdigit() else x))
        
        # Replace the original pages with the combined pages, handling cases where there may be no initial Pages line
        if "Pages:" in details["content"]:
            details["content"] = details["content"].replace(
                details["content"][details["content"].find("Pages:") : details["content"].find("\n", details["content"].find("Pages:"))],
                f"Pages: {combined_pages}"
            )
        else:
            details["content"] += f"\nPages: {combined_pages}" # If there wasn't a pages line, we add it


    
    #print("______________________________RESPONSE__________________________________________")
    
    for ref, response in responses_dict.items():
        response_references = []
    
        # Check which sources are referenced in the sources_dict
        source_list = sources_dict[ref].split("\n\n")
    
        for source in source_list:
            if source.strip():  # Ensure it's not empty
                lines = source.strip().split("\n")
                title_line = next((line for line in lines if "Title:" in line), None)

                if title_line:
                    title = title_line.split(": ", 1)[1].strip()
                    if title in new_sources_dict:
                        response_references.append(new_sources_dict[title]["ref"])
    
        # Append reference numbers to the response
        response_with_refs = f"{response} {' '.join(response_references)}"
        print(f"{response_with_refs}\n")

    #print("______________________________SOURCES__________________________________________")
    for title, details in new_sources_dict.items():
        print(f"Source {details['ref']}:")
        print(details["content"] + "\n")


def query_llm(query_text: str, model_name: str, template_type: str = 'general'):
    """
    Answers a query using only the language model (no RAG) and selects a prompt template based on the 'template_type'.

    Args:
        query_text: The query to answer.
        model_name: The name of the language model to use (e.g., "llama2").
        template_type:  'general' or 'diagnosis'.  Determines which prompt template to use.  Defaults to 'general'.

    Returns:
        The response text from the language model.
    """

    # Define prompt templates
    prompt_templates = {
    'general': "You are a helpful assistant. Answer the following question: {question}",
    'diagnosis': """
    You are a medical diagnosis assistant. Based on the following information, 
    state the information in natural language mentioning the influence of features:

    Legend of results:
    Sex: (0 = F, 1 = M)
    HighChol: High cholesterol (0 = no, 1 = yes)
    Smoker: (0 = no, 1 = yes)
    PhysActivity: Physical activity in last 30 days (0 = no, 1 = yes)
    Fruits: Consume fruits/veggies daily (0 = no, 1 = yes)
    HvyAlcoholConsump: (0 = no, 1 = yes)
    MentHlth: Days of poor mental health
    HighBP: High blood pressure (0 = no, 1 = yes)
    -----------

    Example output : Diagnosis Interpretation: Given the details you've provided, the chance of having diabetes appears to be relatively slim, 
    approximately 8.38 percent according to our analysis.
      However, it's worth noting that there are factors indicating a possible risk of diabetes. 
      The variables with the most significant impact on this risk include High Blood Pressure and High Cholesterol levels. 
      Being male also contributes, though to a lesser degree.

   On the positive side, several factors seem to reduce the likelihood of diabetes. 
   The elements with the greatest influence in this regard are a young age, a Normal Body Mass Index, not smoking, 
   limited alcohol consumption, recent engagement in physical activity, and daily consumption of fruits or vegetables. 
   Despite these favorable factors, it's crucial to remember that even with a low probability, regular check-ups and
    maintaining a healthy lifestyle are vital for addressing any potential health issues and preventing the development of diabetes.

    ----------
    
    Base your answer on these results: {question}
    """
    }   

    # Select prompt template based on template_type
    if template_type in prompt_templates:
        prompt_template_string = prompt_templates[template_type]
    else:
        prompt_template_string = prompt_templates['general']  # Default to general if invalid type

    prompt_template = ChatPromptTemplate.from_template(prompt_template_string)
    prompt = prompt_template.format(question=query_text)

    model_llm = OllamaLLM(model=model_name)
    response_text = model_llm.invoke(prompt)


    clean_prompt_template = """

    Rephrase the following text by evaluating the feature influence strength with natural language rather than numbers. Do not use numbers.
    Text: {text}

    """
    if template_type != 'general':
        clean_prompt_template = ChatPromptTemplate.from_template(clean_prompt_template)
        clean_prompt = clean_prompt_template.format(text=response_text)
        response_text = model_llm.invoke(clean_prompt)

    return response_text

def create_disease_prediction_string(shap_values: list, feature_names: list, prediction_probability: float, disease: str, input_features: dict):
    """
    Creates a formatted string summarizing disease prediction, dividing features into pro-diagnosis and anti-diagnosis.

    Args:
        shap_values: List of SHAP values representing feature impacts.
        feature_names: List of feature names corresponding to SHAP values.
        prediction_probability: The probability of the disease prediction.
        disease: The name of the disease being predicted.
        input_features: Dictionary of input features with their values.

    Returns:
        A formatted string summarizing the prediction and feature impacts.
    """
    # Step 1: Remove keys with None values from input_features
    filtered_features = {k: v for k, v in input_features.items() if v is not None}

    # Step 2: Separate features into pro-diagnosis and anti-diagnosis
    pro_diagnosis = []
    anti_diagnosis = []

    for feature, shap_value in zip(feature_names, shap_values):
        if feature in filtered_features:
            impact = f"{feature} (Value: {filtered_features[feature]}, Impact: {shap_value:.4f})"
            if shap_value > 0:
                pro_diagnosis.append((shap_value, impact))  # Store SHAP value for sorting
            else:
                anti_diagnosis.append((shap_value, impact))

    # Step 3: Sort pro-diagnosis descending and anti-diagnosis descending
    pro_diagnosis.sort(key=lambda x: x[0], reverse=True)  # Sort by SHAP value (descending)
    anti_diagnosis.sort(key=lambda x: x[0], reverse=False)  # Sort by SHAP value (descending)

    # Step 4: Format the contributions as strings
    pro_diagnosis_str = "\n".join([item[1] for item in pro_diagnosis])
    anti_diagnosis_str = "\n".join([item[1] for item in anti_diagnosis])

    # Step 5: Create the final output string
    result_string = (
        f"Disease Prediction: {disease}\n"
        f"Prediction Probability: {prediction_probability:.4f}\n\n"
        f"Pro-Diagnosis Features (supporting {disease}):\n{pro_diagnosis_str if pro_diagnosis_str else 'None'}\n\n"
        f"Anti-Diagnosis Features (against {disease}):\n{anti_diagnosis_str if anti_diagnosis_str else 'None'}"
    )

    return result_string


def create_queries_list(shap_values, feature_names, disease):
    """
    Creates a list of queries based on SHAP values and features.

    Args:
        shap_values (list): List of SHAP values.
        feature_names (list): List of feature names corresponding to SHAP values.
        disease (str): The disease being assessed.
        x (float): Threshold for positive influence.
        y (float): Threshold for negative influence.

    Returns:
        list: A list of queries regarding the influence of features on the risk of the disease.
    """
    x = 0.01
    y = -0.01

    queries = []

    for shap_value, feature_name in zip(shap_values, feature_names):
        if shap_value > x:
            queries.append(f'influence of {feature_name} on risk of {disease}')
        elif shap_value < y:
            queries.append(f'influence of {feature_name} on risk of {disease}')

    return queries



