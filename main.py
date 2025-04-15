import anytree as at
import os
from dotenv import load_dotenv
from pathlib import Path
import openai
import datetime
import json
import tiktoken
from typing import List, Tuple
import shutil
import glob


load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
TEMPERATURE = float(os.getenv("TEMPERATURE"))

enc = tiktoken.encoding_for_model(MODEL)

gpt4_client = openai.AzureOpenAI(
    api_version= API_VERSION,
    azure_endpoint= AZURE_ENDPOINT, 
    api_key= API_KEY)


def call_model(sys_prompt, usr_prompt):

    response = gpt4_client.chat.completions.create(
        temperature = TEMPERATURE,
        seed = 42,
        messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt}
        ],
        model=MODEL,
        )
        
    return response.choices[0].message.content

class MyBaseNodeClass(object):  # Just an example of a base class
    def __init__(self, token_length, swagger_child, needs_modification = None, llm_summary = None, grouped_children_keys = None, original_swagger = None, modified_swagger=None, parent_list = None):

        # Number of tokens in the node
        self.token_length = token_length
        # Swagger of the node child (if any)
        self.swagger_child = swagger_child

        # List of modifications done in the node / group of nodes
        if llm_summary is None:
            self.llm_summary = []
        else:
            self.llm_summary = llm_summary

        # Bool if the node needs modification / group of nodes
        if needs_modification is None:
            self.needs_modification = []
        else:
            self.needs_modification = needs_modification
        
        # List of keys of the children that are grouped
        if grouped_children_keys is None:
            self.grouped_children_keys = []
        else:
            self.grouped_children_keys = grouped_children_keys

        # List of original swagger of the node / group of nodes
        if original_swagger is None:
            self.original_swagger = []
        else:
            self.original_swagger = original_swagger

        # List of modified swagger of the node / group of nodes
        if modified_swagger is None:
            self.modified_swagger = []
        else:
            self.modified_swagger = modified_swagger

        if parent_list is None:
            self.parent_list = []
        else:
            self.parent_list = parent_list


class LLmNodeClass(MyBaseNodeClass, at.AnyNode):  # Add Node feature
    def __init__(self, name, token_length, swagger_child, parent=None, children=None, needs_modification = None, llm_summary = None, grouped_children_keys=None, original_swagger=None, modified_swagger=None, parent_list=None):
        super(LLmNodeClass, self).__init__(token_length, swagger_child, needs_modification, llm_summary, grouped_children_keys, original_swagger, modified_swagger, parent_list)

        # Basic tree structure
        self.name = name
        self.parent = parent
        if children:
            self.children = children

def get_token_length(input: dict) -> int:
    encoded_info = enc.encode(json.dumps(input))
    return len(encoded_info)

def get_token_length_string(input: str) -> int:
    encoded_info = enc.encode(input)
    return len(encoded_info)


# Aux function to get the nodes from the parent node
def get_nodes_from_parent(parent_node, key, value, parent_list):
    tokens = get_token_length(value)
    new_parent = LLmNodeClass(name = key, token_length=tokens, swagger_child=value, parent = parent_node, parent_list = parent_list)

    if isinstance(value, dict):
        for k,v in value.items():
            if not parent_list:
                current_parent = [key]
            else:
                current_parent = parent_list.copy()
                current_parent.append(key)

            get_nodes_from_parent(new_parent, k, v, current_parent)
    else:
        pass


# Entry point to build the tree. It will create the root node and then call the recursive function to create the rest of the tree
def build_tree(swagger: dict) -> LLmNodeClass:
    root_node = LLmNodeClass(name = "root", token_length = get_token_length(swagger), swagger_child=swagger)

    for k,v in swagger.items():
        get_nodes_from_parent(root_node, k, v, None)
        
    return root_node


def get_summary(nodes, high_level_structure, previous_summarization=""):

    if isinstance(nodes, list):
        swagger = {node.name: node.swagger_child for node in nodes}
        parent_list = nodes[0].parent_list
    else:
        swagger = {nodes.name: nodes.swagger_child}
        parent_list = nodes.parent_list

    #print(f"parent list: {parent_list}")
    if parent_list:
        for p in reversed(parent_list):
            swagger = {p:swagger}
    
    #print(swagger)

    p_sys = 'You are an expert in OpenAPI specification and in swagger files.'

    #high_level_structure = list(original_swagger_fle.keys())


    p_usr = f"""The swagger of your API has the structure with the following high level structure: {high_level_structure}
    Current source:
    {json.dumps(swagger, indent=4)}
    
    Summary of previous source text: {previous_summarization}
    
    Task: You need to read current source text and summary of previous source text (if any), and generate a summary to include them both. Do not miss important information from source text or past summary.

    Please make sure to include all necessary information such as data objects and example objects as well as references and correlations for swagger file modification objective. 

    Make sure to focus on detailed information of data objects and example objects. 

    If an object contains references '$ref', make sure to mention it and correlate with the reference. 

    """

    #print(p_usr)
    geni = call_model(p_sys, p_usr)
    #print(geni)
    #print('-------------')

    return geni


def grouping_nodes_v3(node: LLmNodeClass, max_tokens: int, past_summaries: str, parent_keys: list) -> None:
    """
    Logic to group child nodes.

    Inputs:
    - node: the parent node
    - max_tokens: the maximum tokens allowed to group nodes
    - user_query: the user query
    - past_output: the previous model output
    - summarizations: the summarization of the previous model calls.

    Algorithm is as follows:

    1. Initialize the sum of tokens as 0 and the group nodes as an empty list
    2. For each child in the parent node: do a temp sum of tokens with the child token length
    3. If the tmp sum is less than the max tokens, then append the child to the group nodes and update the sum of tokens. And move to the next child.
    4. Elif the child token length is greater than the max tokens (heavy child), then
         a) Call the model for the group nodes already scanned (to avoid jumps)
         b) if the heavy child is a leaf node, panic (leaf node tokens > max tokens) --> need to increase the max token number
         c) else, recursively call the function for the heavy child
    5. If the tmp sum is greater than the max tokens (but with NO heavy child), then flush the group nodes and initialize the sum of tokens with current child token length and the group nodes with current child
    6. Finally, it could be that the loop over child nodes never reaches the max tokens, so we need to flush the group nodes at the end of the loop.

    """
    sum_token_children = 0
    grouped_nodes = []
    #print(f"Here are the last summarizations: {summarizations}")
    for child in node.children:

        tmp_sum_token = sum_token_children + child.token_length
        # print(f"Processing child {child.name} with tokens={child.token_length}")
        # Sum up nodes if it is less than max_tokens
        if tmp_sum_token <= max_tokens:
            #print(f"Children: {child.name} with token length: {child.token_length} appended it")
            sum_token_children += child.token_length
            grouped_nodes.append(child)
        # if child token length is greater than max_tokens, then go deeper
        elif child.token_length > max_tokens:
            #print(f"Children: {child.name} with token length: {child.token_length} is big")
            # Need to flush here to avoid jumps
            if sum_token_children > 0:
                #print("Flushing first")
                past_summaries = get_summary(grouped_nodes, parent_keys, past_summaries)

                sum_token_children = 0
                grouped_nodes = []
                
            if child.is_leaf:
                print(f"Panic {child.name} is a leaf node exceeding max_tokens")
            else:
                #print(f"Recursive call for children {child.name}")
                grouping_nodes_v3(child, max_tokens, past_summaries, parent_keys)
        # otherwise, flush the grouped nodes and start again
        else:
            #print("Flushing since sum_token it is greater than max_tokens")
            past_summaries = get_summary(grouped_nodes, parent_keys, past_summaries)

            sum_token_children = child.token_length
            grouped_nodes = [child]
            
    if sum_token_children > 0:
            #print("Flushing without reaching sum token limit")
            past_summaries = get_summary(grouped_nodes, parent_keys, past_summaries)

            sum_token_children = 0
            grouped_nodes = []
    
    return past_summaries


swagger_filename = "./data/airline_routes.json"
with open(swagger_filename, "r") as f:
    swagger = json.load(f)

tree = build_tree(swagger)
total_tokens = tree.token_length
print(f"Total number of tokens: {total_tokens}")

parent_keys = list(swagger.keys())
token_limit = total_tokens / 3
print(f"Token ratio {token_limit/total_tokens:.2}")

final_summary = grouping_nodes_v3(tree, token_limit, past_summaries="", parent_keys=parent_keys)
summary_filename = f"./data/summary.txt"

with open(summary_filename, "w") as f:
    f.write(final_summary)