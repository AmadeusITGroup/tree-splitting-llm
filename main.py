import json
import tree_splitting_llm.utils as ut
from anytree import RenderTree

swagger_filename = "./data/airline_routes.json"

with open(swagger_filename, "r") as f:
    swagger = json.load(f)

tree = ut.build_tree(swagger)
total_tokens = tree.token_length
print(f"Total number of tokens: {total_tokens}")

parent_keys = list(swagger.keys())
token_limit = total_tokens / 3
print(f"Token ratio {token_limit/total_tokens:.2}")

ut.grouping_nodes(tree, token_limit, past_summaries="", parent_keys=parent_keys)

def get_grouped_node_names(tree_array):
    groups = []
    for i in tree_array:
        nodes = []
        for j in i:
            nodes.append(j.name)
        groups.append(nodes)
    return groups

for row in RenderTree(tree):
    print(f"{row.pre} {row.node.name}, grouped_nodes = {get_grouped_node_names(row.node.grouped_children_keys)}")
