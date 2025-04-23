import json
import tree_splitter_llm.utils as ut
from anytree import RenderTree
from anytree.exporter import UniqueDotExporter

swagger_filename = "./example/airline_routes_swagger.json"

with open(swagger_filename, "r") as f:
    swagger = json.load(f)

tree = ut.build_tree(swagger)
total_tokens = tree.token_length
print(f"Total number of tokens: {total_tokens}")

parent_keys = list(swagger.keys())

# We set the token limit to 1/3 the full file
token_limit = total_tokens / 3
print(f"Token ratio {token_limit / total_tokens:.2}")

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
    print(
        f"{row.pre} {row.node.name}, grouped_nodes = {get_grouped_node_names(row.node.grouped_children_keys)}"
    )

def nodenamefunc(node):
    if node.grouped_children_keys:
        output = f"{node.name}\n\n tokens={node.token_length}\n num_groups={len(node.grouped_children_keys)}"
    else:
        output = f"{node.name}\n\n tokens={node.token_length}\n"
    return output 

UniqueDotExporter(tree,
                  nodenamefunc=nodenamefunc,
                  nodeattrfunc=lambda node: "shape=circle",
                  indent = 7,
                  maxlevel=4
).to_picture("example/graph.png")