import json
import tree_splitter_llm.utils as ut
from anytree import RenderTree
from anytree.exporter import UniqueDotExporter
import argparse

default_swagger_filename = "./example/airline_routes_swagger.json"
default_model = "gpt-4"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tree-splittler-llm",
        description="Build a tree algorith from a JSON file and group the nodes based on a token limit",
    )

    parser.add_argument("-f", "--filename", default=default_swagger_filename)
    parser.add_argument("-m", "--model", default=default_model)
    parser.add_argument("-t", "--token-number", type=int, default=0)
    parser.add_argument("--display-tree", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    with open(args.filename, "r") as f:
        swagger = json.load(f)

    tree = ut.build_tree(swagger, args.model)
    total_tokens = tree.token_length
    print(f"Total number of tokens: {total_tokens}")

    # If no token number given, it defaults to 1/3 total token length
    if args.token_number == 0:
        token_limit = total_tokens // 3
    else:
        token_limit = args.token_number
    print(f"Token limit set to {token_limit}")
    print(f"Token ratio chosen {token_limit / total_tokens:.2}")

    ut.grouping_nodes(tree, token_limit)

    if args.display_tree:

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

        UniqueDotExporter(
            tree,
            nodenamefunc=nodenamefunc,
            nodeattrfunc=lambda node: "shape=circle",
            indent=7,
            maxlevel=4,
        ).to_picture("output_tree.png")
