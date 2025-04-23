from dotenv import load_dotenv
import anytree as at
import os
import tiktoken
import json

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4-32k-0613")

enc = tiktoken.encoding_for_model(MODEL)


class MyBaseNodeClass(object):  # Just an example of a base class
    def __init__(
        self,
        token_length,
        swagger_child,
        grouped_children_keys=None,
        original_swagger=None,
        parent_list=None,
    ):
        # Number of tokens in the node
        self.token_length = token_length
        # Swagger of the node child (if any)
        self.swagger_child = swagger_child

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

        if parent_list is None:
            self.parent_list = []
        else:
            self.parent_list = parent_list


class LLmNodeClass(MyBaseNodeClass, at.AnyNode):  # Add Node feature
    def __init__(
        self,
        name,
        token_length,
        swagger_child,
        parent=None,
        children=None,
        grouped_children_keys=None,
        original_swagger=None,
        parent_list=None,
    ):
        super(LLmNodeClass, self).__init__(
            token_length,
            swagger_child,
            grouped_children_keys,
            original_swagger,
            parent_list,
        )

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
def get_nodes_from_parent(parent_node, key, value, parent_list) -> None:
    tokens = get_token_length(value)
    new_parent = LLmNodeClass(
        name=key,
        token_length=tokens,
        swagger_child=value,
        parent=parent_node,
        parent_list=parent_list,
    )

    if isinstance(value, dict):
        for k, v in value.items():
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
    root_node = LLmNodeClass(
        name="root", token_length=get_token_length(swagger), swagger_child=swagger
    )

    for k, v in swagger.items():
        get_nodes_from_parent(root_node, k, v, None)

    return root_node


def grouping_nodes(
    node: LLmNodeClass, max_tokens: int, past_summaries: str, parent_keys: list
) -> None:
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
    # print(f"Here are the last summarizations: {summarizations}")
    for child in node.children:
        tmp_sum_token = sum_token_children + child.token_length
        # print(f"Processing child {child.name} with tokens={child.token_length}")
        # Sum up nodes if it is less than max_tokens
        if tmp_sum_token <= max_tokens:
            # print(f"Children: {child.name} with token length: {child.token_length} appended it")
            sum_token_children += child.token_length
            grouped_nodes.append(child)
        # if child token length is greater than max_tokens, then go deeper
        elif child.token_length > max_tokens:
            # print(f"Children: {child.name} with token length: {child.token_length} is big")
            # Need to flush here to avoid jumps
            if sum_token_children > 0:
                # print("Flushing first")
                # past_summaries = get_summary(grouped_nodes, parent_keys, past_summaries)
                node.grouped_children_keys.append(grouped_nodes)
                sum_token_children = 0
                grouped_nodes = []

            if child.is_leaf:
                print(f"Panic {child.name} is a leaf node exceeding max_tokens")
            else:
                # print(f"Recursive call for children {child.name}")
                grouping_nodes(child, max_tokens, past_summaries, parent_keys)
        # otherwise, flush the grouped nodes and start again
        else:
            # print("Flushing since sum_token it is greater than max_tokens")
            # past_summaries = get_summary(grouped_nodes, parent_keys, past_summaries)
            node.grouped_children_keys.append(grouped_nodes)
            sum_token_children = child.token_length
            grouped_nodes = [child]

    if sum_token_children > 0:
        # print("Flushing without reaching sum token limit")
        # past_summaries = get_summary(grouped_nodes, parent_keys, past_summaries)
        node.grouped_children_keys.append(grouped_nodes)
        sum_token_children = 0
        grouped_nodes = []

    return None
