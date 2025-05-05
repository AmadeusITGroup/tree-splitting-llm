import anytree as at
import json
import tiktoken

class CustomNodeClass(object):
    def __init__(
        self,
        token_length,
        child_key,
        grouped_children_keys=None,
        parent_list=None,
    ):
        # Number of tokens in the node
        self.token_length = token_length
        # Get the child key (if any)
        self.child_key = child_key

        # List of keys of the children that are grouped
        if grouped_children_keys is None:
            self.grouped_children_keys = []
        else:
            self.grouped_children_keys = grouped_children_keys

        if parent_list is None:
            self.parent_list = []
        else:
            self.parent_list = parent_list


class TreeAndCustomNodeClass(CustomNodeClass, at.AnyNode):
    def __init__(
        self,
        name,
        token_length,
        child_key,
        parent=None,
        children=None,
        grouped_children_keys=None,
        parent_list=None,
    ):
        super(TreeAndCustomNodeClass, self).__init__(
            token_length,
            child_key,
            grouped_children_keys,
            parent_list,
        )

        # Basic tree structure
        self.name = name
        self.parent = parent
        if children:
            self.children = children


def get_token_length(input: dict, enc: tiktoken.Encoding) -> int:
    encoded_info = enc.encode(json.dumps(input))
    return len(encoded_info)


# Aux function to get the nodes from the parent node
def get_nodes_from_parent(parent_node: TreeAndCustomNodeClass, key: str, value: dict | str, encoder: tiktoken.Encoding, parent_list: list) -> None:
    tokens = get_token_length(value, encoder)
    new_parent = TreeAndCustomNodeClass(
        name=key,
        token_length=tokens,
        child_key=value,
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

            get_nodes_from_parent(new_parent, k, v, encoder, current_parent)
    else:
        pass


# Entry point to build the tree. It will create the root node and then call the recursive function to create the rest of the tree
def build_tree(raw_json: dict, model: str) -> TreeAndCustomNodeClass:
    encoder = tiktoken.encoding_for_model(model)

    root_node = TreeAndCustomNodeClass(
        name="root", token_length=get_token_length(raw_json, encoder), child_key=raw_json
    )

    for k, v in raw_json.items():
        get_nodes_from_parent(root_node, k, v, encoder, None)

    return root_node


def grouping_nodes(node: TreeAndCustomNodeClass, max_tokens: int) -> None:
    """
    Logic to group child nodes.

    Inputs:
    - node: the parent node
    - max_tokens: the maximum number of tokens allowed in the node groups

    Algorithm is:

    1. Initialize the sum of tokens as 0 and the group nodes as an empty list
    2. For each child in the parent node: do a temp sum of tokens with the child token length
    3. If the tmp sum is less than the max tokens, then append the child to the group nodes and update the sum of tokens. And move to the next child.
    4. Elif the child token length is greater than the max tokens (heavy child), then
         a) Save the group nodes that already scanned (to avoid jumps)
         b) if the heavy child is a leaf node, panic (leaf node tokens > max tokens) --> need to increase the max token number
         c) else, recursively call the function for the heavy child
    5. If the tmp sum is greater than the max tokens (but with NO heavy child), then flush the group nodes and initialize the sum of tokens with current child token length and the group nodes with current child
    6. Finally, it could be that the loop over child nodes never reaches the max tokens, so we need to flush the group nodes at the end of the loop.

    """

    sum_token_children = 0
    grouped_nodes = []
    for child in node.children:
        tmp_sum_token = sum_token_children + child.token_length
        # Sum up nodes if it is less than max_tokens
        if tmp_sum_token <= max_tokens:
            sum_token_children += child.token_length
            grouped_nodes.append(child)
        # if child token length is greater than max_tokens, then go deeper
        elif child.token_length > max_tokens:
            # Need to flush here to avoid jumps
            if sum_token_children > 0:
                # print("Flushing first")
                node.grouped_children_keys.append(grouped_nodes)
                sum_token_children = 0
                grouped_nodes = []

            if child.is_leaf:
                raise at.TreeError("Leaf node = '{child.name}' is exceeding max_tokens = {max_tokens}. Increase parameter 'token-number'")
            else:
                grouping_nodes(child, max_tokens)
        # otherwise, flush the grouped nodes and start again
        else:
            node.grouped_children_keys.append(grouped_nodes)
            sum_token_children = child.token_length
            grouped_nodes = [child]

    if sum_token_children > 0:
        # Flushing without reaching sum token limit
        node.grouped_children_keys.append(grouped_nodes)
        sum_token_children = 0
        grouped_nodes = []

