# tree-splitter-llm

A tree builder + token-based leaf grouping from a JSON file, that can be used later on for LLM tasks.

## How does it work?

## Run an example
### Env initialization
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) beforehand and create the environment:
```bash
$ uv venv -p python3.12
$ uv sync
$ source .venv/bin/activate
```
### Put a file & and run main
Inside folder examples, you can find a [json example](./example/airline_routes_swagger.json).

Run `python main.py` to build a tree from that file and create grouping nodes based on a defined token limit (in the example is 1/3 of the total file length). Additionally a plot with some tree nodes is saved in the example folder. It should look like this:

![](./example/graph.png)

TODO:
- code refactoring & linting
- documentation & examples
- tests
- notify llama index + open issue amadeus OS