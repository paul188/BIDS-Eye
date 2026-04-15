import yaml

def print_task_tree(data, indent=0):
    # Only iterate if data is a dictionary
    if not isinstance(data, dict):
        return

    # Metadata keys to exclude from the tree
    exclude = {'label', 'standard_code', 'codes', 'synonyms', 'dataset_codes', 'description', 'raw'}

    for key, value in data.items():
        if key not in exclude:
            print("  " * indent + str(key))
            # Recursively check for nested categories
            print_task_tree(value, indent + 1)

# Load the YAML
with open("value_mappings.yaml", "r") as yaml_data:
    parsed_yaml = yaml.safe_load(yaml_data)

# Target the 'task' root
if 'task' in parsed_yaml:
    print("task")
    print_task_tree(parsed_yaml['task'], indent=1)