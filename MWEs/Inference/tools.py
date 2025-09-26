import json
import random

def router(function_name, function_arguments):
    if function_name == "random_number_generator":
        return random_number_generator(**function_arguments)
    else:
        return "Function not found"

# Load tools from JSON file
def load_tools():
    try:
        with open('tools.json', 'r') as file:
            tools = json.load(file)
        return tools
    except FileNotFoundError:
        print("⚠️  tools.json not found. Running without tools.")
        return []
    except json.JSONDecodeError:
        print("⚠️  Error parsing tools.json. Check JSON syntax.")
        return []


def random_number_generator(min: int, max: int):
    return random.randint(min, max) 