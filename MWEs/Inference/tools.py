from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable


def random_number_generator(min: int, max: int) -> int:
    if int(min) != min or int(max) != max:
        raise ValueError("random_number_generator expects integer bounds.")
    if min > max:
        raise ValueError("'min' must be less than or equal to 'max'.")
    return random.randint(int(min), int(max))


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "random_number_generator": random_number_generator,
}


def router(function_name: str, function_arguments: dict[str, Any]) -> Any:
    try:
        tool = TOOL_REGISTRY[function_name]
    except KeyError as exc:
        raise ValueError(f"Unknown tool: {function_name}") from exc
    return tool(**function_arguments)


def load_tools() -> list[dict[str, Any]]:
    tools_path = Path(__file__).with_name("tools.json")
    try:
        with tools_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Warning: tools.json not found. Running without tools.")
        return []
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {tools_path}") from exc
