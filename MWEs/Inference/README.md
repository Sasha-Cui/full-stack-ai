# LLM Inference

This folder is the most directly relevant one if your goal is to learn how LLM applications and simple agents are actually put together.

## Files

- `inference.ipynb`: main notebook.
- `tools.py`: small tool router used in the notebook.
- `tools.json`: OpenAI-style tool schema.
- `GEPA_utils.py`: dataset and metric helpers for the GEPA section.
- `.env.example`: template for API keys.
- `requirements.txt`: dependencies for the notebook and helper modules.

## Topics

- API-based LLM inference
- Model selection tradeoffs
- Tool calling
- MCP concepts
- Prompt engineering
- GEPA prompt optimization
- Context-window management

## Install

```bash
pip install -r requirements.txt
cp .env.example .env
```

Then fill in the keys you actually plan to use.

## Run

```bash
jupyter notebook inference.ipynb
```

## Validation

- `tools.py` and `GEPA_utils.py` compile cleanly.
- The notebook was reviewed for structure, but not executed end to end locally because it requires live API keys and network access.

## Important Notes

- Provider pricing and model availability change quickly. Check the official provider docs instead of relying on old notebook outputs.
- The GEPA section is more expensive and slower than the earlier notebook sections.
- MCP examples are introductory; the main value is understanding the interaction pattern, not memorizing one vendor workflow.

## Suggested Order

1. API basics
2. Tool calling
3. Prompt engineering
4. MCP section
5. GEPA section last

## References

- [OpenAI API docs](https://platform.openai.com/docs)
- [OpenRouter docs](https://openrouter.ai/docs)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [DSPy docs](https://dspy.ai/)
