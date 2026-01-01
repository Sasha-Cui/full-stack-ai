# LLM Inference Tutorial

## Overview
This tutorial covers the practical aspects of LLM inference, deployment, and optimization. You'll learn how to use APIs for inference, understand model selection tradeoffs, implement tool calling, use Model Context Protocol (MCP), and optimize prompts with advanced techniques like GEPA.

## Topics Covered

### 1. **API-Based Inference**
- Using OpenRouter to access multiple LLM providers
- Understanding model selection criteria (cost, performance, latency)
- Comparing popular models (GPT-5, Claude Sonnet 4, DeepSeek, Gemini)
- Cost analysis and optimization

### 2. **Multi-Modal Inference**
- Text-to-text generation
- Vision-language models
- Understanding modalities and their use cases

### 3. **Model Selection & Tradeoffs**
- Pricing considerations ($/1M tokens)
- Context length requirements
- Supported parameters (temperature, reasoning, tools)
- Performance vs cost analysis

### 4. **Tool Calling**
- What are tools and why they matter
- Implementing custom tools
- Tool routing and execution
- Sequential tool calls

### 5. **Model Context Protocol (MCP)**
- Understanding MCP servers
- Integrating third-party services (Notion, Google Calendar)
- Standardized tool interfaces for LLMs

### 6. **Prompt Engineering**
- OpenAI Model Specs and chain of command
- System prompts vs user prompts
- Developer-level instructions
- Context window management

### 7. **Advanced Prompt Optimization (GEPA)**
- Gradient-free prompt optimization
- Automated prompt engineering
- Metric-driven optimization
- Evaluation and iteration

### 8. **Context Management**
- Understanding context limits
- Needle in a Haystack evaluation
- Context rot and degradation
- Conversation summarization strategies

## Prerequisites
- Python 3.8+
- Basic understanding of LLMs
- API keys (see Setup section)
- Internet connection for API calls

## Installation

### 1. Create Environment
```bash
conda create -n inference-tutorial python=3.10
conda activate inference-tutorial
```

### 2. Install Dependencies
```bash
pip install python-dotenv openai requests ipython jupyter
pip install dspy-ai datasets transformers accelerate
pip install matplotlib seaborn pandas numpy
```

### 3. Setup API Keys

**Option A: Using .env file (Recommended)**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# nano .env
```

**Option B: Export as environment variables**
```bash
export OPENROUTER_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
export NOTION_API_KEY="your_key_here"  # Optional
export GOOGLE_API_KEY="your_key_here"  # Optional
```

### Getting API Keys

#### OpenRouter (Required)
1. Visit https://openrouter.ai/keys
2. Sign up or log in
3. Click "Create Key"
4. Copy the key to your .env file
5. **Costs**: Pay-per-use, starts from $0 (some models are free)

#### OpenAI (Required)
1. Visit https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key to your .env file
5. **Costs**: Pay-per-use, requires credit card

#### Notion (Optional - for MCP section)
1. Visit https://www.notion.so/my-integrations
2. Click "+ New integration"
3. Give it a name and select workspace
4. Copy the Internal Integration Token

#### Google Calendar (Optional - for MCP section)
1. Visit https://console.cloud.google.com/
2. Create a new project or select existing
3. Enable Google Calendar API
4. Create credentials (API key)
5. Copy the API key

## File Structure
```
Inference/
├── inference.ipynb          # Main tutorial notebook
├── tools.py                 # Tool definitions and router
├── tools.json              # Tool schemas (OpenAI format)
├── GEPA_utils.py           # GEPA optimization utilities
├── README.md               # This file
├── .env.example            # Example environment variables
├── .env                    # Your actual API keys (gitignored)
└── images/                 # Tutorial figures
    ├── figure1.png
    ├── figure2.png
    ├── figure3.png
    ├── figure4.png
    ├── figure5.png
    └── figure6.png
```

## Running the Tutorial

### Jupyter Notebook
```bash
jupyter notebook inference.ipynb
```

### JupyterLab
```bash
jupyter lab inference.ipynb
```

### Important Notes
- **API Costs**: Running the full notebook will incur API costs (approximately $0.10-$0.50 depending on models)
- **Rate Limits**: OpenRouter and OpenAI have rate limits; the notebook includes appropriate delays
- **Internet Required**: All API calls require internet connectivity
- **Optional Sections**: MCP sections require additional API keys and can be skipped

## Tutorial Sections

### Section 1: OpenRouter Basics (Required)
- **Cost**: ~$0.01
- **Time**: 5 minutes
- **Prerequisites**: OPENROUTER_API_KEY

### Section 2: Model Comparison (Required)
- **Cost**: ~$0.05
- **Time**: 10 minutes
- **Prerequisites**: OPENROUTER_API_KEY

### Section 3: Tools & Function Calling (Required)
- **Cost**: ~$0.02
- **Time**: 15 minutes
- **Prerequisites**: OPENAI_API_KEY
- **Includes**: Custom tool implementation, tool routing

### Section 4: MCP Integration (Optional)
- **Cost**: ~$0.01
- **Time**: 10 minutes
- **Prerequisites**: OPENAI_API_KEY, GOOGLE_API_KEY or NOTION_API_KEY
- **Note**: Can be skipped if you don't have MCP API keys

### Section 5: Prompt Engineering (Required)
- **Cost**: ~$0.05
- **Time**: 10 minutes
- **Prerequisites**: OPENROUTER_API_KEY

### Section 6: GEPA Optimization (Advanced)
- **Cost**: ~$0.30 (due to multiple optimization iterations)
- **Time**: 30-60 minutes
- **Prerequisites**: OPENAI_API_KEY
- **Note**: Computationally intensive, uses GPT-4 for optimization

## Expected Total Cost
- **Minimum** (skipping GEPA): ~$0.10
- **Full tutorial**: ~$0.50
- **Multiple runs**: Scale accordingly

## Common Issues & Solutions

### Issue: `KeyError: 'OPENROUTER_API_KEY'`
**Solution**: Ensure you've created a `.env` file with your API keys or exported environment variables.

### Issue: `401 Unauthorized`
**Solution**: Check that your API keys are valid and not expired. Verify you've added credits to your OpenRouter/OpenAI account.

### Issue: `RateLimitError`
**Solution**: Wait a few seconds and retry. Consider adding delays between API calls or using lower rate-limit models.

### Issue: `InvalidRequestError: model not found`
**Solution**: Some models may not be available in all regions or may be deprecated. Check OpenRouter's model list: https://openrouter.ai/models

### Issue: GEPA optimization is slow
**Solution**: This is expected. GEPA makes many API calls to optimize prompts. Consider reducing the number of training examples or using a smaller dataset.

### Issue: `ImportError: No module named 'dspy'`
**Solution**: Install DSPy: `pip install dspy-ai`

### Issue: MCP server not connecting
**Solution**: Verify your API keys for Notion/Google Calendar are correct and have the necessary permissions.

## Learning Path

### Beginner
1. Start with Section 1 (OpenRouter Basics)
2. Explore Section 2 (Model Comparison)
3. Learn Section 3 (Tools & Function Calling)
4. Review Section 5 (Prompt Engineering)

### Intermediate
1. Complete all beginner sections
2. Experiment with Section 4 (MCP Integration)
3. Try Section 6 (GEPA) with small datasets

### Advanced
1. Modify GEPA optimization metrics
2. Implement custom MCP servers
3. Build multi-step reasoning workflows with tools
4. Experiment with routing strategies

## Key Takeaways

1. **Model Selection Matters**: Different models have different strengths; choose based on your specific needs
2. **Cost Optimization**: Free models (DeepSeek) can be competitive with paid models for many tasks
3. **Tools Extend Capabilities**: Tool calling makes LLMs much more powerful and reliable
4. **Prompt Engineering is Critical**: Small changes in prompts can lead to large performance improvements
5. **Context Management**: Understanding context limits and degradation is crucial for production systems

## Additional Resources
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Model Context Protocol Spec](https://modelcontextprotocol.io/)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [GEPA Paper](https://arxiv.org/abs/2507.19457)

## Contributing
Found an issue or want to improve this tutorial? Please open an issue or pull request in the main repository.

## License
This tutorial is part of the Full-Stack AI working group materials at Yale University.

## Acknowledgments
Developed for the "Becoming Full-Stack AI Researchers" working group at Yale University, supported by the Wu Tsai Institute.

## Citation
If you use these materials in your research or teaching, please cite:
```
@misc{fullstackai2025,
  title={Becoming Full-Stack AI Researchers: Inference Tutorial},
  author={Cui, Sasha and Le, Quan and Mader, Alexander and Sanok Dufallo, Will},
  year={2025},
  institution={Yale University}
}
```

