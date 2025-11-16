# Claude Code Proxy

> **Experimental Project** - Use Claude Code with OpenRouter models

A lightweight proxy server that translates Anthropic API requests to OpenRouter format, enabling you to use [Claude Code](https://claude.com/claude-code) with models available on [OpenRouter](https://openrouter.ai).

---

## What Does This Do?

This proxy sits between Claude Code and OpenRouter, allowing you to:

- Use kimi-k2 models trought **OpenRouter models** (other models not tested)
- Keep using the familiar Claude Code interface
- Support streaming, tool calls and thinking mode
- Log conversations for debugging (optional)

```
Claude Code → This Proxy → OpenRouter → OpenRouter style Response → Transform → Anthropic style response → Claude Code
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/nikolaev-fb/claude-code-proxy
cd claude-code-proxy
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your API Key

Copy the example configuration file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```bash
# Get your API key from https://openrouter.ai/keys
OPENROUTER_API_KEY=your-key-here
```

### 4. Run the Proxy

```bash
python proxy.py
```

You should see:

```
INFO:     Starting Anthropic-to-OpenRouter Proxy
INFO:     Host: 0.0.0.0
INFO:     Port: 3002
INFO:     Uvicorn running on http://0.0.0.0:3002
```

### 5. Configure Claude Code

Run the claude code with openrouter api endpoint and model

```bash
# Set the base URL to your proxy
ANTHROPIC_BASE_URL="http://localhost:3002" && \
ANTHROPIC_MODEL="moonshotai/kimi-k2-thinking" && \
ANTHROPIC_DEFAULT_HAIKU_MODEL="moonshotai/kimi-k2-0905" && \
ANTHROPIC_DEFAULT_SONNET_MODEL="moonshotai/kimi-k2-thinking" && \
CLAUDE_CODE_DISABLE_TELEMETRY="true" && \
claude
```

Now when you use Claude Code, requests will be routed through your proxy to OpenRouter!

---

## Configuration

All settings are configured in the `.env` file:

### Basic Settings

```bash
# Proxy server address and port
PROXY_HOST=0.0.0.0
PROXY_PORT=3002

# Your OpenRouter API key (required)
OPENROUTER_API_KEY=sk-or-v1-...
```


### File Logging (Optional)

**⚠ Warning: Log files can grow very quickly!**

```bash
# Enable logging
FILE_LOGGING_ENABLED=true
```

When enabled, all requests and responses are saved to `logs/messages.log`. This is useful for debugging but:

- **Remember to disable it after debugging** (`FILE_LOGGING_ENABLED=false`)
- Log files can become several MB in minutes with streaming responses
- Contains full conversation history (including system prompts)

Enabling logging allows you to log all opentrouter streaming responses into file `logs/opentrouter-streaming.log`
And all transformed anthropic responses into `logs/anthropic-streaming.log`

it is recommended to use langfuse to log all traces and enable file logging ony if you want to change the code of this proxy.

### LangFuse Integration

If you're running [LangFuse](https://langfuse.com) locally for observability:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_API_KEY=your-langfuse-key
LANGFUSE_SECRET_KEY=your-langfuse-secret
LANGFUSE_HOST=http://localhost:3000
```

---

## Features

- ✅ Streaming responses (real-time)
- ✅ Non-streaming responses
- ✅ Tool/function calling support
- ✅ Thinking mode support support
- ✅ Token usage tracking
- ✅ Error handling & logging
- ✅ LangFuse integration

---

## Known Limitations

- This is an **experimental project** - expect bugs!
- Only kimi-k2 models were tested 
- Some Anthropic features may not work perfectly with all OpenRouter models
- File uploads not yet supported
- Batch API not implemented
- Skills feature is not supported

---

## Support & Contributing

- **Issues**: Feel free to create new PR
- **OpenRouter Docs**: https://openrouter.ai/docs
- **Claude Code Docs**: https://docs.claude.com/claude-code

---

## License

MIT License - feel free to modify and use as needed

---
