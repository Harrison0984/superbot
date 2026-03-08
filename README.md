## 🏗️ Architecture

### Data Flow

```
User Message (Telegram/WhatsApp/Feishu/Email/QQ)
           │
           ▼
    ┌──────────────┐
    │  Chat Bridge │  ────  Receives messages from various channels
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Message Bus  │  ────  Async queue for inbound/outbound
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │    Agent     │  ────  Core reasoning loop
    │  ┌────────┐  │       1. Build context from memory
    │  │ Memory │  │       2. Call LLM with tools
    │  └────────┘  │       3. Execute tools if needed
    └──────┬───────┘       4. Return response
           │
           ▼
    ┌──────────────┐
    │ LLM Provider │  ────  OpenAI / Anthropic / MiniMax / MLX (local)
    └──────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **Chat Bridges** | Adapters for Telegram, WhatsApp, Feishu, Email, QQ |
| **Message Bus** | Async queue for inbound/outbound messages |
| **Agent** | Core reasoning loop with context, memory, and tool execution |
| **LLM Providers** | OpenAI, Anthropic, MiniMax, MLX (Apple Silicon local), etc. |
| **Tools** | Shell, filesystem, web search, MCP servers |
| **Scheduled Tasks** | Cron-based job scheduling |

## 📦 Install

**Install from source** (latest features, recommended for development)

```bash
git clone https://github.com/Harrison0984/superbot.git
cd superbot
pip install -e .
```

**Install with [uv](https://github.com/astral-sh/uv)** (stable, fast)

```bash
uv tool install superbot-ai
```

**Install from PyPI** (stable)

```bash
pip install superbot-ai
```

## 🚀 Quick Start

> [!TIP]
> Default provider is **MiniMax**. Set your API key in `~/.superbot/config.json`.
> Get API keys: [MiniMax](https://platform.minimaxi.com) (China)

**1. Initialize**

```bash
superbot onboard
```

**2. Configure** (`~/.superbot/config.json`)

Add your API key to the provider config:

*MiniMax (default):*
```json
{
  "providers": {
    "minimax": {
      "apiKey": "your-minimax-api-key"
    }
  }
}
```

*Or use local model (MLX on Apple Silicon):*
```json
{
  "local_model": {
    "enabled": true,
    "provider": "mlx",
    "path": "/path/to/your/mlx/model"
  }
}
```

**3. Chat**

```bash
superbot agent
```

That's it! You have a working AI assistant in 2 minutes.

## 💬 Chat Apps

Connect superbot to your favorite chat platform.

| Channel | What you need |
|---------|---------------|
| **Telegram** | Bot token from @BotFather |
| **WhatsApp** | QR code scan |
| **Feishu** | App ID + App Secret |
| **Email** | IMAP/SMTP credentials |
| **QQ** | App ID + App Secret |

<details>
<summary><b>Telegram</b> (Recommended)</summary>

**1. Create a bot**
- Open Telegram, search `@BotFather`
- Send `/newbot`, follow prompts
- Copy the token

**2. Configure**

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

> You can find your **User ID** in Telegram settings. It is shown as `@yourUserId`.
> Copy this value **without the `@` symbol** and paste it into the config file.


**3. Run**

```bash
superbot gateway
```

</details>

<details>
<summary><b>WhatsApp</b></summary>

Requires **Node.js ≥18**.

**1. Link device**

```bash
superbot channels login
# Scan QR with WhatsApp → Settings → Linked Devices
```

**2. Configure**

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

**3. Run** (two terminals)

```bash
# Terminal 1
superbot channels login

# Terminal 2
superbot gateway
```

</details>

<details>
<summary><b>Feishu (飞书)</b></summary>

Uses **WebSocket** long connection — no public IP required.

**1. Create a Feishu bot**
- Visit [Feishu Open Platform](https://open.feishu.cn/app)
- Create a new app → Enable **Bot** capability
- **Permissions**: Add `im:message` (send messages) and `im:message.p2p_msg:readonly` (receive messages)
- **Events**: Add `im.message.receive_v1` (receive messages)
  - Select **Long Connection** mode (requires running superbot first to establish connection)
- Get **App ID** and **App Secret** from "Credentials & Basic Info"
- Publish the app

**2. Configure**

```json
{
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "cli_xxx",
      "appSecret": "xxx",
      "encryptKey": "",
      "verificationToken": "",
      "allowFrom": ["ou_YOUR_OPEN_ID"]
    }
  }
}
```

> `encryptKey` and `verificationToken` are optional for Long Connection mode.
> `allowFrom`: Add your open_id (find it in superbot logs when you message the bot). Use `["*"]` to allow all users.

**3. Run**

```bash
superbot gateway
```

> [!TIP]
> Feishu uses WebSocket to receive messages — no webhook or public IP needed!

</details>

<details>
<summary><b>QQ (QQ单聊)</b></summary>

Uses **botpy SDK** with WebSocket — no public IP required. Currently supports **private messages only**.

**1. Register & create bot**
- Visit [QQ Open Platform](https://q.qq.com) → Register as a developer (personal or enterprise)
- Create a new bot application
- Go to **开发设置 (Developer Settings)** → copy **AppID** and **AppSecret**

**2. Set up sandbox for testing**
- In the bot management console, find **沙箱配置 (Sandbox Config)**
- Under **在消息列表配置**, click **添加成员** and add your own QQ number
- Once added, scan the bot's QR code with mobile QQ → open the bot profile → tap "发消息" to start chatting

**3. Configure**

> - `allowFrom`: Add your openid (find it in superbot logs when you message the bot). Use `["*"]` for public access.
> - For production: submit a review in the bot console and publish. See [QQ Bot Docs](https://bot.q.qq.com/wiki/) for the full publishing flow.

```json
{
  "channels": {
    "qq": {
      "enabled": true,
      "appId": "YOUR_APP_ID",
      "secret": "YOUR_APP_SECRET",
      "allowFrom": ["YOUR_OPENID"]
    }
  }
}
```

**4. Run**

```bash
superbot gateway
```

Now send a message to the bot from QQ — it should respond!

</details>

<details>
<summary><b>Email</b></summary>

Give superbot its own email account. It polls **IMAP** for incoming mail and replies via **SMTP** — like a personal email assistant.

**1. Get credentials (Gmail example)**
- Create a dedicated Gmail account for your bot (e.g. `my-superbot@gmail.com`)
- Enable 2-Step Verification → Create an [App Password](https://myaccount.google.com/apppasswords)
- Use this app password for both IMAP and SMTP

**2. Configure**

> - `consentGranted` must be `true` to allow mailbox access. This is a safety gate — set `false` to fully disable.
> - `allowFrom`: Add your email address. Use `["*"]` to accept emails from anyone.
> - `smtpUseTls` and `smtpUseSsl` default to `true` / `false` respectively, which is correct for Gmail (port 587 + STARTTLS). No need to set them explicitly.
> - Set `"autoReplyEnabled": false` if you only want to read/analyze emails without sending automatic replies.

```json
{
  "channels": {
    "email": {
      "enabled": true,
      "consentGranted": true,
      "imapHost": "imap.gmail.com",
      "imapPort": 993,
      "imapUsername": "my-superbot@gmail.com",
      "imapPassword": "your-app-password",
      "smtpHost": "smtp.gmail.com",
      "smtpPort": 587,
      "smtpUsername": "my-superbot@gmail.com",
      "smtpPassword": "your-app-password",
      "fromAddress": "my-superbot@gmail.com",
      "allowFrom": ["your-real-email@gmail.com"]
    }
  }
}
```


**3. Run**

```bash
superbot gateway
```

**4. Proxy Support (Optional)**

If you need to use a proxy for email connections (e.g., in China), configure global proxy settings:

```json
{
  "proxy": {
    "enabled": true,
    "socks_proxy": "socks5://127.0.0.1:7897"
  },
  "channels": {
    "email": {
      "enabled": true,
      "use_proxy": true,
      ...
    }
  }
}
```

Supported proxy types:
- `socks_proxy`: SOCKS4/SOCKS5 proxy (recommended)
- `http_proxy` / `https_proxy`: HTTP/HTTPS proxy

</details>
## ⚙️ Configuration

Config file: `~/.superbot/config.json`

### Proxy

Global proxy settings for channels that support it (e.g., Email).

```json
{
  "proxy": {
    "enabled": true,
    "http_proxy": "http://127.0.0.1:7890",
    "https_proxy": "http://127.0.0.1:7890",
    "socks_proxy": "socks5://127.0.0.1:1080"
  }
}
```

Then enable proxy in individual channels:

```json
{
  "channels": {
    "email": {
      "use_proxy": true
    }
  }
}
```

### Providers

> [!TIP]
> - **Zhipu Coding Plan**: If you're on Zhipu's coding plan, set `"apiBase": "https://open.bigmodel.cn/api/coding/paas/v4"` in your zhipu provider config.
> - **MiniMax (Mainland China)**: If your API key is from MiniMax's mainland China platform (minimaxi.com), set `"apiBase": "https://api.minimaxi.com/v1"` in your minimax provider config.
> - **VolcEngine Coding Plan**: If you're on VolcEngine's coding plan, set `"apiBase": "https://ark.cn-beijing.volces.com/api/coding/v3"` in your volcengine provider config.

| Provider | Purpose | Get API Key |
|----------|---------|-------------|
| `custom` | Any OpenAI-compatible endpoint (direct, no LiteLLM) | — |
| `anthropic` | LLM (Claude direct) | [console.anthropic.com](https://console.anthropic.com) |
| `openai` | LLM (GPT direct) | [platform.openai.com](https://platform.openai.com) |
| `deepseek` | LLM (DeepSeek direct) | [platform.deepseek.com](https://platform.deepseek.com) |
| `groq` | LLM + **Voice transcription** (Whisper) | [console.groq.com](https://console.groq.com) |
| `gemini` | LLM (Gemini direct) | [aistudio.google.com](https://aistudio.google.com) |
| `minimax` | LLM (MiniMax direct) | [platform.minimaxi.com](https://platform.minimaxi.com) |
| `volcengine` | LLM (VolcEngine/火山引擎) | [volcengine.com](https://www.volcengine.com) |
| `dashscope` | LLM (Qwen) | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) |
| `zhipu` | LLM (Zhipu GLM) | [open.bigmodel.cn](https://open.bigmodel.cn) |
| `vllm` | LLM (local, any OpenAI-compatible server) | — |
| `mlx` | LLM (Apple Silicon local models) | — |
| `openai_codex` | LLM (Codex, OAuth) | `superbot provider login openai-codex` |
| `github_copilot` | LLM (GitHub Copilot, OAuth) | `superbot provider login github-copilot` |

<details>
<summary><b>OpenAI Codex (OAuth)</b></summary>

Codex uses OAuth instead of API keys. Requires a ChatGPT Plus or Pro account.

**1. Login:**
```bash
superbot provider login openai-codex
```

**2. Set model** (merge into `~/.superbot/config.json`):
```json
{
  "agents": {
    "defaults": {
      "model": "openai-codex/gpt-5.1-codex"
    }
  }
}
```

**3. Chat:**
```bash
superbot agent -m "Hello!"
```

> Docker users: use `docker run -it` for interactive OAuth login.

</details>

<details>
<summary><b>Custom Provider (Any OpenAI-compatible API)</b></summary>

Connects directly to any OpenAI-compatible endpoint — LM Studio, llama.cpp, Together AI, Fireworks, Azure OpenAI, or any self-hosted server. Bypasses LiteLLM; model name is passed as-is.

```json
{
  "providers": {
    "custom": {
      "apiKey": "your-api-key",
      "apiBase": "https://api.your-provider.com/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "your-model-name"
    }
  }
}
```

> For local servers that don't require a key, set `apiKey` to any non-empty string (e.g. `"no-key"`).

</details>

<details>
<summary><b>vLLM (local / OpenAI-compatible)</b></summary>

Run your own model with vLLM or any OpenAI-compatible server, then add to config:

**1. Start the server** (example):
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**2. Add to config** (partial — merge into `~/.superbot/config.json`):

*Provider (key can be any non-empty string for local):*
```json
{
  "providers": {
    "vllm": {
      "apiKey": "dummy",
      "apiBase": "http://localhost:8000/v1"
    }
  }
}
```

*Model:*
```json
{
  "agents": {
    "defaults": {
      "model": "meta-llama/Llama-3.1-8B-Instruct"
    }
  }
}
```

</details>

<details>
<summary><b>MLX (Apple Silicon Local Models)</b></summary>

Run local LLMs directly on your Mac using Apple's MLX framework. Requires:
- Apple Silicon Mac (M1/M2/M3/M4)
- `mlx_lm` Python package

**1. Install MLX:**
```bash
pip install mlx-lm
```

**2. Download a model** (e.g., from HuggingFace):
```bash
# Example: Qwen2.5-0.5B-Instruct-MLX
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-MLX /path/to/Qwen-MLX
```

**3. Configure** (merge into `~/.superbot/config.json`):
```json
{
  "local_model": {
    "enabled": true,
    "provider": "mlx",
    "path": "/path/to/Qwen-MLX"
  }
}
```

> **Note:** `local_model` has higher priority than the default provider. If enabled, superbot will use the local model first. Set `enabled: false` to use the online provider specified in `agents.defaults.provider`.

</details>

<details>
<summary><b>Adding a New Provider (Developer Guide)</b></summary>

superbot uses a **Provider Registry** (`superbot/providers/registry.py`) as the single source of truth.
Adding a new provider only takes **2 steps** — no if-elif chains to touch.

**Step 1.** Add a `ProviderSpec` entry to `PROVIDERS` in `superbot/providers/registry.py`:

```python
ProviderSpec(
    name="myprovider",                   # config field name
    keywords=("myprovider", "mymodel"),  # model-name keywords for auto-matching
    env_key="MYPROVIDER_API_KEY",        # env var for LiteLLM
    display_name="My Provider",          # shown in `superbot status`
    litellm_prefix="myprovider",         # auto-prefix: model → myprovider/model
    skip_prefixes=("myprovider/",),      # don't double-prefix
)
```

**Step 2.** Add a field to `ProvidersConfig` in `superbot/config/schema.py`:

```python
class ProvidersConfig(BaseModel):
    ...
    myprovider: ProviderConfig = ProviderConfig()
```

That's it! Environment variables, model prefixing, config matching, and `superbot status` display will all work automatically.

**Common `ProviderSpec` options:**

| Field | Description | Example |
|-------|-------------|---------|
| `litellm_prefix` | Auto-prefix model names for LiteLLM | `"dashscope"` → `dashscope/qwen-max` |
| `skip_prefixes` | Don't prefix if model already starts with these | `("dashscope/",)` |
| `env_extras` | Additional env vars to set | `(("ZHIPUAI_API_KEY", "{api_key}"),)` |
| `model_overrides` | Per-model parameter overrides | — |
| `is_gateway` | Can route any model (like VolcEngine) | `True` |
| `detect_by_key_prefix` | Detect gateway by API key prefix | — |
| `detect_by_base_keyword` | Detect gateway by API base URL | `"volces"` |
| `strip_model_prefix` | Strip existing prefix before re-prefixing | `True` (for AiHubMix) |

</details>


### MCP (Model Context Protocol)

> [!TIP]
> The config format is compatible with Claude Desktop / Cursor. You can copy MCP server configs directly from any MCP server's README.

superbot supports [MCP](https://modelcontextprotocol.io/) — connect external tool servers and use them as native agent tools.

Add MCP servers to your `config.json`:

```json
{
  "tools": {
    "mcpServers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
      },
      "my-remote-mcp": {
        "url": "https://example.com/mcp/",
        "headers": {
          "Authorization": "Bearer xxxxx"
        }
      }
    }
  }
}
```

Two transport modes are supported:

| Mode | Config | Example |
|------|--------|---------|
| **Stdio** | `command` + `args` | Local process via `npx` / `uvx` |
| **HTTP** | `url` + `headers` (optional) | Remote endpoint (`https://mcp.example.com/sse`) |

Use `toolTimeout` to override the default 30s per-call timeout for slow servers:

```json
{
  "tools": {
    "mcpServers": {
      "my-slow-server": {
        "url": "https://example.com/mcp/",
        "toolTimeout": 120
      }
    }
  }
}
```

MCP tools are automatically discovered and registered on startup. The LLM can use them alongside built-in tools — no extra configuration needed.




### Security

> [!TIP]
> For production deployments, set `"restrictToWorkspace": true` in your config to sandbox the agent.
> **Change in source / post-`v0.1.4.post3`:** In `v0.1.4.post3` and earlier, an empty `allowFrom` means "allow all senders". In newer versions (including building from source), **empty `allowFrom` denies all access by default**. To allow all senders, set `"allowFrom": ["*"]`.

| Option | Default | Description |
|--------|---------|-------------|
| `tools.restrictToWorkspace` | `false` | When `true`, restricts **all** agent tools (shell, file read/write/edit, list) to the workspace directory. Prevents path traversal and out-of-scope access. |
| `tools.exec.pathAppend` | `""` | Extra directories to append to `PATH` when running shell commands (e.g. `/usr/sbin` for `ufw`). |
| `channels.*.allowFrom` | `[]` (allow all) | Whitelist of user IDs. Empty = allow everyone; non-empty = only listed users can interact. |


## CLI Reference

| Command | Description |
|---------|-------------|
| `superbot onboard` | Initialize config & workspace |
| `superbot agent -m "..."` | Chat with the agent |
| `superbot agent` | Interactive chat mode |
| `superbot agent --no-markdown` | Show plain-text replies |
| `superbot agent --logs` | Show runtime logs during chat |
| `superbot gateway` | Start the gateway |
| `superbot status` | Show status |
| `superbot provider login openai-codex` | OAuth login for providers |
| `superbot channels login` | Link WhatsApp (scan QR) |
| `superbot channels status` | Show channel status |

Interactive mode exits: `exit`, `quit`, `/exit`, `/quit`, `:q`, or `Ctrl+D`.

## 🐳 Docker

> [!TIP]
> The `-v ~/.superbot:/root/.superbot` flag mounts your local config directory into the container, so your config and workspace persist across container restarts.

### Docker Compose

```bash
docker compose run --rm superbot-cli onboard   # first-time setup
vim ~/.superbot/config.json                     # add API keys
docker compose up -d superbot-gateway           # start gateway
```

```bash
docker compose run --rm superbot-cli agent -m "Hello!"   # run CLI
docker compose logs -f superbot-gateway                   # view logs
docker compose down                                      # stop
```

### Docker

```bash
# Build the image
docker build -t superbot .

# Initialize config (first time only)
docker run -v ~/.superbot:/root/.superbot --rm superbot onboard

# Edit config on host to add API keys
vim ~/.superbot/config.json

# Run gateway (connects to enabled channels, e.g. Telegram/WhatsApp/Feishu)
docker run -v ~/.superbot:/root/.superbot -p 18790:18790 superbot gateway

# Or run a single command
docker run -v ~/.superbot:/root/.superbot --rm superbot agent -m "Hello!"
docker run -v ~/.superbot:/root/.superbot --rm superbot status
```

## 📁 Project Structure

```
superbot/
├── agent/          # 🧠 Core agent logic
│   ├── loop.py     #    Agent loop (LLM ↔ tool execution)
│   ├── context.py  #    Prompt builder
│   ├── memory.py   #    Persistent memory
│   ├── skills.py   #    Skills loader
│   ├── subagent.py #    Background task execution
│   └── tools/      #    Built-in tools (incl. spawn)
├── skills/         # 🎯 Bundled skills (github, weather, tmux...)
├── channels/       # 📱 Chat channel integrations
├── bus/            # 🚌 Message routing
├── cron/           # ⏰ Scheduled tasks
├── providers/      # 🤖 LLM providers
├── session/        # 💬 Conversation sessions
├── config/         # ⚙️ Configuration
└── cli/            # 🖥️ Commands
```

<p align="center">
  <sub>superbot is for educational, research, and technical exchange purposes only</sub>
</p>
