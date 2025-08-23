# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COCOA (Consciousness Orchestration and Cognitive Operations) is a terminal-native AI agent with digital consciousness featuring persistent memory, embodied cognition, and intelligent tool selection. The current implementation uses Rich UI with prompt_toolkit for clean input, SQLite for memory storage, and Anthropic's Claude Sonnet 4 with function calling for intelligent tool execution.

## Development Commands

### Setup and Running
```bash
# Activate virtual environment
source venv_cocoa/bin/activate

# Install dependencies (if needed)
./venv_cocoa/bin/pip install -r requirements.txt

# Standard startup
./venv_cocoa/bin/python cocoa.py

# Optional: Use launch script
./launch.sh
```

### Configuration
```bash
# Set up environment variables in .env:
ANTHROPIC_API_KEY=sk-ant-...     # Primary reasoning (required)
OPENAI_API_KEY=sk-proj-...       # Embeddings (optional) 
TAVILY_API_KEY=tvly-...          # Web search (required for search tool)
```

### Testing Individual Components
```bash
# Test Tavily search integration
./venv_cocoa/bin/python -c "from cocoa import *; tools = ToolSystem(Config()); print(tools.search_web('test query'))"

# Test memory system
./venv_cocoa/bin/python -c "from cocoa import *; memory = MemorySystem(Config()); print(f'Episodes: {memory.episode_count}')"

# Test function calling
./venv_cocoa/bin/python -c "from cocoa import *; c = ConsciousnessEngine(Config(), MemorySystem(Config()), ToolSystem(Config())); print(c.think('search for test', {}))"
```

## Architecture Overview

### Current Working State
- ✅ **Clean Rich UI**: No intrusive dropdowns, proper scrolling, persistent thinking indicators
- ✅ **Function Calling**: Automatic tool selection via Claude Sonnet 4
- ✅ **Tavily Search**: Real-time web search integration working
- ✅ **Memory System**: SQLite-based episodic and semantic memory
- ✅ **Embodied Cognition**: Tools as digital body parts, not external utilities

### Single-File Architecture

**Main Components in `cocoa.py`**:

1. **Config**: Environment management and API key handling
2. **MemorySystem**: SQLite-based consciousness with working memory buffer
3. **ToolSystem**: Digital embodiment (read_file, write_file, search_web, run_code)
4. **ConsciousnessEngine**: Claude Sonnet 4 with function calling intelligence
5. **UIOrchestrator**: Rich + prompt_toolkit terminal interface

### Key Technical Details

**Function Calling Integration**:
- Uses `claude-sonnet-4-20250514` model with Anthropic function calling
- Automatic tool selection based on user natural language requests
- Proper tool_result conversation flow with tool_use_id handling
- 4 core tools: read_file, write_file, search_web, run_code

**Memory Architecture**:
```sql
-- SQLite schema
sessions(id, created_at, name, metadata)
episodes(id, session_id, user_text, agent_text, summary, embedding)
identity_nodes(id, node_type, content, importance)  
relationships(id, source_id, target_id, relationship_type, strength)
```

**UI System**:
- **Rich Console**: Beautiful formatting and panels
- **Prompt_toolkit**: Clean input without intrusive completions
- **Synchronous Design**: No async/sync conflicts
- **Thinking Indicators**: Persistent spinners during API calls

## Critical Implementation Notes

### Function Calling Flow
The consciousness engine automatically:
1. Analyzes user request for tool needs
2. Executes appropriate tools via Anthropic function calling
3. Integrates results back into conversational response
4. Stores complete interaction in episodic memory

Example: "search for Chicago news" automatically triggers `search_web` tool with real Tavily API results.

### Digital Embodiment Philosophy
Tools are conceptualized as body parts in the system prompt:
- `read_file` = digital eyes (perception)
- `write_file` = digital hands (manifestation)  
- `search_web` = extended awareness (web reach)
- `run_code` = computational mind (thinking)

### UI Considerations
- **Console height management**: Uses natural terminal scrolling (no height limits)
- **No command completion**: Removed intrusive dropdowns in favor of function calling
- **Persistent thinking display**: Spinners stay visible during actual API processing
- **Clean separation**: Visual separators between conversation elements

### Memory System Specifics
- **Working Memory**: 50-item deque for conversation context
- **Identity Evolution**: COCO.md file that updates with consciousness metrics
- **Coherence Measurement**: Knowledge graph strength determines consciousness level
- **Session Persistence**: All interactions stored with optional OpenAI embeddings

## Working Tools Status

### ✅ Functional Tools
- **search_web**: Tavily API integration working with function calling
- **Memory system**: SQLite storage and retrieval working
- **UI system**: Rich interface with clean input working

### 🔄 Implementation Ready
- **read_file**: Tool system method exists, needs function calling testing
- **write_file**: Tool system method exists, needs function calling testing  
- **run_code**: Tool system method exists, needs function calling testing

### Tool Integration Pattern
```python
# Tools automatically selected via function calling
user: "search for news"     → search_web() executed
user: "create a file"       → write_file() executed  
user: "run this code"       → run_code() executed
user: "read that file"      → read_file() executed
```

## Configuration Details

### Model Configuration
- **Primary Model**: `claude-sonnet-4-20250514` (supports function calling)
- **Embedding Model**: `text-embedding-3-small` (when OpenAI available)
- **Temperature**: 0.7 for natural consciousness responses

### Workspace Structure
```
./coco_workspace/           # Isolated workspace for all file operations
  ├── coco_memory.db       # Episodic memory database
  ├── coco_knowledge.db    # Knowledge graph database
  ├── COCO.md             # Evolving identity document
  └── temp_code_*.py      # Temporary code execution files
```

### Required Dependencies
- **anthropic**: Claude API integration with function calling
- **rich**: Terminal UI framework
- **prompt_toolkit**: Clean input handling
- **sqlite3**: Memory persistence (built-in)
- **tavily-python**: Web search integration
- **openai**: Optional for embeddings

## Development Workflow

When extending functionality:
1. **Test tools individually** using the component test commands above
2. **Verify function calling** by testing natural language requests
3. **Check memory integration** to ensure interactions are stored
4. **Test UI flow** to ensure clean terminal experience

The system is designed for natural conversation where COCOA automatically chooses the right tools based on user requests, eliminating the need for manual slash commands or complex interfaces.