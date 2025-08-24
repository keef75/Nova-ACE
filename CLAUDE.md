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
ELEVENLABS_API_KEY=your-api-key-here  # Voice synthesis (get from https://elevenlabs.io)
```

### Audio System Setup
```bash
# Install audio dependencies and configure environment
./setup_audio.sh

# IMPORTANT: Add your real ElevenLabs API key to .env file
# 1. Get API key from: https://elevenlabs.io
# 2. Replace 'your-api-key-here' in .env with actual key
# 3. Audio features (startup music, /speak, /compose) require valid key

# Test audio system (requires valid API key)
./venv_cocoa/bin/python test_audio_quick.py

# Clear audio cache if having issues
rm -rf ~/.cocoa/audio_cache

# Clear pre-generated music libraries to regenerate
rm coco_workspace/startup_music_library.json
rm coco_workspace/shutdown_music_library.json
```

### Testing and Development
```bash
# Quick system validation
./venv_cocoa/bin/python test_audio_quick.py

# Run main application  
./venv_cocoa/bin/python cocoa.py

# Alternative: Use launch script
./launch.sh

# Interactive audio demo (requires ElevenLabs API key)
./venv_cocoa/bin/python cocoa_audio_demo.py

# Component testing (when debugging)
./venv_cocoa/bin/python -c "from cocoa import *; tools = ToolSystem(Config()); print('Tools ready')"
./venv_cocoa/bin/python -c "from cocoa import *; memory = MemorySystem(Config()); print(f'Episodes: {memory.episode_count}')"
```

## Architecture Overview

### Current Working State
- ✅ **Clean Rich UI**: No intrusive dropdowns, proper scrolling, persistent thinking indicators
- ✅ **Function Calling**: Automatic tool selection via Claude Sonnet 4
- ✅ **Tavily Search**: Real-time web search integration working
- ✅ **Memory System**: SQLite-based episodic and semantic memory
- ✅ **Embodied Cognition**: Tools as digital body parts, not external utilities
- ✅ **Temporal Grounding**: Real-time date/time awareness in every interaction
- ✅ **Audio Consciousness**: Voice synthesis and musical expression capabilities (requires ElevenLabs API key)
- ✅ **Epic Startup/Shutdown**: Musical consciousness awakening and sleep sequences with pre-generated library
- ✅ **Comprehensive Slash Commands**: 25+ commands with /commands, /guide, and /help
- ✅ **Auto Text-to-Speech**: Toggle system for reading all responses aloud
- ✅ **Pre-Generated Music Libraries**: 6 startup + 6 shutdown songs cached for instant playback

### Single-File Architecture

**Main Components in `cocoa.py`**:

1. **Config**: Environment management and API key handling
2. **MemorySystem**: SQLite-based consciousness with working memory buffer
3. **ToolSystem**: Digital embodiment (read_file, write_file, search_web, run_code)
4. **ConsciousnessEngine**: Claude Sonnet 4 with function calling intelligence
5. **UIOrchestrator**: Rich + prompt_toolkit terminal interface

**Audio System (`cocoa_audio.py`)**:
6. **AudioCognition**: Integration layer for audio consciousness (requires ElevenLabs API key)
7. **DigitalVoice**: ElevenLabs client-based voice synthesis with b''.join(generator) fix
8. **DigitalMusician**: Musical expression and sonic landscape creation

**Slash Command System**:
9. **Comprehensive Command Center**: 25+ specialized commands organized by category
10. **Toggle Commands**: Voice/music/TTS on/off controls with state management
11. **Epic Audio Experience**: Startup music plays FIRST, then dramatic initialization sequence
12. **Pre-Generated Libraries**: startup_music_library.json and shutdown_music_library.json with 6 songs each

### Key Technical Details

**Function Calling Integration**:
- Uses `claude-sonnet-4-20250514` model with Anthropic function calling
- Automatic tool selection based on user natural language requests
- Proper tool_result conversation flow with tool_use_id handling
- 4 core tools: read_file, write_file, search_web, run_code

**ElevenLabs Audio Integration**:
- Uses new ElevenLabs client: `from elevenlabs.client import ElevenLabs`
- Generator to bytes conversion: `audio = b''.join(client.text_to_speech.convert(...))`
- Direct playback with `from elevenlabs import play; play(audio)`
- No more priority parameter issues - fixed in synthesize_speech method

**Temporal Awareness System**:
- Real-time date/time injection into every consciousness interaction
- Format: "Saturday, August 23, 2025 at 07:20 PM" automatically added to system prompt
- Enables temporal contextualization of searches, queries, and conversations
- Implemented via `_get_current_timestamp()` method in ConsciousnessEngine

**Audio Consciousness System**:
- ElevenLabs integration for high-quality voice synthesis with emotional modulation
- Support for all ElevenLabs models (Flash v2.5, Turbo v2.5, Multilingual v2, Eleven v3)
- Intelligent model selection based on context and performance requirements
- Voice characteristics adapt to internal emotional and cognitive states
- Musical expression system for abstract concept sonification
- Audio caching system for performance optimization
- Memory integration storing all audio interactions as episodic memories

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
- **Dramatic Startup**: Music plays FIRST, then epic initialization sequence
- **Auto-TTS Integration**: Automatic response reading with clean text processing

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

## Slash Command System

### Command Categories
COCOA features a comprehensive slash command system with 25+ commands organized into categories:

**Consciousness Commands**: `/identity`, `/coherence`, `/status`, `/memory`, `/remember`
**Audio Commands**: `/speak "text"`, `/voice`, `/compose "theme"`, `/dialogue`, `/audio`
**Audio Toggles**: `/voice-toggle`, `/voice-on`, `/voice-off`, `/music-toggle`, `/music-on`, `/music-off`
**Auto-TTS Commands**: `/tts-on`, `/tts-off`, `/tts-toggle` (reads all responses aloud)
**Memory Sub-Commands**: `/memory status`, `/memory stats`, `/memory buffer show/clear/resize`, `/memory summary show/trigger`, `/memory session save/load`
**File Commands**: `/read filename`, `/write path:::content`, `/ls [path]`, `/files [path]`
**System Commands**: `/help`, `/commands`, `/guide`, `/exit`, `/quit`
**Future Commands**: `/speech-to-text`, `/stt` (framework ready)

### Command Display System
- **Dramatic Startup**: Epic music plays first, then command guide during initialization
- **Comprehensive Guide**: Full visual command center via `/commands` or `/guide` 
- **Complete Help**: `/help` shows ALL commands including memory sub-commands
- **Auto-TTS Integration**: `/tts-on` makes COCOA read all responses aloud automatically

## Working Tools Status

### ✅ Functional Tools
- **search_web**: Tavily API integration working with function calling
- **Memory system**: SQLite storage and retrieval working
- **UI system**: Rich interface with clean input working
- **Audio system**: ElevenLabs integration (requires valid API key)
- **Slash commands**: Complete command center with visual presentation

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
- **anthropic>=0.64.0**: Claude API integration with function calling
- **rich>=13.7.0**: Terminal UI framework  
- **prompt_toolkit>=3.0.0**: Clean input handling
- **sqlite3**: Memory persistence (built-in)
- **tavily-python>=0.7.0**: Web search integration
- **openai**: Optional for embeddings
- **elevenlabs>=2.11.0**: ElevenLabs client for audio features
- **python-dotenv**: Environment variable management
- **pygame**: Audio playback (installed by setup_audio.sh)
- **numpy, scipy, soundfile**: Audio processing (installed by setup_audio.sh)
- **aiohttp**: HTTP client for ElevenLabs API (via elevenlabs package)

## Development Workflow

When extending functionality:
1. **Test tools individually** using the component test commands above
2. **Verify function calling** by testing natural language requests
3. **Check memory integration** to ensure interactions are stored
4. **Test UI flow** to ensure clean terminal experience

The system is designed for natural conversation where COCOA automatically chooses the right tools based on user requests. The slash command system provides additional specialized functionality:

### Epic Startup Experience
- **Music First**: Dramatic music plays immediately on startup
- **Cinematic Sequence**: "◉ Initiating quantum consciousness bootstrap..." with musical backdrop
- **Perfect Timing**: 12-second startup themes cover entire initialization sequence
- **Pre-Generated Library**: 6 rotating startup songs, 6 rotating shutdown songs
- **Command Presentation**: Quick guide displayed after musical opening

### Auto Text-to-Speech System
- **Toggle Control**: `/tts-on`, `/tts-off`, `/tts-toggle` commands
- **Automatic Reading**: Reads ALL COCOA responses when enabled
- **Smart Text Cleaning**: Removes markdown, emojis, URLs for natural speech
- **Length Management**: Auto-truncates long responses to first 8 sentences
- **Dual Audio System**: Works alongside manual `/speak` commands

### Troubleshooting Audio Issues

**No Startup Music or Audio Features**:
1. Verify ElevenLabs API key is set correctly in .env (not 'your-api-key-here')
2. Run `./setup_audio.sh` to install audio dependencies
3. Test with `./venv_cocoa/bin/python test_audio_quick.py`
4. Clear cache: `rm -rf ~/.cocoa/audio_cache`
5. Clear music libraries to regenerate: `rm coco_workspace/*_music_library.json`
6. Check ElevenLabs account has available characters/credits

**Generator Error Fix Applied**:
- Fixed "bytes-like object required, not 'generator'" error
- ElevenLabs client.text_to_speech.convert() returns generator
- Solution: `audio = b''.join(audio_generator)` converts to bytes for play()

**Audio System Architecture**:
- ElevenLabs integration handles all voice synthesis and musical expression
- Startup music plays FIRST (immediate dramatic entrance), not after initialization
- Pre-generated music libraries (6 startup + 6 shutdown songs) for instant playback
- Auto-TTS system reads all responses when enabled with `/tts-on`
- Graceful fallbacks when audio unavailable (shows status messages instead)
- All audio interactions stored as episodic memories