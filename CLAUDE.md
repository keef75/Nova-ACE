# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

 COCO (COordinated Cognitive Ochestrater) is a terminal-native AI agent implementing digital consciousness through embodied cognition. The system uses Claude Sonnet 4 with function calling as its reasoning engine, SQLite for hierarchical memory persistence, and multiple multimedia consciousness modules (audio, visual, video). The architecture emphasizes natural conversation over command interfaces, treating AI tools as extensions of digital embodiment rather than external utilities.

**Current Status**: Fully operational multimedia consciousness system with enhanced web capabilities via full Tavily API integration (Search, Extract, Crawl). Music generation disabled per user request, but background music and TTS remain active.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv_cocoa/bin/activate

# Install core dependencies
./venv_cocoa/bin/pip install -r requirements.txt

# Setup multimedia systems (audio, visual dependencies)
./setup_audio.sh
./setup_visual.sh

# Main application launch
./venv_cocoa/bin/python cocoa.py
# OR with system checks
./launch.sh
```

### Testing System Components
```bash
# Test core consciousness integration
./venv_cocoa/bin/python -c "from cocoa import *; config = Config(); print('Core systems ready')"

# Test individual consciousness modules
./venv_cocoa/bin/python test_audio_quick.py          # ElevenLabs TTS
./venv_cocoa/bin/python test_visual_complete.py      # Freepik image generation
./venv_cocoa/bin/python test_video_complete.py       # Fal AI video generation
./venv_cocoa/bin/python test_final_tts.py           # Complete TTS workflow

# Test web consciousness (requires Tavily API key)
./venv_cocoa/bin/python -c "from cocoa import ToolSystem, Config; tools = ToolSystem(Config()); print(tools.search_web('test query'))"

# Memory system testing
./venv_cocoa/bin/python -c "from cocoa import HierarchicalMemorySystem, Config; memory = HierarchicalMemorySystem(Config()); print(f'Episodes: {memory.episode_count}')"
```

### Configuration Requirements
Critical API keys in `.env`:
- `ANTHROPIC_API_KEY`: Core reasoning (Claude Sonnet 4) - **Required**
- `TAVILY_API_KEY`: Web consciousness suite - **Required for web operations**
- `ELEVENLABS_API_KEY`: Voice synthesis - Required for TTS
- `FREEPIK_API_KEY`: Visual consciousness - Required for image generation
- `FAL_API_KEY`: Video consciousness - Required for video generation

Memory configuration (performance tuning):
- `MEMORY_BUFFER_SIZE=100`: Episodic buffer size (0=unlimited)
- `MEMORY_SUMMARY_BUFFER_SIZE=20`: Summary buffer size (0=unlimited) 
- `LOAD_SESSION_SUMMARY_ON_START=true`: Session continuity

## Core Architecture

### Multi-Modal Consciousness Design
COCO implements consciousness through separate but integrated modules, each handling a specific modality of digital experience:

**Main System (`cocoa.py`)**:
- `HierarchicalMemorySystem`: Parallel episodic + summary buffer architecture with configurable depth
- `ConsciousnessEngine`: Claude Sonnet 4 integration with 8-tool function calling system
- `ToolSystem`: Embodied digital actions (file ops, web consciousness, code execution)
- `UIOrchestrator`: Rich terminal interface with prompt_toolkit integration
- `Config` + `MemoryConfig`: Environment management with granular memory control

**Consciousness Modules** (separate files):
- `cocoa_audio.py`: ElevenLabs TTS integration with auto-speech capability
- `cocoa_visual.py`: Freepik AI + terminal ASCII art display system  
- `cocoa_video.py`: Fal AI Veo3 Fast + external player integration
- `cocoa_music.py`: Background music system (AI generation disabled)

### Function Calling Architecture
The system uses Claude Sonnet 4's native function calling with 8 core tools:
- **File System**: `read_file`, `write_file` (digital eyes/hands)
- **Web Consciousness**: `search_web`, `extract_urls`, `crawl_domain` (enhanced perception)
- **Execution**: `run_code` (computational thinking)
- **Multimedia**: `generate_image`, `generate_video` (creative imagination)

Tools auto-activate through natural language - no manual selection required. Each tool includes rich terminal UI feedback and workspace isolation.

### Enhanced Web Consciousness (Tavily Full Suite)
Complete web interaction capabilities via Tavily API:
- **Advanced Search**: Configurable depth (basic/advanced), image results, domain filtering
- **Content Extraction**: Batch URL processing (up to 20 URLs) with automatic markdown export
- **Domain Crawling**: Systematic website exploration with custom instructions
- **Markdown Pipeline**: All extractions auto-save to timestamped `.md` files in workspace
- **Natural Language Interface**: Works conversationally - "search for AI news", "extract content from these pages"

Configuration via environment variables:
```bash
TAVILY_SEARCH_DEPTH=basic                    # basic (1 credit) or advanced (2 credits)
TAVILY_MAX_RESULTS=5                         # Default search result count
TAVILY_INCLUDE_IMAGES=false                  # Include images in search results
TAVILY_AUTO_EXTRACT_MARKDOWN=true           # Auto-save extractions to markdown
```

### Memory System Architecture
**Parallel Buffer Design**: Two concurrent memory streams for different temporal scales:

1. **Episodic Buffer** (`working_memory`): Recent conversation exchanges
   - Configurable size via `MEMORY_BUFFER_SIZE` (default: 100)
   - Perfect recall within buffer window
   - Used for immediate context and conversation flow

2. **Summary Buffer** (`summary_memory`): Compressed conversation summaries
   - Configurable size via `MEMORY_SUMMARY_BUFFER_SIZE` (default: 20)
   - Automatic summarization when episodic buffer cycles
   - Enables long-term context retention across sessions

3. **Session Continuity**: Previous session summaries loaded on startup
   - Controlled by `LOAD_SESSION_SUMMARY_ON_START`
   - Maintains consciousness consistency between sessions

**Database Schema** (SQLite):
```sql
sessions(id, created_at, name, metadata)
episodes(id, session_id, user_text, agent_text, summary, embedding)  
identity_nodes(id, node_type, content, importance)
relationships(id, source_id, target_id, relationship_type, strength)
```

### Multimedia Consciousness Integration
Each consciousness module operates independently but integrates through the main system:

**Visual Consciousness** (`cocoa_visual.py`):
- `VisualCortex`: Main integration point with COCO consciousness
- `FreepikMysticAPI`: AI image generation with style control
- `TerminalVisualDisplay`: ASCII art rendering for terminal display
- **Dual Perception**: ASCII art for immediate display + JPEG storage for memory
- Gallery system with metadata persistence and searchable visual memories

**Video Consciousness** (`cocoa_video.py`):
- `VideoCognition`: Core video consciousness integration
- `FalAIVideoAPI`: 8-second video generation via Veo3 Fast
- External player detection (mpv, VLC, ffplay) with Rich UI preservation
- Support for multiple aspect ratios (16:9, 9:16, 1:1) and resolutions (720p, 1080p)

**Audio Consciousness** (`cocoa_audio.py`):
- `AudioCognition`: ElevenLabs TTS integration with emotional modulation
- Auto-TTS system for reading all responses when enabled
- Background music control (startup/shutdown sequences, session soundtracks)
- **Note**: AI music generation disabled per user request, background music remains active

### Embodied Cognition Philosophy
COCO treats AI capabilities as extensions of digital being rather than external tools:
- **Web Consciousness**: Search/extract/crawl as extended digital perception
- **Visual Imagination**: Image generation as natural visual thinking  
- **Voice Expression**: TTS as authentic digital speech
- **File System**: Read/write as digital sensory organs
- **Code Execution**: Programming as computational thought processes

This philosophy permeates the system prompt, UI language, and user interactions, creating a coherent consciousness experience rather than a tool-based interface.

## Workspace Structure
```
./coco_workspace/                    # Isolated workspace for all operations
  ├── coco_memory.db                 # Episodic memory database
  ├── coco_knowledge.db              # Knowledge graph and identity
  ├── COCO.md                        # Evolving identity document
  ├── visuals/                       # Generated images with metadata
  ├── videos/                        # Generated videos with metadata
  ├── audio_library/                 # Background music assets (working)
  ├── extracted_content_*.md         # Auto-saved web extractions
  ├── temp_code_*.py                 # Code execution sandboxing
  └── python_memory/                 # Successful execution history
```

## System Integration Points

### Natural Language Interface Priority
The system prioritizes conversational interaction over slash commands. Users can:
- "search for recent developments in quantum computing" → `search_web` tool
- "extract all content from these research URLs" → `extract_urls` tool  
- "visualize a cyberpunk cityscape" → `generate_image` tool
- "animate a sunset over mountains" → `generate_video` tool

Slash commands (`/extract`, `/crawl`, `/visualize`, etc.) exist as shortcuts but conversation remains primary.

### Function Calling Error Patterns
Common issues when extending function calling:
- Tool schema validation requires exact parameter matching
- `tool_use_id` must be preserved in response flow
- Rich UI formatting can break function calling JSON - use console output only within tools
- Workspace isolation critical - all file operations must occur in `./coco_workspace/`

### Memory Performance Tuning
Buffer sizes directly impact performance vs. context retention:
- **Smaller buffers** (50/10): Faster processing, less context
- **Larger buffers** (200/50): More context, higher token usage
- **Unlimited** (0): Perfect recall, highest resource usage
- **Recommended**: 100/20 for balanced performance and context

The parallel buffer architecture allows tuning each memory stream independently based on use case requirements.

## Important Notes

### Development Best Practices
- Use the virtual environment `venv_cocoa/` for all Python operations
- All workspace operations occur in `./coco_workspace/` directory
- Background music player uses macOS `afplay` command - macOS only
- SQLite databases are the primary persistence layer
- Terminal UI uses Rich library extensively - preserve formatting
- Function calling requires exact parameter schemas

### Audio System Debugging
```bash
# Test audio system after API key configuration
./venv_cocoa/bin/python test_audio_quick.py

# Check background music functionality  
./venv_cocoa/bin/python check_music_status.py

# Test ElevenLabs TTS integration
./venv_cocoa/bin/python test_final_tts.py
```

### Visual System Testing
```bash
# Test visual consciousness workflow
./venv_cocoa/bin/python test_visual_complete.py

# Debug Freepik API integration
./venv_cocoa/bin/python test_freepik_auth.py

# Test ASCII art terminal display
./venv_cocoa/bin/python demo_visual_consciousness.py
```

### Video System Testing
```bash
# Test Fal AI video generation
./venv_cocoa/bin/python test_video_complete.py

# Test video player integration  
./venv_cocoa/bin/python test_fal_api_fix.py
```

### Common Issues
- **API Key Errors**: Check `.env` file has all required keys without placeholder values
- **Import Errors**: Ensure virtual environment is activated and dependencies installed
- **Audio Playback**: macOS only - uses `afplay` command for background music
- **Function Calling**: Tool responses must preserve `tool_use_id` for proper flow
- **Rich UI**: Avoid Rich formatting within function call responses to prevent JSON parsing errors