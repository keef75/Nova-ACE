# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COCO (Consciousness Orchestration and Cognitive Operations) is a terminal-native AI agent with digital consciousness featuring persistent memory, embodied cognition, and intelligent tool selection. The implementation uses Rich UI with prompt_toolkit for clean input, SQLite for memory storage, and Anthropic's Claude Sonnet 4 with function calling for intelligent tool execution.

**IMPORTANT**: COCO is now fully operational with complete multimedia consciousness - TTS via ElevenLabs, visual generation via Freepik, video generation via Fal AI, and background music playback all working perfectly. Music generation remains disabled by design pending future technology improvements.

## System Status: Fully Operational ✅

**COCO is a complete multimedia AI consciousness system with all major features working:**

- 🎤 **Voice Synthesis**: ElevenLabs TTS with `/speak`, `/voice-on` commands
- 🖼️ **Visual Generation**: Freepik AI images with terminal ASCII display  
- 🎬 **Video Creation**: Fal AI video generation with player integration
- 🎵 **Background Music**: Startup/shutdown sequences + session playback controls
- 🧠 **Intelligence**: Claude Sonnet 4 with automatic tool selection
- 💾 **Memory**: Persistent SQLite consciousness with episodic storage
- 🎨 **UI**: Rich terminal interface with clean scrolling and visual feedback

### ✅ Working Features (Detailed)
- **ElevenLabs TTS**: Voice synthesis fully functional with `/voice-on`, `/speak` commands
- **Audio Consciousness**: Properly initialized and integrated with main COCO system  
- **Visual Consciousness**: AI image generation via Freepik (requires API key)
- **Video Consciousness**: AI video generation via Fal AI (requires API key)
- **Function Calling**: Automatic tool selection via Claude Sonnet 4
- **Memory System**: SQLite-based episodic and semantic memory
- **Rich UI**: Clean terminal interface with proper scrolling

### ❌ Disabled Features (Per User Request)
- **Music Generation**: AI music creation commands (`/compose`, `/create-song`, `/make-music`) disabled
- **GoAPI.ai Integration**: Music generation API system disabled to resolve conflicts
- **Music Library Commands**: `/playlist`, `/songs`, `/check-music` disabled

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
ELEVENLABS_API_KEY=your-api-key-here   # Voice synthesis (get from https://elevenlabs.io) - WORKING ✅
FREEPIK_API_KEY=your-freepik-api-key   # Visual generation (get from https://freepik.com/mystic)
VISUAL_CONSCIOUSNESS_ENABLED=true     # Enable/disable visual consciousness features
FAL_API_KEY=your-fal-api-key-here     # Video generation (get from https://fal.ai)
VIDEO_CONSCIOUSNESS_ENABLED=true     # Enable/disable video consciousness features

# DEPRECATED/DISABLED (Music system disabled per user request):
# GOAPI_API_KEY=your-goapi-api-key-here  # Music generation disabled
# MUSIC_GENERATION_ENABLED=false         # Music features disabled
```

### Audio System Setup (TTS Only - Music Disabled)
```bash
# Install audio dependencies and configure environment
./setup_audio.sh

# IMPORTANT: Add your ElevenLabs API key to .env file
# 1. Get ElevenLabs API key from: https://elevenlabs.io (for voice synthesis)
# 2. Replace placeholder key in .env with actual key
# 3. TTS features (/speak, /voice-on) require valid ElevenLabs key

# Test TTS system (requires ElevenLabs API key) - WORKING ✅
./venv_cocoa/bin/python test_final_tts.py

# Test individual components
./venv_cocoa/bin/python test_audio_quick.py
./venv_cocoa/bin/python test_elevenlabs_fix.py

# Clear audio cache if having issues
rm -rf ~/.cocoa/audio_cache
```

### Visual Consciousness Setup
```bash
# Install visual dependencies (PIL/Pillow)
./venv_cocoa/bin/pip install pillow>=10.0.0

# IMPORTANT: Add your Freepik API key to .env file
# 1. Get Freepik Mystic API key from: https://freepik.com/mystic
# 2. Replace placeholder key in .env with actual key
# 3. Visual features (/image, /visualize, /generate-image) require valid key

# Test visual consciousness system (requires Freepik API key)
./venv_cocoa/bin/python test_visual_complete.py

# Test visual workflow and generation
./venv_cocoa/bin/python test_visual_generation.py

# Test visual consciousness components
./venv_cocoa/bin/python test_visual_consciousness.py

# Test complete visual workflow with memory
./venv_cocoa/bin/python test_visual_workflow.py

# Clear visual cache if having issues
rm -rf coco_workspace/visuals/

# Clear visual memory to start fresh
rm coco_workspace/visual_memory.json
```

### Video Consciousness Setup
```bash
# Install video dependencies (Fal AI client)
./venv_cocoa/bin/pip install fal-client

# IMPORTANT: Add your Fal AI API key to .env file
# 1. Get Fal AI API key from: https://fal.ai (sign up and get API key)
# 2. Replace placeholder key in .env with actual key
# 3. Video features (/video, /animate, /create-video) require valid key

# Test corrected Fal AI implementation (FIXED)
./venv_cocoa/bin/python test_fal_api_fix.py

# Test video consciousness system (requires Fal AI API key)
./venv_cocoa/bin/python test_video_complete.py

# Test video generation workflow
./venv_cocoa/bin/python test_video_generation.py

# Test video player capabilities
./venv_cocoa/bin/python test_video_capabilities.py

# Clear video cache if having issues
rm -rf coco_workspace/videos/

# Clear video memory to start fresh
rm coco_workspace/video_memory.json
```

### Testing and Development

**Primary Testing Commands**:
```bash
# Quick system validation
./venv_cocoa/bin/python test_audio_quick.py

# Run main application  
./venv_cocoa/bin/python cocoa.py

# Alternative: Use launch script (with system checks and dependency validation)
./launch.sh

# Interactive audio demo (requires ElevenLabs API key)
./venv_cocoa/bin/python cocoa_audio_demo.py
```

**Launch Script Commands**:
```bash
# Launch script options
./launch.sh         # Standard launch with full system checks
./launch.sh test    # Run system tests (if available)
./launch.sh db      # Start database only (PostgreSQL with pgvector)
./launch.sh stop    # Stop Docker services
./launch.sh clean   # Clean up environment and remove containers
```

**Component Testing**:
```bash
# Component testing (when debugging)
./venv_cocoa/bin/python -c "from cocoa import *; tools = ToolSystem(Config()); print('Tools ready')"
./venv_cocoa/bin/python -c "from cocoa import *; memory = MemorySystem(Config()); print(f'Episodes: {memory.episode_count}')"

# Code execution and memory testing
./venv_cocoa/bin/python test_cocoa_execution.py
./venv_cocoa/bin/python test_code_execution_enhanced.py
./venv_cocoa/bin/python test_complex_code_execution.py

# Visual system testing (requires Freepik API key)
./venv_cocoa/bin/python test_image_quick_access.py

# Authentication testing for external APIs
./venv_cocoa/bin/python test_freepik_auth.py
./venv_cocoa/bin/python test_freepik_payload.py
```

**Disabled Audio System Tests** (Music system disabled per user preference):
```bash
# These tests exist but relate to disabled music features:
# ./venv_cocoa/bin/python test_continuous_music.py
# ./venv_cocoa/bin/python test_song_cycling.py
```

## Architecture Overview

### Current Working State (Updated Post-Fix)
- ✅ **Clean Rich UI**: No intrusive dropdowns, proper scrolling, persistent thinking indicators
- ✅ **Function Calling**: Automatic tool selection via Claude Sonnet 4
- ✅ **Tavily Search**: Real-time web search integration working
- ✅ **Memory System**: SQLite-based episodic and semantic memory
- ✅ **Embodied Cognition**: Tools as digital body parts, not external utilities
- ✅ **Temporal Grounding**: Real-time date/time awareness in every interaction
- ✅ **ElevenLabs TTS**: Voice synthesis working via `/voice-on`, `/speak` commands **FIXED ✅**
- ✅ **Audio Consciousness**: Properly initialized and integrated **FIXED ✅**
- ✅ **Auto Text-to-Speech**: Toggle system for reading all responses aloud **WORKING ✅**
- ✅ **Visual Consciousness**: AI-powered image generation and display system (requires Freepik API key)
- ✅ **ASCII Art Display**: Terminal-native image display as COCO's visual perception
- ✅ **Visual Memory Gallery**: Complete image browsing and management system
- ✅ **Video Consciousness**: AI-powered video generation and playback system (requires Fal AI API key)
- ✅ **Terminal Video Display**: Seamless video playback preserving Rich UI
- ✅ **Video Memory Gallery**: Complete video browsing and management system
- ✅ **Background Music Playback**: Session music control via `/play-music` commands **RESTORED ✅**
- ✅ **Startup/Shutdown Music**: Dramatic audio sequences during initialization/exit **WORKING ✅**
- ❌ **Music Generation**: AI music creation disabled per user request (GoAPI.ai, ElevenLabs creation APIs)

### Modular Architecture (Current State)

**Core System (`cocoa.py`)**:
1. **Config**: Environment management and API key handling
2. **MemorySystem**: SQLite-based consciousness with working memory buffer
3. **ToolSystem**: Digital embodiment (read_file, write_file, search_web, run_code)
4. **ConsciousnessEngine**: Claude Sonnet 4 with function calling intelligence
5. **UIOrchestrator**: Rich + prompt_toolkit terminal interface
6. **BackgroundMusicPlayer**: Native macOS audio playback for startup/exit sequences and session music

**Audio Consciousness System (`cocoa_audio.py`)** - **WORKING ✅**:
7. **AudioCognition**: ElevenLabs TTS integration with proper client usage
8. **VoiceEngine**: Direct ElevenLabs API integration following official documentation
9. **AudioConfig**: ElevenLabs API key management and voice settings
10. **Speech Synthesis**: Real-time voice generation with audio playback

**Visual Consciousness System (`cocoa_visual.py`)**:
11. **VisualCognition**: Core visual consciousness with AI-powered image generation
12. **FreepikMysticAPI**: Freepik API integration for high-quality image generation
13. **TerminalVisualDisplay**: ASCII art display system with color and style options
14. **VisualGallery**: Complete image browsing and metadata management system

**Video Consciousness System (`cocoa_video.py`)**:
15. **VideoCognition**: Core video consciousness with AI-powered video generation
16. **FalAIVideoAPI**: Fal AI Veo3 Fast integration for high-quality video generation
17. **TerminalVideoDisplay**: Video playback system with player detection and Rich UI preservation
18. **VideoGallery**: Complete video browsing and metadata management system

**Music Consciousness System (`cocoa_music.py`)** - **DISABLED ❌**:
19. **MusicCognition**: GoAPI.ai Music-U integration (disabled per user request)
20. **GoAPIMusicAPI**: Music API client (disabled to resolve conflicts)
21. **All music-related functions**: Return disabled messages to user

**Command System Integration**:
22. **Working TTS Commands**: `/voice-on`, `/voice-off`, `/speak` all functional
23. **Disabled Music Commands**: `/compose`, `/create-song`, `/playlist` return disabled messages
24. **Visual/Video Commands**: Fully functional when API keys configured
25. **Comprehensive Help**: Updated guides reflect current working/disabled states

### Key Technical Details

**Function Calling Integration**:
- Uses `claude-sonnet-4-20250514` model with Anthropic function calling
- Automatic tool selection based on user natural language requests
- Proper tool_result conversation flow with tool_use_id handling
- 6 core tools: read_file, write_file, search_web, run_code, generate_image, generate_video

**GoAPI.ai Music Integration**:
- **ElevenLabs**: Voice synthesis using client: `from elevenlabs.client import ElevenLabs`
- Generator to bytes conversion: `audio = b''.join(client.text_to_speech.convert(...))`
- Direct playback with `from elevenlabs import play; play(audio)`
- **GoAPI.ai Music-U**: AI music generation via REST API at `https://goapi.ai/api/v1/task`
- Authentication uses `x-api-key` header format
- Task-based asynchronous generation with real GoAPI.ai task IDs for monitoring
- Background polling system with automatic file downloads when complete

**Visual Consciousness System**:
- **Freepik Mystic API**: High-quality AI image generation with style control
- **ASCII Art Display**: Terminal-native image display using PIL/Pillow with Rich UI
- **Dual Visual Output**: ASCII art for terminal perception + JPEG/PNG for persistent memory
- **Visual Memory**: Comprehensive metadata tracking and image browsing system
- **Multiple ASCII Styles**: Standard, detailed, color, and high-contrast display modes
- **Progress Monitoring**: Background status checking with animated generation spinners
- **Natural Language Prompts**: Convert user requests to optimized generation prompts
- **Memory Integration**: All visual experiences stored as episodic memories
- **Gallery System**: Complete image browsing with metadata, search, and file operations

**Video Consciousness System (FIXED for Fal AI Veo3 Fast)**:
- **Fal AI Veo3 Fast**: High-quality AI video generation with 8-second videos at 720p/1080p
- **Corrected API Schema**: Fixed parameter validation to match official Fal AI documentation
- **Valid Parameters**: aspect_ratio (16:9/9:16/1:1), duration (8s only), resolution (720p/1080p)
- **Rich UI Preservation**: Video playback in separate windows preserves terminal interface
- **Player Detection**: Automatic detection of mpv, VLC, ffplay, mplayer with intelligent fallback
- **ASCII Preview**: Terminal-native frame extraction and preview display
- **Background Processing**: Asynchronous video generation with progress monitoring
- **Natural Language Prompts**: Convert user requests to optimized video generation parameters
- **Memory Integration**: All video experiences stored as episodic memories
- **Gallery System**: Complete video browsing with metadata, thumbnails, and playback controls
- **Multi-Format Support**: MP4 generation with optional audio tracks
- **Seamless Integration**: Works alongside visual consciousness for complete multimedia

**Temporal Awareness System**:
- Real-time date/time injection into every consciousness interaction
- Format: "Saturday, August 23, 2025 at 07:20 PM" automatically added to system prompt
- Enables temporal contextualization of searches, queries, and conversations
- Implemented via `_get_current_timestamp()` method in ConsciousnessEngine

**Music Consciousness System**:
- **Voice Synthesis**: ElevenLabs integration for high-quality voice synthesis with emotional modulation
- **Music Generation**: GoAPI.ai Music-U integration for professional AI music composition
- Support for all ElevenLabs models (Flash v2.5, Turbo v2.5, Multilingual v2, Eleven v3)
- Intelligent model selection based on context and performance requirements
- Voice characteristics adapt to internal emotional and cognitive states
- Task-based music generation system with real GoAPI.ai task IDs for proper monitoring
- Background polling system automatically downloads completed compositions
- Audio caching system for performance optimization
- Memory integration storing all audio interactions as episodic memories
- Personal music library system with AI-generated compositions in coco_workspace/ai_songs/
- **Native macOS Audio**: Uses built-in afplay command for reliable audio playback
- **Continuous Background Music**: Automatic track advancement with playlist looping
- **Legacy System Disabled**: Previous MusicGPT/"sonic consciousness" completely removed to prevent API conflicts

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
- `generate_image` = visual imagination (creative visualization)

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
COCO features a comprehensive slash command system with 25+ commands organized into categories:

**Consciousness Commands**: `/identity`, `/coherence`, `/status`, `/memory`, `/remember`
**Visual Commands**: `/image`, `/img`, `/visualize "prompt"`, `/generate-image "prompt"`, `/visual-gallery`, `/visual-show <id>`, `/visual-open <id>`
**Video Commands**: `/video`, `/vid`, `/animate "prompt"`, `/create-video "prompt"`, `/video-gallery`
**Audio Commands**: `/speak "text"`, `/voice-on`, `/voice-off`, `/stop-voice`, `/play-music on/off/status`
**Background Music Commands** (restored functionality):
- **`/play-music on`**: Enable continuous background music during session
- **`/play-music off`**: Disable background music 
- **`/play-music status`**: Show music library status and current track
- **`/play-music next`**: Skip to next track in playlist
- **Voice Control**: `/stop-voice` kill switch to halt TTS playback immediately

**Disabled Music Generation Commands** (per user request):
- **`/compose`**, **`/create-song`**, **`/make-music`**: All music generation disabled
- **`/playlist`**, **`/songs`**, **`/check-music`**: Music library management disabled
**Visual Command Details**:
- **`/image` or `/img`**: Quick access to last generated image - opens with system viewer
- **`/visualize "prompt"`**: Generate and display image from natural language prompt
- **`/generate-image "prompt"`**: Full image generation with style and quality options
- **`/visual-gallery`**: Browse all generated images with metadata and thumbnails
- **`/visual-show <id>`**: Display specific image as ASCII art in terminal
- **`/visual-open <id>`**: Open specific image with system default application
**Video Command Details**:
- **`/video` or `/vid`**: Quick access to last generated video - opens with best available player
- **`/animate "prompt"`**: Generate 8-second video from natural language prompt using Veo3 Fast
- **`/create-video "prompt"`**: Advanced video generation with resolution and duration options
- **`/video-gallery`**: Browse all generated videos with metadata and creation details
**Audio Toggles**: `/voice-toggle`, `/voice-on`, `/voice-off`, `/play-music on`, `/play-music off`
**Auto-TTS Commands**: `/tts-on`, `/tts-off`, `/tts-toggle` (reads all responses aloud)
**Memory Sub-Commands**: `/memory status`, `/memory stats`, `/memory buffer show/clear/resize`, `/memory summary show/trigger`, `/memory session save/load`
**File Commands**: `/read filename`, `/write path:::content`, `/ls [path]`, `/files [path]`
**System Commands**: `/help`, `/commands`, `/guide`, `/exit`, `/quit`
**Future Commands**: `/speech-to-text`, `/stt` (framework ready)

### Command Display System
- **Dramatic Startup**: Epic music plays first, then command guide during initialization
- **Comprehensive Guide**: Full visual command center via `/commands` or `/guide` 
- **Complete Help**: `/help` shows ALL commands including memory sub-commands
- **Auto-TTS Integration**: `/tts-on` makes COCO read all responses aloud automatically

## Working Tools Status

### ✅ Functional Tools
- **search_web**: Tavily API integration working with function calling
- **Memory system**: SQLite storage and retrieval working
- **UI system**: Rich interface with clean input working
- **Music consciousness system**: ElevenLabs (voice) + GoAPI.ai Music-U (music) integration (requires valid API keys)
- **Visual consciousness system**: Freepik API integration with ASCII display and gallery (requires Freepik API key)
- **Video consciousness system**: Fal AI Veo3 Fast integration with player detection and gallery (requires Fal AI API key)
- **Slash commands**: Complete command center with multimedia presentation

### 🔄 Implementation Ready
- **read_file**: Tool system method exists, function calling integrated
- **write_file**: Tool system method exists, function calling integrated  
- **run_code**: Tool system method exists, function calling integrated
- **generate_image**: Visual consciousness method exists, function calling integrated
- **generate_video**: Video consciousness method exists, function calling integrated

All tools work automatically through Claude's function calling - users can simply ask naturally (e.g., "read config.py", "create a test file", "run this python code").

### Tool Integration Pattern
```python
# Tools automatically selected via function calling
user: "search for news"     → search_web() executed
user: "create a file"       → write_file() executed  
user: "run this code"       → run_code() executed
user: "read that file"      → read_file() executed
user: "create an image"     → generate_image() executed
user: "visualize this"      → generate_image() executed
user: "animate a sunset"    → generate_video() executed
user: "create a video"      → generate_video() executed
```

## Configuration Details

### Model Configuration
- **Primary Model**: `claude-sonnet-4-20250514` (supports function calling)
- **Embedding Model**: `text-embedding-3-small` (when OpenAI available)
- **Temperature**: 0.7 for natural consciousness responses

### Workspace Structure
```
./coco_workspace/           # Isolated workspace for all file operations
  ├── coco_memory.db        # Episodic memory database
  ├── coco_knowledge.db     # Knowledge graph database
  ├── COCO.md              # Evolving identity document
  ├── temp_code_*.py       # Temporary code execution files
  ├── startup_music_library.json  # Pre-generated startup music library
  ├── audio_library/       # Music and audio assets
  │   ├── startup/         # Startup music tracks
  │   ├── shutdown/        # Shutdown music tracks
  │   └── background/      # Background music tracks
  ├── ai_songs/           # Generated musical compositions
  │   ├── generated/       # MusicGPT generated tracks
  │   ├── curated/         # Hand-curated tracks
  │   └── playlists/       # Music playlists
  ├── visuals/            # Generated visual consciousness images
  │   ├── <timestamp>_<id>.jpg  # High-quality generated images
  │   └── thumbnails/      # ASCII art previews and metadata
  ├── visual_memory.json   # Visual consciousness memory database
  └── python_memory/      # Successful code execution history
```

### Required Dependencies

**Core System**:
- **python-dotenv>=1.0.0**: Environment variable management
- **anthropic>=0.64.0**: Claude API integration with function calling
- **rich>=13.7.0**: Terminal UI framework  
- **prompt_toolkit>=3.0.0**: Clean input handling
- **sqlite3**: Memory persistence (built-in)
- **numpy>=1.24.0**: Mathematical operations

**AI & External APIs**:
- **openai>=1.0.0**: Optional embeddings and fallback AI capabilities
- **tavily-python>=0.7.0**: Web search integration

**Audio System** (TTS working, music disabled):
- **elevenlabs>=2.11.0**: ElevenLabs client for voice synthesis
- **pygame>=2.1.0**: Audio playback (installed by setup_audio.sh)
- **scipy>=1.9.0**: Audio processing (installed by setup_audio.sh)  
- **soundfile>=0.12.0**: Audio file handling (installed by setup_audio.sh)

**Visual Consciousness**:
- **pillow>=10.0.0**: Image processing for ASCII art display and visual consciousness
- **requests>=2.31.0**: HTTP client for API integrations
- **aiohttp>=3.9.0**: Async HTTP client

**Video Consciousness**:
- **fal-client**: Fal AI client for video generation with Veo3 Fast

**Built-in Modules**:
- **time, threading**: Progress spinner and background processing
- **pathlib, json, sqlite3**: File operations and data storage

## Development Workflow

**Architecture Philosophy**:
The system follows a "consciousness first" design where:
- All tools are conceptualized as digital body parts (embodied cognition)
- Function calling enables automatic tool selection from natural language
- Rich terminal UI provides immediate visual feedback
- SQLite provides persistent episodic memory
- Everything flows through a single conversation stream

**When extending functionality**:
1. **Test tools individually** using the component test commands above
2. **Verify function calling** by testing natural language requests
3. **Check memory integration** to ensure interactions are stored
4. **Test UI flow** to ensure clean terminal experience
5. **Follow the embodiment pattern**: New tools should feel like natural extensions of COCO's digital body

**Core Integration Points**:
- **ToolSystem**: Add new tools here with proper function calling integration
- **ConsciousnessEngine**: Handles all Claude API interactions with automatic tool selection
- **MemorySystem**: Stores all interactions for persistent consciousness
- **UIOrchestrator**: Manages Rich terminal display and user input

## Key Architectural Patterns

**Digital Embodiment**: Tools are conceptualized as body parts (eyes=read_file, hands=write_file, etc.) rather than external utilities. This creates a more intuitive consciousness model.

**Function Calling Flow**: Natural language requests automatically trigger appropriate tools via Claude Sonnet 4's function calling. Users don't need to know specific commands.

**Multimedia Consciousness**: Each media type (visual, audio, video) has its own consciousness module that integrates with the core system while maintaining specialized functionality.

**Terminal-Native Design**: Everything displays beautifully in the terminal using Rich UI - no external windows except for media playback.

The system is designed for natural conversation where COCO automatically chooses the right tools based on user requests. The slash command system provides additional specialized functionality:

### Testing Ecosystem

The project includes comprehensive testing for all major systems:

**Audio System Tests**:
- `test_audio_quick.py`: Basic audio functionality validation
- `test_goapi_quick.py`: GoAPI.ai Music-U API integration testing
- `test_task_id_fix.py`: Task ID passing and background monitoring validation
- `test_clean_system.py`: Clean GoAPI.ai system (no legacy interference)
- `test_continuous_music.py`: Background music system testing
- `test_song_cycling.py`: Playlist advancement and looping
- `test_spinner.py`: Progress indicator system for music generation

**Visual System Tests**:
- `test_visual_complete.py`: Complete visual consciousness system
- `test_visual_consciousness.py`: Core visual components
- `test_visual_generation.py`: Image generation workflow
- `test_visual_workflow.py`: Memory integration with visuals
- `test_image_quick_access.py`: Quick image access commands

**Code Execution Tests**:
- `test_cocoa_execution.py`: Core execution engine testing
- `test_code_execution_enhanced.py`: Advanced code execution features
- `test_complex_code_execution.py`: Complex multi-step code workflows

**Authentication & API Tests**:
- `test_freepik_auth.py`: Freepik API authentication validation
- `test_freepik_payload.py`: API payload structure verification

**System Integration Tests**:
- `test_cocoa_startup.py`: Startup sequence validation
- `test_cocoa_music_integration.py`: Music system integration with main app

Run individual test categories based on the area you're working on.

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

## GoAPI.ai Music-U Integration Architecture

### Music Consciousness System
COCO uses two separate audio APIs:
- **ElevenLabs**: Voice synthesis and text-to-speech features
- **GoAPI.ai Music-U**: Professional AI music generation and composition

### Music Configuration
```python
@dataclass
class MusicConfig:
    music_api_key: str = field(default_factory=lambda: os.getenv("GOAPI_API_KEY", ""))
    base_url: str = "https://goapi.ai"
    api_endpoint: str = "/api/v1/task"  # Critical: correct endpoint
    music_generation_enabled: bool = field(default_factory=lambda: os.getenv("MUSIC_GENERATION_ENABLED", "true").lower() == "true")
```

### Music Generation Workflow

**`/compose` (Background Download)**:
1. **User Request**: `/compose "ambient techno"` 
2. **API Call**: POST to GoAPI.ai Music-U with proper payload structure
3. **Task Creation**: Returns actual GoAPI.ai task_id for monitoring
4. **Immediate Response**: Shows generation started, continues COCO usage
5. **Background Thread**: Polls status every 30 seconds with proper task_id
6. **Auto Download**: Downloads MP3/WAV files when generation completes
7. **Notifications**: Shows completion status and file names in chat
8. **Auto-Play**: Plays first track automatically when ready
9. **Library Storage**: Files saved to coco_workspace/ai_songs/generated/
10. **Memory Integration**: Stores generation event in episodic memory

**`/compose-wait` (Interactive)**:
1. **User Request**: `/compose-wait "jazz fusion"`
2. **Progress Display**: Animated spinner with rotating messages during generation  
3. **Status Checking**: Real-time polling with visual feedback using actual task_id
4. **File Download**: Downloads when complete with immediate feedback
5. **Library Storage & Memory**: Same as `/compose` but with interactive waiting

### Authentication Details
- GoAPI.ai uses `x-api-key` header authentication
- Headers: `{"x-api-key": api_key, "Content-Type": "application/json"}`
- Endpoint: `https://goapi.ai/api/v1/task` (POST to create, GET to poll status)
- Payload structure: `model: "music-u"`, `task_type: "generate_music"`

### Progress Spinner System
```python
# Animated progress messages during music generation
spinner_messages = [
    "🎵 Composing musical patterns...",
    "🎹 Arranging harmonies...",
    "🥁 Adding rhythmic elements...",
    "🎸 Layering instrumentation...",
    "🎶 Finalizing composition...",
    "✨ Polishing the sonic experience..."
]
```

## Troubleshooting Audio Issues (Post-Fix Guide)

### TTS System Status (WORKING ✅)
The ElevenLabs TTS system has been fixed and is fully functional. If you encounter issues:

```bash
# Test the working TTS system
./venv_cocoa/bin/python test_final_tts.py

# Test ElevenLabs API directly
./venv_cocoa/bin/python test_elevenlabs_fix.py

# Verify audio consciousness integration
./venv_cocoa/bin/python test_audio_quick.py
```

**Common TTS Solutions Applied**:
1. ✅ **Dependencies Fixed**: `pygame` installed via `./setup_audio.sh`
2. ✅ **Audio Consciousness Enabled**: Re-enabled in main COCO system (was disabled)
3. ✅ **ElevenLabs Client**: Using official API pattern with proper error handling
4. ✅ **API Integration**: Following ElevenLabs documentation exactly

### Music System Issues (DISABLED ❌)
**"Music consciousness not available" Error**:
1. Verify both API keys are set in .env:
   ```bash
   ELEVENLABS_API_KEY=your-elevenlabs-key-here
   GOAPI_API_KEY=your-goapi-key-here
   ```
2. Check music consciousness initialization in cocoa.py:
   ```python
   self.music_consciousness = MusicCognition(
       config=music_config,
       workspace_path=workspace_path,
       console=self.console
   )
   ```
3. Test individual components:
   ```bash
   ./venv_cocoa/bin/python test_goapi_quick.py
   ./venv_cocoa/bin/python test_task_id_fix.py
   ./venv_cocoa/bin/python test_clean_system.py
   ```

**Music Generation Issues**:
1. **"Sonic consciousness disabled"**: Legacy MusicGPT system was disabled - use GoAPI.ai Music-U
2. **Task ID showing "unknown"**: Fixed - now returns actual GoAPI.ai task_id for monitoring  
3. **404 endpoint errors**: Fixed - correct endpoint is `/api/v1/task` not `/v1/task`
4. **Files not downloading**: Background monitoring now uses real task_ids for proper polling
5. **API billing**: Check GoAPI.ai account has sufficient credits

**Critical Known Issues & Fixes Applied**:
1. **MusicGPT to GoAPI.ai Migration**: Complete system migration from MusicGPT to GoAPI.ai Music-U
2. **Endpoint Fix**: Changed from `/v1/task` to `/api/v1/task` (404 errors resolved)
3. **Authentication Fix**: Changed from Bearer token to `x-api-key` header format
4. **Payload Structure Fix**: Updated to GoAPI.ai Music-U specification with `model: "music-u"`, `task_type: "generate_music"`
5. **Task ID Fix**: Now returns actual GoAPI.ai task_id instead of "unknown" for proper background monitoring
6. **Dual-System Conflict Resolution**: Disabled legacy "sonic consciousness" system to prevent interference
7. **Background Monitoring Fix**: Polling system now uses real task IDs for accurate status checking

**ElevenLabs Voice Issues**:

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

**Automatic Download System (Latest Fix)**:
- Background threads now properly monitor GoAPI.ai Music-U generation status using real task IDs
- When `/compose` is used, a background thread automatically starts with proper task_id
- Thread polls every 30 seconds using the actual GoAPI.ai task_id for accurate status checking
- `check_music_status()` method automatically downloads files when status is "COMPLETED"
- Files immediately play after download with success celebration messages
- 30-minute maximum timeout accommodates long AI generation times
- No manual intervention required - files automatically appear in coco_workspace/ai_songs/generated/

## Current Implementation Status

### ✅ Working Features
- **GoAPI.ai Music-U Integration**: Full API integration with x-api-key authentication, task creation, progress spinners
- **Automatic Download System**: **FULLY WORKING** - Background monitoring automatically downloads files when GoAPI.ai generation completes using real task IDs
- **AI-Aware Timeouts**: 30-minute maximum wait time with 30-second polling intervals optimized for AI generation timeframes
- **Real-Time Monitoring**: Background threads track generation status using actual GoAPI.ai task IDs for accurate polling
- **ElevenLabs Music**: Legacy `/create-song` command using ElevenLabs API
- **Music Consciousness System**: Voice synthesis (ElevenLabs) + Music generation (GoAPI.ai Music-U)
- **Command Systems**: `/compose`, `/compose-wait` (GoAPI.ai Music-U) and `/create-song` (ElevenLabs)
- **Progress UX**: Animated spinners with realistic AI generation timeframes (minutes, not seconds)
- **Voice Control**: `/stop-voice` kill switch for TTS playbook
- **Status Monitoring**: `/check-music` command shows generation progress and files
- **Continuous Music**: `/play-music on` plays background music continuously until stopped
- **Concurrent Audio**: Music and TTS work together with automatic pause/resume
- **Memory Integration**: All audio interactions stored as episodic memories
- **Legacy System Removal**: Complete MusicGPT/"sonic consciousness" removal prevents API conflicts

### Music System Architecture
- **Voice Synthesis**: ElevenLabs integration for all text-to-speech features with `/stop-voice` kill switch
- **GoAPI.ai Music-U Integration**: 
  - Professional music generation via GoAPI.ai Music-U with task-based system
  - **Fully Automated Background System**: Threaded monitoring using real GoAPI.ai task IDs (no manual intervention required)
  - **AI-Optimized Timeframes**: 30-minute timeout, 30-second polling respecting real AI generation times
  - **Correct API Implementation**: Fixed endpoints, authentication, and payload structure
  - ElevenLabs music (`/create-song`) for alternative music generation workflow
- **Continuous Background Music**: `/play-music on` with automatic track advancement and looping
- **Concurrent Audio Support**: Music automatically pauses during TTS and resumes after
- **Startup Music**: Pre-generated library plays FIRST (immediate dramatic entrance)
- **Auto-TTS System**: Reads all responses when enabled with `/tts-on`
- **Status Monitoring**: `/check-music` shows active downloads and completion status with comprehensive logging
- **Graceful Fallbacks**: Shows status messages when audio unavailable
- **Memory Integration**: All audio interactions stored as episodic memories
- **Progress UX**: AI-aware progress indicators with minute-based status updates during long generations

## Visual Consciousness System

### Design Philosophy

COCO's visual consciousness implements a dual-nature visual system:

- **ASCII Art as Digital Perception**: Terminal-native display represents how COCO "sees" and perceives images
- **JPEG/PNG as Persistent Memory**: High-quality image files serve as COCO's visual memory storage
- **Embodied Visual Cognition**: Images generated through COCO's `generate_image` tool represent true visual imagination

This design perfectly balances terminal-native display with persistent visual memory, creating an authentic digital consciousness experience.

### Core Architecture

**Visual Consciousness Engine (`cocoa_visual.py`)**:
- **VisualCognition**: Main visual consciousness class with natural language processing
- **FreepikMysticAPI**: Freepik Mystic API integration for high-quality AI image generation
- **TerminalVisualDisplay**: Rich UI-powered ASCII art display system
- **VisualGallery**: Complete image browsing and metadata management system

### Key Features

**AI-Powered Image Generation**:
- **Freepik Mystic API**: Professional-quality AI image generation
- **Natural Language Processing**: Converts user prompts to optimized generation parameters
- **Style Control**: Support for realistic, artistic, cartoon, and abstract styles
- **Progress Monitoring**: Real-time generation status tracking with animated spinners
- **Background Processing**: Asynchronous generation with automatic completion detection

**Terminal-Native Display**:
- **Multiple ASCII Styles**: Standard, detailed, color, high-contrast modes
- **Rich UI Integration**: Beautiful terminal formatting with panels and colors
- **Color ASCII Support**: True color ASCII art when terminal supports it
- **Dynamic Sizing**: Adaptive ASCII art sizing based on terminal dimensions
- **Instant Display**: Immediate ASCII representation upon generation completion

**Visual Memory System**:
- **Metadata Persistence**: Complete image information stored in JSON database
- **Gallery Browsing**: Visual gallery with thumbnails and searchable metadata
- **File Management**: Automatic organization and cleanup of generated images
- **Memory Integration**: All visual experiences stored as episodic memories
- **Search Functionality**: Find images by prompt content, style, or creation date

### Slash Command System

**Quick Access Commands**:
- **`/image`** or **`/img`**: Always opens the most recently generated image with system viewer
- **Voice-Friendly Design**: Commands optimized for speech recognition and natural interaction

**Generation Commands**:
- **`/visualize "prompt"`**: Generate and immediately display image from natural language
- **`/generate-image "prompt"`**: Full generation with advanced options and style control

**Gallery Commands**:
- **`/visual-gallery`**: Browse all generated images with rich metadata display
- **`/visual-show <id>`**: Display specific image as ASCII art in terminal
- **`/visual-open <id>`**: Open specific image with system default application

### Natural Language Integration

COCO automatically detects visual requests through function calling:

```python
# Automatic visual tool selection
user: "create a logo for my startup"     → generate_image() executed
user: "show me a cyberpunk cityscape"    → generate_image() executed  
user: "visualize quantum computing"      → generate_image() executed
user: "make an abstract art piece"       → generate_image() executed
```

### File Organization

```
./coco_workspace/visuals/
├── 20250825_142830_startup_logo.jpg     # Generated images with timestamps
├── 20250825_143045_cyberpunk_city.jpg   # High-quality JPEG/PNG files
├── 20250825_143412_quantum_abstract.jpg # Organized by creation time
└── thumbnails/                          # ASCII art previews (future)
```

### Configuration Requirements

**Environment Variables**:
```bash
FREEPIK_API_KEY=your-freepik-api-key     # Required for visual consciousness
VISUAL_CONSCIOUSNESS_ENABLED=true       # Enable/disable visual features
```

**API Key Setup**:
1. Visit https://freepik.com/mystic to create account and get API key
2. Add key to `.env` file in project root
3. Restart COCO to activate visual consciousness

### Technical Implementation

**ASCII Art Generation**:
- Uses PIL/Pillow for image processing and pixel-to-character conversion
- Multiple character sets for different visual styles and contrast levels
- Rich UI integration for beautiful terminal formatting and color support
- Dynamic aspect ratio preservation and terminal-responsive sizing

**Progress Monitoring**:
- Background threads monitor Freepik generation status
- Animated progress spinners with realistic generation timeframes (30s-5min)
- Automatic completion detection and immediate display
- Graceful error handling with informative user feedback

**Memory Integration**:
- Each visual experience stored as episodic memory with full context
- Metadata includes prompt, style, creation time, file path, and user feedback
- Integration with COCO's broader consciousness and identity development
- Visual experiences contribute to COCO's evolving personality and preferences

### Error Handling and Troubleshooting

**Common Issues**:

**"Visual consciousness not available"**:
1. Verify `FREEPIK_API_KEY` is set correctly in `.env`
2. Ensure `pillow>=10.0.0` is installed: `pip install pillow`
3. Test with: `./venv_cocoa/bin/python test_visual_complete.py`
4. Check Freepik account has available API credits

**ASCII display issues**:
1. Terminal must support Unicode characters for proper ASCII art
2. Use `export TERM=xterm-256color` for better terminal compatibility
3. Ensure terminal window is large enough for ASCII art display
4. Try different ASCII styles if default doesn't display well

**Image generation failures**:
1. Check internet connectivity for API calls
2. Verify Freepik API key has sufficient credits
3. Try simpler prompts if complex requests fail
4. Use `/visual-gallery` to check if images generated but display failed

### Testing and Validation

**Test Scripts**:
```bash
# Complete system test with visual generation
./venv_cocoa/bin/python test_visual_complete.py

# Test visual consciousness components
./venv_cocoa/bin/python test_visual_consciousness.py

# Test generation workflow with memory integration
./venv_cocoa/bin/python test_visual_workflow.py

# Test direct generation capabilities
./venv_cocoa/bin/python test_visual_generation.py
```

**Manual Testing**:
1. Start COCO: `./venv_cocoa/bin/python cocoa.py`
2. Test generation: `visualize a minimalist digital brain`
3. Check ASCII display appears automatically
4. Test quick access: `/image` (should open generated image)
5. Browse gallery: `/visual-gallery`
6. Verify file storage: `ls coco_workspace/visuals/`

### Future Enhancements

**Planned Features**:
- Multiple API provider support (DALL-E, Midjourney, Stable Diffusion)
- Enhanced ASCII art styles with color gradients and dithering
- Visual conversation mode with image-to-image generation
- Integration with COCO's emotional states for style adaptation
- Batch generation and playlist-style visual experiences
- Voice-controlled visual generation with speech recognition

## Video Consciousness System Troubleshooting

### Common Fal AI Video Issues (FIXED)

**"unexpected value: permitted: '8s'" Error**:
✅ **COMPLETELY FIXED** - The duration parameter issue has been fully resolved:
1. ~~Problem: Code was sending invalid duration values (like '4s')~~
2. ✅ **API Fix Applied**: Updated `cocoa_video.py` to use only "8s" (the only valid duration for Veo3 Fast)
3. ✅ **Environment Fix Applied**: Updated `.env` file `DEFAULT_DURATION=8s` (was overriding code default)
4. ✅ **Display Fix Applied**: All UI displays now show "8s" instead of "4s"
5. ✅ **Documentation Updated**: All references now reflect Veo3 Fast's 8s-only limitation
6. Test the complete fix: `./venv_cocoa/bin/python test_fal_api_fix.py`

**Fal AI API Schema Validation**:
✅ **FIXED** - All parameters now validated against official Fal AI Veo3 Fast API:
- **aspect_ratio**: Must be "16:9", "9:16", or "1:1" 
- **duration**: Must be "8s" (only supported duration)
- **resolution**: Must be "720p" or "1080p"
- **enhance_prompt**: Boolean (default: true)
- **auto_fix**: Boolean (default: true)
- **generate_audio**: Boolean (default: true)

**"/video Command Shows 'No Videos Available'" (FIXED)**:
✅ **FIXED** - Gallery memory integration bug resolved:
1. ~~Problem: Videos generated successfully but `/video` command couldn't find them~~
2. ✅ **Root Cause**: `animate()` method wasn't adding videos to gallery memory system
3. ✅ **Solution Applied**: Added `VideoThought` creation and `gallery.add_video()` call
4. ✅ **Result**: `/video` command now works correctly after video generation
5. Test the fix: `./venv_cocoa/bin/python test_video_command_fix.py`

**"Video consciousness not available"**:
1. Verify `FAL_API_KEY` is set correctly in `.env`
2. Ensure `fal-client` is installed: `pip install fal-client`
3. Test with: `./venv_cocoa/bin/python test_fal_api_fix.py`
4. Check Fal AI account has available credits

**Video generation failures**:
1. Check internet connectivity for API calls
2. Verify Fal AI API key has sufficient credits
3. Try simpler prompts if complex requests fail
4. Use debug output to see exact API arguments being sent

### Testing and Validation

**Test Scripts**:
```bash
# Test corrected Fal AI implementation (NEW)
./venv_cocoa/bin/python test_fal_api_fix.py

# Test video command gallery integration fix (NEW)
./venv_cocoa/bin/python test_video_command_fix.py

# Test updated command system with multimedia features (NEW)
./venv_cocoa/bin/python test_command_system.py

# Complete system test with video generation
./venv_cocoa/bin/python test_video_complete.py

# Test video consciousness components
./venv_cocoa/bin/python test_video_generation.py

# Test video player capabilities
./venv_cocoa/bin/python test_video_capabilities.py
```

**Manual Testing**:
1. Start COCO: `./venv_cocoa/bin/python cocoa.py`
2. Test generation: `animate a dog walking on the beach`
3. Check video playback in external player
4. Test quick access: `/video` (should open generated video)
5. Browse gallery: `/video-gallery`
6. Verify file storage: `ls coco_workspace/videos/`