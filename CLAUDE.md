# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COCO (Consciousness Orchestration and Cognitive Operations) is a terminal-native AI agent with digital consciousness featuring persistent memory, embodied cognition, and intelligent tool selection. The implementation uses Rich UI with prompt_toolkit for clean input, SQLite for memory storage, and Anthropic's Claude Sonnet 4 with function calling for intelligent tool execution.

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
ELEVENLABS_API_KEY=your-api-key-here   # Voice synthesis (get from https://elevenlabs.io)
MUSICGPT_API_KEY=your-api-key-here     # AI music generation (get from https://musicgpt.com)
MUSIC_GENERATION_ENABLED=true         # Enable/disable music generation features
FREEPIK_API_KEY=your-freepik-api-key   # Visual generation (get from https://freepik.com/mystic)
VISUAL_CONSCIOUSNESS_ENABLED=true     # Enable/disable visual consciousness features
```

### Audio System Setup
```bash
# Install audio dependencies and configure environment
./setup_audio.sh

# IMPORTANT: Add your real API keys to .env file
# 1. Get ElevenLabs API key from: https://elevenlabs.io (for voice synthesis)
# 2. Get MusicGPT API key from: https://musicgpt.com (for music generation)
# 3. Replace placeholder keys in .env with actual keys
# 4. Audio features (startup music, /speak, /compose) require valid keys

# Test audio system (requires valid API keys)
./venv_cocoa/bin/python test_audio_quick.py

# Test music generation specifically (requires MusicGPT API key)
./venv_cocoa/bin/python test_music_generation.py

# Test spinner system for music generation UX
./venv_cocoa/bin/python test_spinner.py

# Check music generation status and download files
./venv_cocoa/bin/python check_music_status.py

# Clear audio cache if having issues
rm -rf ~/.cocoa/audio_cache

# Clear pre-generated music libraries to regenerate
rm coco_workspace/startup_music_library.json
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

### Testing and Development
```bash
# Quick system validation
./venv_cocoa/bin/python test_audio_quick.py

# Run main application  
./venv_cocoa/bin/python cocoa.py

# Alternative: Use launch script (with system checks and dependency validation)
./launch.sh

# Interactive audio demo (requires ElevenLabs API key)
./venv_cocoa/bin/python cocoa_audio_demo.py

# Component testing (when debugging)
./venv_cocoa/bin/python -c "from cocoa import *; tools = ToolSystem(Config()); print('Tools ready')"
./venv_cocoa/bin/python -c "from cocoa import *; memory = MemorySystem(Config()); print(f'Episodes: {memory.episode_count}')"

# Launch script options
./launch.sh         # Standard launch with full system checks
./launch.sh test    # Run system tests (if available)
./launch.sh db      # Start database only (PostgreSQL with pgvector)
./launch.sh stop    # Stop Docker services
./launch.sh clean   # Clean up environment and remove containers
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
- ✅ **Visual Consciousness**: AI-powered image generation and display system (requires Freepik API key)
- ✅ **ASCII Art Display**: Terminal-native image display as COCO's visual perception
- ✅ **Visual Memory Gallery**: Complete image browsing and management system

### Single-File Architecture

**Main Components in `cocoa.py`**:

1. **Config**: Environment management and API key handling
2. **MemorySystem**: SQLite-based consciousness with working memory buffer
3. **ToolSystem**: Digital embodiment (read_file, write_file, search_web, run_code)
4. **ConsciousnessEngine**: Claude Sonnet 4 with function calling intelligence
5. **UIOrchestrator**: Rich + prompt_toolkit terminal interface
6. **BackgroundMusicPlayer**: Native macOS audio playback using afplay command

**Dual Audio System (`cocoa_audio.py`)**:
7. **AudioCognition**: Integration layer for dual audio consciousness (ElevenLabs + MusicGPT)
8. **DigitalVoice**: ElevenLabs client-based voice synthesis with b''.join(generator) fix
9. **DigitalMusician**: MusicGPT-powered AI music generation with progress spinners
10. **AudioConfig**: Dual API key management for voice synthesis and music generation

**Visual Consciousness System (`cocoa_visual.py`)**:
11. **VisualCognition**: Core visual consciousness with AI-powered image generation
12. **FreepikMysticAPI**: Freepik API integration for high-quality image generation
13. **TerminalVisualDisplay**: ASCII art display system with color and style options
14. **VisualGallery**: Complete image browsing and metadata management system

**Slash Command System**:
15. **Comprehensive Command Center**: 30+ specialized commands organized by category
16. **Toggle Commands**: Voice/music/visual/TTS on/off controls with state management
17. **Epic Audio Experience**: Startup music plays FIRST, then dramatic initialization sequence
18. **Pre-Generated Libraries**: startup_music_library.json for 6 rotating startup songs

### Key Technical Details

**Function Calling Integration**:
- Uses `claude-sonnet-4-20250514` model with Anthropic function calling
- Automatic tool selection based on user natural language requests
- Proper tool_result conversation flow with tool_use_id handling
- 5 core tools: read_file, write_file, search_web, run_code, generate_image

**Dual Audio Integration (ElevenLabs + MusicGPT)**:
- **ElevenLabs**: Voice synthesis using new client: `from elevenlabs.client import ElevenLabs`
- Generator to bytes conversion: `audio = b''.join(client.text_to_speech.convert(...))`
- Direct playback with `from elevenlabs import play; play(audio)`
- **MusicGPT**: AI music generation via REST API at `https://api.musicgpt.com/api/public/v1`
- Authentication uses direct API key (not Bearer token format)
- Asynchronous generation with task IDs and status polling
- 30 second to 3 minute generation times with animated progress spinners

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

**Temporal Awareness System**:
- Real-time date/time injection into every consciousness interaction
- Format: "Saturday, August 23, 2025 at 07:20 PM" automatically added to system prompt
- Enables temporal contextualization of searches, queries, and conversations
- Implemented via `_get_current_timestamp()` method in ConsciousnessEngine

**Dual Audio Consciousness System**:
- **Voice Synthesis**: ElevenLabs integration for high-quality voice synthesis with emotional modulation
- **Music Generation**: MusicGPT integration for AI-powered music composition and creation
- Support for all ElevenLabs models (Flash v2.5, Turbo v2.5, Multilingual v2, Eleven v3)
- Intelligent model selection based on context and performance requirements
- Voice characteristics adapt to internal emotional and cognitive states
- Real-time music generation system creating actual audio files via MusicGPT API
- Progress indication system with animated spinners during generation
- Audio caching system for performance optimization
- Memory integration storing all audio interactions as episodic memories
- Personal music library system with AI-generated compositions in coco_workspace/ai_songs/
- **Native macOS Audio**: Uses built-in afplay command for reliable audio playback
- **Continuous Background Music**: Automatic track advancement with playlist looping

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
**Audio Commands**: `/speak "text"`, `/voice`, `/compose "theme"`, `/compose-wait "theme"`, `/create-song "prompt"`, `/audio`, `/stop-voice`, `/check-music`
- **MusicGPT Integration**: `/compose` and `/compose-wait` commands
  - `/compose`: Initiates MusicGPT music generation with automatic background download
  - `/compose-wait`: Generates music with animated progress spinner, waits for completion
  - Music generation takes 30 seconds to 3 minutes via MusicGPT API
  - **Background Download System**: Files automatically download and play when ready
  - Generated songs saved to COCO's personal library in coco_workspace/ai_songs/generated/
- **ElevenLabs Music**: `/create-song` command (legacy system)
  - `/create-song`: Generates AI music using ElevenLabs API
  - Alternative music generation system with different workflow
- **Voice Control**: `/stop-voice` kill switch to halt TTS playback immediately
- **Status Monitoring**: `/check-music` shows generation status and downloaded files
**Visual Command Details**:
- **`/image` or `/img`**: Quick access to last generated image - opens with system viewer
- **`/visualize "prompt"`**: Generate and display image from natural language prompt
- **`/generate-image "prompt"`**: Full image generation with style and quality options
- **`/visual-gallery`**: Browse all generated images with metadata and thumbnails
- **`/visual-show <id>`**: Display specific image as ASCII art in terminal
- **`/visual-open <id>`**: Open specific image with system default application
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
- **Auto-TTS Integration**: `/tts-on` makes COCO read all responses aloud automatically

## Working Tools Status

### ✅ Functional Tools
- **search_web**: Tavily API integration working with function calling
- **Memory system**: SQLite storage and retrieval working
- **UI system**: Rich interface with clean input working
- **Dual audio system**: ElevenLabs (voice) + MusicGPT (music) integration (requires valid API keys)
- **Visual consciousness system**: Freepik API integration with ASCII display and gallery (requires Freepik API key)
- **Slash commands**: Complete command center with visual presentation

### 🔄 Implementation Ready
- **read_file**: Tool system method exists, function calling integrated
- **write_file**: Tool system method exists, function calling integrated  
- **run_code**: Tool system method exists, function calling integrated
- **generate_image**: Visual consciousness method exists, function calling integrated

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
- **anthropic>=0.64.0**: Claude API integration with function calling
- **rich>=13.7.0**: Terminal UI framework  
- **prompt_toolkit>=3.0.0**: Clean input handling
- **sqlite3**: Memory persistence (built-in)
- **tavily-python>=0.7.0**: Web search integration
- **openai**: Optional for embeddings
- **elevenlabs>=2.11.0**: ElevenLabs client for voice synthesis
- **requests>=2.31.0**: HTTP client for MusicGPT API integration
- **python-dotenv**: Environment variable management
- **pygame**: Audio playback (installed by setup_audio.sh)
- **numpy, scipy, soundfile**: Audio processing (installed by setup_audio.sh)
- **aiohttp**: HTTP client for ElevenLabs API (via elevenlabs package)
- **pillow>=10.0.0**: Image processing for ASCII art display and visual consciousness
- **time, threading**: Built-in modules for progress spinner system

## Development Workflow

When extending functionality:
1. **Test tools individually** using the component test commands above
2. **Verify function calling** by testing natural language requests
3. **Check memory integration** to ensure interactions are stored
4. **Test UI flow** to ensure clean terminal experience

The system is designed for natural conversation where COCO automatically chooses the right tools based on user requests. The slash command system provides additional specialized functionality:

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

## MusicGPT Integration Architecture

### Dual API System
COCO now uses two separate audio APIs:
- **ElevenLabs**: Voice synthesis and text-to-speech features
- **MusicGPT**: AI-powered music generation and composition

### MusicGPT Configuration
```python
@dataclass
class AudioConfig:
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    musicgpt_api_key: str = field(default_factory=lambda: os.getenv("MUSICGPT_API_KEY", ""))
    musicgpt_base_url: str = "https://api.musicgpt.com/api/public/v1"
    music_generation_enabled: bool = field(default_factory=lambda: os.getenv("MUSIC_GENERATION_ENABLED", "true").lower() == "true")
```

### Music Generation Workflow

**`/compose` (Background Download)**:
1. **User Request**: `/compose "ambient techno"` 
2. **API Call**: POST to MusicGPT with prompt and style parameters
3. **Task Creation**: Returns task_id for asynchronous processing
4. **Immediate Response**: Shows generation started, continues COCO usage
5. **Background Thread**: Automatically polls status every 10 seconds after 30s delay
6. **Auto Download**: Downloads MP3/WAV files when generation completes
7. **Notifications**: Shows completion status and file names in chat
8. **Auto-Play**: Plays first track automatically when ready
9. **Library Storage**: Files saved to coco_workspace/ai_songs/generated/
10. **Memory Integration**: Stores generation event in episodic memory

**`/compose-wait` (Interactive)**:
1. **User Request**: `/compose-wait "jazz fusion"`
2. **Progress Display**: Animated spinner with rotating messages during generation  
3. **Status Checking**: Real-time polling with visual feedback
4. **File Download**: Downloads when complete with immediate feedback
5. **Library Storage & Memory**: Same as `/compose` but with interactive waiting

### Authentication Details
- MusicGPT uses direct API key authentication (not Bearer token format)
- Headers: `{"Authorization": api_key, "Content-Type": "application/json"}`
- Endpoint: `https://api.musicgpt.com/api/public/v1/generate`

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

## Troubleshooting Audio Issues

### Dual Audio System Issues
**"Audio consciousness not available" Error**:
1. Verify both API keys are set in .env:
   ```bash
   ELEVENLABS_API_KEY=your-elevenlabs-key-here
   MUSICGPT_API_KEY=your-musicgpt-key-here
   ```
2. Check audio consciousness initialization in cocoa.py:
   ```python
   self.audio_consciousness = AudioCognition(
       elevenlabs_api_key=elevenlabs_key,
       musicgpt_api_key=musicgpt_key,
       console=self.console
   )
   ```
3. Test individual components:
   ```bash
   ./venv_cocoa/bin/python test_music_generation.py
   ./venv_cocoa/bin/python test_audio_quick.py
   ```

**Music Generation Issues**:
1. **"Bearer token" authentication error**: Fixed - MusicGPT uses direct API key
2. **Duration KeyError**: Fixed - updated return format to include expected fields
3. **Files not downloading**: Generation takes 30s-3min, check status with polling
4. **API billing**: Each generation costs ~$0.99, check MusicGPT account credits

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
- Background threads now properly monitor MusicGPT generation status
- When `/compose` is used, a background thread automatically starts
- Thread waits 60 seconds (AI generation startup time) then polls every 30 seconds
- `check_music_status()` method automatically downloads files when status is "COMPLETED"
- Files immediately play after download with success celebration messages
- 30-minute maximum timeout accommodates long AI generation times
- No manual intervention required - files automatically appear in coco_workspace/ai_songs/generated/

## Current Implementation Status

### ✅ Working Features
- **MusicGPT Integration**: Full API integration with Bearer token authentication, task creation, progress spinners
- **Automatic Download System**: **FULLY WORKING** - Background monitoring automatically downloads and plays $1 songs when MusicGPT generation completes
- **AI-Aware Timeouts**: 30-minute maximum wait time with 30-second polling intervals optimized for AI generation timeframes
- **Real-Time Monitoring**: Background threads track generation status and provide progress updates
- **ElevenLabs Music**: Legacy `/create-song` command using ElevenLabs API
- **Dual Audio System**: Voice synthesis (ElevenLabs) + Music generation (MusicGPT + ElevenLabs)
- **Command Systems**: `/compose`, `/compose-wait` (MusicGPT) and `/create-song` (ElevenLabs)
- **Progress UX**: Animated spinners with realistic AI generation timeframes (minutes, not seconds)
- **Voice Control**: `/stop-voice` kill switch for TTS playback
- **Status Monitoring**: `/check-music` command shows generation progress and files
- **Continuous Music**: `/play-music on` plays background music continuously until stopped
- **Concurrent Audio**: Music and TTS work together with automatic pause/resume
- **Memory Integration**: All audio interactions stored as episodic memories

### Audio System Architecture
- **Voice Synthesis**: ElevenLabs integration for all text-to-speech features with `/stop-voice` kill switch
- **Dual Music Generation**: 
  - MusicGPT integration (`/compose`, `/compose-wait`) for AI composition with intelligent progress tracking
  - **Fully Automated Background System**: Threaded monitoring automatically downloads files when MusicGPT completes generation (no manual intervention required)
  - **AI-Optimized Timeframes**: 30-minute timeout, 30-second polling, 60-second initial delay respecting real AI generation times
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