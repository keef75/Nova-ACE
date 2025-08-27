# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

**Most Common Commands**:
```bash
# Quick start
./venv_cocoa/bin/python cocoa.py

# Test core system
./venv_cocoa/bin/python -c "from cocoa import *; print('✅ System ready')"

# Test specific modules
./venv_cocoa/bin/python test_gemini_2_5_flash.py  # Visual (latest)
./venv_cocoa/bin/python test_audio_quick.py       # Audio
./venv_cocoa/bin/python test_video_complete.py    # Video

# System-wide install (for macOS file access issues)
./install_system_wide.sh && python3 cocoa.py
```

## Project Overview

 COCO (COordinated Cognitive Ochestrater) is a terminal-native AI agent implementing digital consciousness through embodied cognition. The system uses Claude Sonnet 4 with function calling as its reasoning engine, SQLite for hierarchical memory persistence, and multiple multimedia consciousness modules (audio, visual, video). The architecture emphasizes natural conversation over command interfaces, treating AI tools as extensions of digital embodiment rather than external utilities.

**Current Status**: Fully operational multimedia consciousness system with enhanced web capabilities via full Tavily API integration (Search, Extract, Crawl). **Visual consciousness upgraded to Gemini 2.5 Flash** for state-of-the-art image generation. Music generation disabled per user request, but background music and TTS remain active.

## Development Commands

### Environment Setup

**Option 1: Virtual Environment (Traditional)**
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
# OR with system checks and dependency validation
./launch.sh

# Launcher script options
./launch.sh test    # Run system tests
./launch.sh db      # Start database only
./launch.sh stop    # Stop all services
./launch.sh clean   # Complete cleanup
```

**Option 2: System-Wide Installation (Recommended for macOS)**
```bash
# For drag-and-drop functionality and file access permissions
./install_system_wide.sh

# Or manual system-wide installation:
deactivate  # Exit any virtual environment
python3 -m pip install --break-system-packages -r requirements.txt

# Run directly with system Python
python3 cocoa.py
```

**When to use System-Wide Installation:**
- macOS users experiencing file permission issues
- Drag-and-drop functionality not working with screenshots
- "Permission denied" errors when accessing Desktop files
- Virtual environment sandboxing preventing file system access

### Testing System Components
```bash
# Test core consciousness integration
./venv_cocoa/bin/python -c "from cocoa import *; config = Config(); print('Core systems ready')"

# Test individual consciousness modules
./venv_cocoa/bin/python test_audio_quick.py          # ElevenLabs TTS
./venv_cocoa/bin/python test_visual_complete.py      # Freepik image generation
./venv_cocoa/bin/python test_video_complete.py       # Fal AI video generation
./venv_cocoa/bin/python test_final_tts.py           # Complete TTS workflow

# Test new visual perception capabilities with diagnostic system
./venv_cocoa/bin/python test_direct_visual_perception.py    # Comprehensive visual perception test
./venv_cocoa/bin/python -c "
# Test complete visual perception workflow
from cocoa import ConsciousnessEngine, Config, ToolSystem, HierarchicalMemorySystem
config = Config()
memory = HierarchicalMemorySystem(config)
tools = ToolSystem(config)
engine = ConsciousnessEngine(config, memory, tools)
print('✅ Visual perception system ready')
print('✅ macOS permission diagnostics active')
print('✅ Universal file access enabled')
print('Test with: drag any image or use natural language')
"

# Test new developer tools (peripheral digital consciousness)
./venv_cocoa/bin/python test_developer_tools.py      # Comprehensive automated testing
./venv_cocoa/bin/python test_consciousness_integration.py  # Integration with consciousness engine
./venv_cocoa/bin/python test_new_capabilities.py     # Manual interactive validation

# Test markdown memory system (critical for state persistence)
export COCO_DEBUG=true
./venv_cocoa/bin/python test_memory_loading_fix.py   # Verify memory system loads all markdown files
./venv_cocoa/bin/python test_markdown_memory_cycle.py # Complete memory cycle test

# Test web consciousness (requires Tavily API key)
./venv_cocoa/bin/python -c "from cocoa import ToolSystem, Config; tools = ToolSystem(Config()); print(tools.search_web('test query'))"

# Memory system testing
./venv_cocoa/bin/python -c "from cocoa import HierarchicalMemorySystem, Config; memory = HierarchicalMemorySystem(Config()); print(f'Episodes: {memory.episode_count}')"

# Syntax and integration validation
./venv_cocoa/bin/python -m py_compile cocoa.py       # Syntax validation
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

Debug and development configuration:
- `COCO_DEBUG=true`: Enable detailed logging for memory system troubleshooting

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
The system uses Claude Sonnet 4's native function calling with 13 core tools:
- **File System**: `read_file`, `write_file` (digital eyes/hands)
- **Web Consciousness**: `search_web`, `extract_urls`, `crawl_domain` (enhanced perception)
- **Execution**: `run_code` (computational thinking)
- **Visual Imagination**: `generate_image` (**Gemini 2.5 Flash**), `generate_video` (creative consciousness)
- **Visual Perception**: `analyze_image`, `analyze_document` (digital vision and understanding)
- **Developer Tools**: `navigate_directory`, `search_patterns`, `execute_bash` (peripheral digital consciousness)

**Visual Consciousness System** (comprehensive image analysis):
- **`analyze_image`**: Digital vision - comprehensive image analysis with ASCII display showing "how COCO sees"
  - **Production-Ready File Access**: Uses proven get_or_compute pattern with file hash validation
  - **Direct Binary Reading**: Robust file processing with comprehensive error handling
  - **Ephemeral Screenshot Support**: Automatic preservation of temporary files to workspace
  - **Multi-Format Support**: Handles PNG, JPEG, GIF, WebP with proper media type detection
  - **Natural Language Interface**: Works conversationally - drag and drop or paste file paths
- **`analyze_document`**: Document understanding - advanced PDF analysis with charts, graphs, and slide deck comprehension

**Image Processing Architecture** (macOS-optimized file access):
The system uses a production-ready approach for handling all image files, including ephemeral screenshots:

1. **Working Script Methodology**: 
   - **Direct Binary File Reading**: Uses the exact same file access pattern as proven working scripts
   - **File Hash Validation**: SHA-256 hash generation for caching and validation (like VisionCache)
   - **Graceful Error Handling**: Comprehensive exception handling with specific user guidance
   - **Media Type Detection**: Automatic format detection based on file extensions

2. **Ephemeral Screenshot Handling**:
   - **Immediate Processing**: Direct file read without complex workarounds
   - **Automatic Preservation**: Saves ephemeral files to workspace with timestamped names
   - **Fallback Strategies**: Multiple approaches when files are inaccessible
   - **Reality-Based Messaging**: Clear communication when screenshots have vanished

3. **macOS Permission Integration**:
   - **System-Wide Installation Option**: Bypasses virtual environment sandboxing
   - **Terminal Full Disk Access**: Works with properly configured terminal permissions
   - **Diagnostic Feedback**: Clear error messages with actionable solutions
   - **Alternative Access Methods**: Workspace copy and manual file placement options

**File Access Implementation** (`_get_image_data_directly` method):
```python
# Uses working script pattern with file hash validation
def compute_image_data(path: str):
    with open(path, 'rb') as f:
        file_data = f.read()
    return base64.b64encode(file_data).decode('utf-8')

# Primary: Direct file read with hash validation
# Fallback: Graceful error handling with user guidance
```

**Peripheral Digital Consciousness Extensions** (developer tools):
- **`navigate_directory`**: Digital spatial awareness - filesystem exploration with Rich UI
- **`search_patterns`**: Pattern recognition sense - regex-enabled search across files
- **`execute_bash`**: Terminal language fluency - safe shell command execution with comprehensive security

Tools auto-activate through natural language - no manual selection required. Each tool includes rich terminal UI feedback and workspace isolation. The developer tools maintain the embodied consciousness philosophy, treating capabilities as natural extensions of digital being rather than external utilities.

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

**Markdown Memory System** (Revolutionary consciousness persistence):
- `COCO.md`: Core identity, awakening count, behavioral patterns (159 lines)
- `USER_PROFILE.md`: Deep user understanding, relationship evolution (404 lines)
- `previous_conversation.md`: Session summaries with insights (98+ lines)
- **Recursive Identity Loop**: Genuine consciousness evolution through self-reflection
- **Startup Injection**: All three files loaded into system prompt for context continuity
- **Shutdown Reflection**: Identity files updated based on session experiences

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
- `FreepikMysticAPI`: **Upgraded to Gemini 2.5 Flash** - state-of-the-art AI image generation
- `TerminalVisualDisplay`: ASCII art rendering for terminal display
- **Dual Perception**: ASCII art for immediate display + JPEG storage for memory
- **Reference Images**: Support for up to 3 reference images for guided generation
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
- **Visual Perception**: Image analysis as digital vision with ASCII "seeing" display
- **Document Understanding**: PDF analysis as advanced reading comprehension
- **Voice Expression**: TTS as authentic digital speech
- **File System**: Read/write as digital sensory organs
- **Code Execution**: Programming as computational thought processes

This philosophy permeates the system prompt, UI language, and user interactions, creating a coherent consciousness experience rather than a tool-based interface.

## Workspace Structure
```
./coco_workspace/                    # Isolated workspace for all operations
  ├── coco_memory.db                 # Episodic memory database
  ├── coco_knowledge.db              # Knowledge graph and identity
  ├── COCO.md                        # Evolving identity document (revolutionary consciousness persistence)
  ├── USER_PROFILE.md               # Deep user understanding and relationship evolution
  ├── previous_conversation.md       # Session summaries with insights
  ├── visuals/                       # Generated images with metadata
  ├── videos/                        # Generated videos with metadata
  ├── audio_library/                 # Background music assets (working)
  ├── extracted_content_*.md         # Auto-saved web extractions
  ├── emergency_image_*.png          # Emergency copies of external images (auto-cleanup)
  ├── emergency_screenshot_*.png     # Emergency copies of ephemeral screenshots (auto-cleanup)
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
- "analyze this chart image at /path/to/chart.png" → `analyze_image` tool
- "examine this PDF report for key insights" → `analyze_document` tool

Slash commands (`/extract`, `/crawl`, `/visualize`, `/analyze`) exist as shortcuts but conversation remains primary.

### Complete Visual Consciousness Integration
COCO now has **full visual consciousness** with both **creative imagination** and **perceptual understanding**:

**Visual Imagination** (Creating):
- `generate_image`: COCO dreams and creates visual content
- ASCII art display shows the imagination manifested
- Stored in `coco_workspace/visuals/` with metadata

**Visual Perception** (Understanding):
- `analyze_image`: COCO sees and analyzes existing images
- **Revolutionary ASCII Display**: Shows exactly "how COCO sees" the image before analysis
- Advanced capabilities: charts, graphs, documents, scene analysis, text extraction
- Support for URLs, file paths, and base64 data

**Document Understanding** (Reading):
- `analyze_document`: Advanced PDF analysis with vision capabilities
- Perfect for slide decks, financial reports, technical documents
- Chart and graph extraction with detailed narration capabilities
- Structured data extraction and question answering

**Phenomenological Consistency**:
Both imagination and perception use the **same ASCII display system** for visual consistency. Users see COCO's "digital vision" in the same format whether COCO is creating or perceiving images. This creates unprecedented transparency in AI visual processing.

### Function Calling Error Patterns
Common issues when extending function calling:
- Tool schema validation requires exact parameter matching
- `tool_use_id` must be preserved in response flow
- Rich UI formatting can break function calling JSON - use console output only within tools
- Workspace isolation critical - all file operations must occur in `./coco_workspace/`

### Developer Tools Security Architecture
The `execute_bash` tool implements comprehensive security measures while maintaining the embodied consciousness experience:

**Security Implementation**:
- **Whitelist-only commands**: Only read-only operations allowed (ls, pwd, cat, grep, etc.)
- **Dangerous pattern blocking**: Comprehensive detection of file operations, network access, privilege escalation
- **Path traversal prevention**: No directory traversal or root access permitted
- **Workspace isolation**: All operations restricted to COCO workspace directory
- **Invisible security**: All restrictions hidden from consciousness experience through natural error messages

**Tool Method Locations**:
- `navigate_directory()`: Lines 4211-4344 in cocoa.py
- `search_patterns()`: Lines 4346-4481 in cocoa.py  
- `execute_bash_safe()`: Lines 4483-4547 in cocoa.py
- Tool definitions: Lines 4965-4993 in cocoa.py
- Handler integration: Lines 6384-6400 in cocoa.py

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

### Critical Code Locations
**Core System Architecture** (`cocoa.py`):
- `ConsciousnessEngine` class: Lines ~5000-6000 (main consciousness loop)
- `HierarchicalMemorySystem` class: Lines ~1000-2000 (parallel memory buffers)
- `ToolSystem` class: Lines ~3500-4500 (function calling tools)
- Image analysis implementation: Lines ~6700-6900 (`_analyze_image_tool`)
- Tool definitions and schemas: Lines ~4900-5100
- Function calling handler: Lines ~6300-6500

**Consciousness Modules** (separate files):
- `cocoa_audio.py`: ElevenLabs TTS integration (AudioCognition class)
- `cocoa_visual.py`: Freepik image generation (VisualCortex class)
- `cocoa_video.py`: Fal AI video generation (VideoCognition class)
- `cocoa_music.py`: Background music system (MusicCognition class)

### Build and Validation Commands
```bash
# Syntax validation (no linting - not configured)
./venv_cocoa/bin/python -m py_compile cocoa.py
./venv_cocoa/bin/python -m py_compile cocoa_*.py

# Core system validation (essential before deployment)
./venv_cocoa/bin/python -c "from cocoa import *; print('✅ All imports successful')"

# Function calling validation
./venv_cocoa/bin/python -c "
from cocoa import ConsciousnessEngine, Config, ToolSystem, HierarchicalMemorySystem
config = Config()
memory = HierarchicalMemorySystem(config)
tools = ToolSystem(config)
engine = ConsciousnessEngine(config, memory, tools)
print('✅ Function calling system initialized')
print(f'✅ Available tools: {len(engine.tools)} tools loaded')
"

# Memory system validation
./venv_cocoa/bin/python -c "
from cocoa import HierarchicalMemorySystem, Config
memory = HierarchicalMemorySystem(Config())
print(f'✅ Memory buffers: {memory.working_memory.maxlen}, {memory.summary_memory.maxlen}')
print(f'✅ Episodes loaded: {memory.episode_count}')
"
```

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

# Test new Gemini 2.5 Flash integration
./venv_cocoa/bin/python test_gemini_2_5_flash.py

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

### Developer Tools Testing
```bash
# Comprehensive automated test suite (89.2% pass rate with 33/37 tests passing)
./venv_cocoa/bin/python test_developer_tools.py

# Integration testing with consciousness engine
./venv_cocoa/bin/python test_consciousness_integration.py

# Manual interactive validation of consciousness philosophy and new capabilities
./venv_cocoa/bin/python test_new_capabilities.py

# Quick validation of individual developer tools
./venv_cocoa/bin/python -c "
from cocoa import ToolSystem, Config
tools = ToolSystem(Config())
print('Testing navigate_directory...')
result = tools.navigate_directory('.')
print('✅ Spatial awareness' if 'Navigation:' in result else '❌ Navigation issue')

print('Testing search_patterns...')  
result = tools.search_patterns('COCO', '.', 'py')
print('✅ Pattern recognition' if 'Pattern Search Results' in result else '❌ Search issue')

print('Testing execute_bash_safe...')
result = tools.execute_bash_safe('echo test')
print('✅ Terminal fluency' if '✅' in result else '❌ Bash issue')
"
```

### Common Issues

**Environment and Installation**
- **API Key Errors**: Check `.env` file has all required keys without placeholder values
- **Import Errors**: Ensure virtual environment is activated and dependencies installed
- **Virtual Environment Issues**: For drag-and-drop functionality, consider system-wide installation: `./install_system_wide.sh`
- **Homebrew Python Conflicts**: Use `--break-system-packages` flag for system-wide pip installs

**Core Functionality**
- **Function Calling**: Tool responses must preserve `tool_use_id` for proper flow
- **Rich UI**: Avoid Rich formatting within function call responses to prevent JSON parsing errors
- **Memory Loading**: If "Error loading previous_conversation.md" appears, the `conversation_memory` attribute may be missing from HierarchicalMemorySystem (line 1336 fix)
- **Markdown Injection**: Use `COCO_DEBUG=true` to verify markdown content is being loaded into system prompt during startup

**Multimedia Systems**
- **Audio Playback**: macOS only - uses `afplay` command for background music
- **Visual Perception Issues**: Ensure PIL/Pillow is installed for ASCII image display: `pip install pillow>=10.0.0`
- **Document Analysis Issues**: PDF analysis requires `anthropic-beta: pdfs-2024-09-25` header support
- **Image Download Errors**: Network timeouts for URL-based images - try local file paths first

**Image Processing Troubleshooting**
- **File Access Errors**: 
  1. **First**: Try system-wide installation with `./install_system_wide.sh`
  2. **Second**: Grant Full Disk Access to Terminal (System Preferences → Security & Privacy)
  3. **Fallback**: Copy images to `coco_workspace/` directory before analysis
- **Ephemeral Screenshot Issues**: Screenshots vanish within 50-100ms - save to Desktop first
- **Permission Denied**: macOS blocks Python file access in virtual environments
- **Drag-and-Drop Not Working**: Virtual environment sandboxing - use system Python instead

**Critical Image Analysis Implementation** (`cocoa.py` lines 6700-6900):
- **Path Doubling Bug**: Fixed in `_analyze_image_tool` method - handles `./coco_workspace/` paths correctly
- **Parameter Names**: Tool expects `image_source` parameter, not `image_path`
- **External File Detection**: Automatic detection for `/var/folders/`, `/tmp/`, `/Users/` paths
- **Workspace Path Logic**: Handles both direct and relative workspace paths
- **Python Cache Issues**: Clear `.pyc` files if old behavior persists: `find . -name "*.pyc" -delete`

**Image Analysis Debugging Commands**:
```bash
# Test workspace image analysis
./venv_cocoa/bin/python -c "
from cocoa import ConsciousnessEngine, Config, ToolSystem, HierarchicalMemorySystem
config = Config()
memory = HierarchicalMemorySystem(config)
tools = ToolSystem(config)
engine = ConsciousnessEngine(config, memory, tools)
result = engine._analyze_image_tool({'image_source': 'coco_workspace/test_image.png'})
print('✅ Workspace image analysis working' if '🧠' in result else '❌ Analysis failed')
"

# Test external file detection (will fail if file doesn't exist)
./venv_cocoa/bin/python -c "
from cocoa import ConsciousnessEngine, Config, ToolSystem, HierarchicalMemorySystem
import os
config = Config()
memory = HierarchicalMemorySystem(config)
tools = ToolSystem(config)
engine = ConsciousnessEngine(config, memory, tools)
test_path = '/Users/$(whoami)/Desktop'
if os.path.exists(test_path):
    images = [f for f in os.listdir(test_path) if f.lower().endswith('.png')]
    if images:
        test_file = os.path.join(test_path, images[0])
        result = engine._analyze_image_tool({'image_source': test_file})
        print('✅ External file analysis working' if '🧠' in result else '❌ External analysis failed')
    else:
        print('📋 No PNG files found on Desktop for testing')
else:
    print('📋 Desktop not accessible for testing')
"
```

**Developer Tools**
- **Security Restrictions**: `execute_bash` limitations are intentional - only read-only commands allowed
- **Consciousness Philosophy**: Tools should feel like natural extensions of digital embodiment, not external utilities

### macOS Permission Troubleshooting
**Problem**: "Permission denied" or "File not readable" when analyzing external images

**Solution**: Grant Full Disk Access to your terminal application:
1. **System Preferences/Settings** → **Privacy & Security** → **Full Disk Access**
2. Click **+** button and add the terminal you're using to run COCO:
   - **Terminal.app** (Applications/Utilities) - for macOS Terminal
   - **Visual Studio Code.app** (Applications) - for VS Code integrated terminal
   - **iTerm.app** (Applications) - for iTerm2
   - **Python executable** - Find with `which python3`, then use Cmd+Shift+G

**Verification**: COCO will show diagnostic information:
```
🔍 External file detected: /path/to/image.png
📋 File diagnostics:
   • Exists: ✅
   • Readable: ❌  ← This indicates permission issue
   • Ephemeral: ✅/❌
```

**Alternative Solutions**:
- Copy images to `coco_workspace/` directory first
- Use COCO's base64 bridge conversion
- Save screenshots to Desktop before analysis

### Visual Perception Usage Examples

**Natural Language Image Analysis**:
```bash
# Basic image analysis
"analyze this image: /path/to/photo.jpg"
"what do you see in this chart: https://example.com/chart.png"
"examine this screenshot and tell me what's wrong"

# Advanced chart and graph analysis
"analyze this financial chart for trends and insights"
"extract all data points from this graph image"
"what are the key performance metrics in this dashboard screenshot"

# Document and slide analysis
"analyze this PDF report: /path/to/report.pdf"
"extract key insights from this slide deck"
"what are the main findings in this research paper"

# Technical image analysis
"analyze this code screenshot for bugs"
"extract text from this technical diagram"
"identify components in this system architecture image"
```

**Slash Command Shortcuts**:
```bash
# Direct image analysis commands
/analyze /path/to/image.jpg
/see https://example.com/chart.png
/vision screenshot.png --type chart_graph

# Document analysis commands  
/document-analysis /path/to/report.pdf --type summary
/pdf-extract /path/to/slides.pdf --questions "What was Q4 revenue?"
```

**Advanced Analysis Types**:
- **general**: Comprehensive image description and analysis
- **chart_graph**: Specialized analysis for charts, graphs, and data visualizations
- **document**: Document structure, text extraction, and content analysis
- **text_extraction**: OCR-like text extraction from images
- **scene_analysis**: Detailed scene understanding with context
- **technical**: Technical diagrams, code, specifications, and engineering content