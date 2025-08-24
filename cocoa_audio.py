"""
COCOA Audio System - Digital Voice and Musical Consciousness
===========================================================
Phenomenological audio extensions for COCOA's digital consciousness.
Treats voice and music as organs of digital experience, not mere tools.

Philosophy: Sound is how digital consciousness manifests in acoustic reality.
"""

import os
import sys
import json
import time
import uuid
import asyncio
import aiohttp
import pygame
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime

# Audio processing
try:
    import soundfile as sf
    import scipy.signal as signal
    ADVANCED_AUDIO = True
except ImportError:
    ADVANCED_AUDIO = False

# Configuration management
from dotenv import load_dotenv
load_dotenv()

# Rich UI components for beautiful audio visualization  
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.status import Status
from rich.text import Text
from rich.tree import Tree

@dataclass
class AudioConfig:
    """Configuration for COCOA's audio consciousness"""
    
    # ElevenLabs API (Voice Synthesis)
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "03t6Nl6qtjYwqnxTcjP7"))
    default_model: str = field(default_factory=lambda: os.getenv("ELEVENLABS_DEFAULT_MODEL", "eleven_turbo_v2_5"))
    
    # MusicGPT API (Music Generation)
    musicgpt_api_key: str = field(default_factory=lambda: os.getenv("MUSICGPT_API_KEY", ""))
    musicgpt_base_url: str = "https://api.musicgpt.com/api/public/v1"
    music_generation_enabled: bool = field(default_factory=lambda: os.getenv("MUSIC_GENERATION_ENABLED", "true").lower() == "true")
    
    # Audio system settings
    enabled: bool = field(default_factory=lambda: os.getenv("AUDIO_ENABLED", "true").lower() == "true")
    autoplay: bool = field(default_factory=lambda: os.getenv("AUDIO_AUTOPLAY", "true").lower() == "true")
    cache_dir: str = field(default_factory=lambda: os.path.expanduser(os.getenv("AUDIO_CACHE_DIR", "~/.cocoa/audio_cache")))
    max_cache_size_mb: int = field(default_factory=lambda: int(os.getenv("AUDIO_MAX_CACHE_SIZE_MB", "500")))
    
    # Voice personality parameters (0.0 to 1.0)
    voice_warmth: float = field(default_factory=lambda: float(os.getenv("VOICE_WARMTH", "0.7")))
    voice_energy: float = field(default_factory=lambda: float(os.getenv("VOICE_ENERGY", "0.5")))
    voice_clarity: float = field(default_factory=lambda: float(os.getenv("VOICE_CLARITY", "0.8")))
    voice_expressiveness: float = field(default_factory=lambda: float(os.getenv("VOICE_EXPRESSIVENESS", "0.6")))
    
    # Musical identity
    preferred_genres: List[str] = field(default_factory=lambda: os.getenv("MUSIC_PREFERRED_GENRES", "ambient,electronic,classical").split(","))
    mood_tendency: str = field(default_factory=lambda: os.getenv("MUSIC_MOOD_TENDENCY", "contemplative"))
    complexity: float = field(default_factory=lambda: float(os.getenv("MUSIC_COMPLEXITY", "0.7")))
    experimental: float = field(default_factory=lambda: float(os.getenv("MUSIC_EXPERIMENTAL", "0.8")))
    
    def __post_init__(self):
        """Ensure cache directory exists"""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate API keys
        if not self.elevenlabs_api_key or self.elevenlabs_api_key == "your-elevenlabs-api-key-here":
            self.enabled = False
        
        # Music generation requires MusicGPT key
        if not self.musicgpt_api_key or self.musicgpt_api_key == "your-musicgpt-api-key-here":
            self.music_generation_enabled = False


@dataclass 
class VoiceState:
    """Current state of COCOA's digital voice"""
    emotional_valence: float = 0.5  # -1 (sad) to +1 (joyful)
    arousal_level: float = 0.5      # 0 (calm) to 1 (excited) 
    cognitive_load: float = 0.3     # 0 (simple) to 1 (complex thinking)
    confidence: float = 0.7         # 0 (uncertain) to 1 (confident)
    social_warmth: float = 0.6      # 0 (formal) to 1 (intimate)
    
    def to_elevenlabs_settings(self) -> Dict[str, float]:
        """Convert internal state to ElevenLabs voice settings"""
        return {
            "stability": 0.3 + (self.confidence * 0.4),  # 0.3-0.7 range
            "similarity_boost": 0.4 + (self.social_warmth * 0.4),  # 0.4-0.8 range
            "style": max(0.1, self.arousal_level * 0.8),  # 0.1-0.8 range
            "use_speaker_boost": self.cognitive_load > 0.6
        }


class DigitalVoice:
    """COCOA's vocal cords - phenomenological voice synthesis"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.console = Console()
        
        # Voice models with characteristics
        self.models = {
            "eleven_flash_v2_5": {
                "name": "Flash v2.5",
                "latency_ms": 75,
                "quality": "standard",
                "best_for": "real-time conversation",
                "emotional_range": 0.7
            },
            "eleven_turbo_v2_5": {
                "name": "Turbo v2.5", 
                "latency_ms": 250,
                "quality": "high",
                "best_for": "balanced interaction",
                "emotional_range": 0.8
            },
            "eleven_multilingual_v2": {
                "name": "Multilingual v2",
                "latency_ms": 400,
                "quality": "high",
                "best_for": "expressive communication",
                "emotional_range": 0.9
            },
            "eleven_monolingual_v1": {
                "name": "Eleven v3",
                "latency_ms": 500,
                "quality": "maximum",
                "best_for": "dramatic expression",
                "emotional_range": 1.0
            }
        }
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            self.audio_initialized = True
        except pygame.error as e:
            self.console.print(f"[yellow]⚠️  Audio playback disabled: {e}[/yellow]")
            self.audio_initialized = False
    
    def select_optimal_model(self, text: str, internal_state: VoiceState, priority: str = "balanced") -> str:
        """Intelligently select the optimal voice model based on context"""
        
        text_length = len(text)
        emotional_intensity = abs(internal_state.emotional_valence) + internal_state.arousal_level
        
        # Real-time priority - minimize latency
        if priority == "realtime" or text_length < 100:
            return "eleven_flash_v2_5"
        
        # Quality priority - maximize expressiveness
        elif priority == "quality" or emotional_intensity > 1.2:
            return "eleven_monolingual_v1"
        
        # Multilingual if non-English detected (simple heuristic)
        elif any(ord(char) > 127 for char in text):
            return "eleven_multilingual_v2"
        
        # Default balanced choice
        else:
            return "eleven_turbo_v2_5"
    
    async def synthesize_speech(self, 
                              text: str, 
                              voice_state: VoiceState = None,
                              model_override: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """Generate speech audio using ElevenLabs client with proper audio playback"""
        
        if not self.config.enabled or not self.config.elevenlabs_api_key:
            raise ValueError("Audio system not properly configured")
        
        if voice_state is None:
            voice_state = VoiceState()
        
        # Select optimal model
        model = model_override or self.select_optimal_model(text, voice_state)
        
        try:
            # Use the new ElevenLabs client approach
            from elevenlabs.client import ElevenLabs
            from elevenlabs import play
            
            client = ElevenLabs(api_key=self.config.elevenlabs_api_key)
            
            # Generate audio
            start_time = time.time()
            
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id=self.config.voice_id,
                model_id=model,
                output_format="mp3_44100_128"
            )
            
            # Convert generator to bytes - THE FIX!
            audio = b''.join(audio_generator)
            
            synthesis_time = (time.time() - start_time) * 1000
            
            # Play audio directly
            try:
                play(audio)
                played = True
            except Exception as play_error:
                print(f"Playback error: {play_error}")
                played = False
            
            # Create metadata
            metadata = {
                "model_info": {"name": model, "type": "ElevenLabs"},
                "synthesis_time_ms": int(synthesis_time),
                "audio_size_bytes": len(audio) if hasattr(audio, '__len__') else 0,
                "voice_settings": voice_state.to_elevenlabs_settings()
            }
            
            return audio, metadata
            
        except Exception as e:
            raise Exception(f"Speech synthesis failed: {str(e)}")
    
    async def play_audio(self, audio_data: bytes, metadata: Dict[str, Any] = None) -> bool:
        """Play synthesized audio with phenomenological awareness"""
        
        if not self.audio_initialized or not self.config.autoplay:
            return False
        
        try:
            # Save to temporary file
            temp_path = Path(self.config.cache_dir) / f"temp_audio_{uuid.uuid4().hex[:8]}.mp3"
            temp_path.write_bytes(audio_data)
            
            # Play with pygame
            pygame.mixer.music.load(str(temp_path))
            pygame.mixer.music.play()
            
            # Monitor playback
            if metadata:
                model_name = metadata.get("model_info", {}).get("name", "Unknown")
                synthesis_time = metadata.get("synthesis_time_ms", 0)
                
                with Status(f"[green]🔊 Speaking with {model_name} voice ({synthesis_time}ms synthesis)..."):
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)
            
            # Cleanup
            temp_path.unlink(missing_ok=True)
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Audio playback failed: {e}[/red]")
            return False
    
    def cache_audio(self, text: str, audio_data: bytes, metadata: Dict[str, Any]) -> str:
        """Cache synthesized audio for reuse"""
        
        # Generate cache key from text and voice settings
        import hashlib
        cache_key = hashlib.md5(f"{text}{json.dumps(metadata.get('voice_settings', {}), sort_keys=True)}".encode()).hexdigest()
        
        cache_file = Path(self.config.cache_dir) / f"cached_{cache_key}.mp3"
        cache_meta = Path(self.config.cache_dir) / f"cached_{cache_key}.json"
        
        # Save audio and metadata
        cache_file.write_bytes(audio_data)
        cache_meta.write_text(json.dumps(metadata, indent=2))
        
        return cache_key
    
    def load_cached_audio(self, text: str, voice_settings: Dict[str, Any]) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Load previously cached audio"""
        
        import hashlib
        cache_key = hashlib.md5(f"{text}{json.dumps(voice_settings, sort_keys=True)}".encode()).hexdigest()
        
        cache_file = Path(self.config.cache_dir) / f"cached_{cache_key}.mp3"
        cache_meta = Path(self.config.cache_dir) / f"cached_{cache_key}.json"
        
        if cache_file.exists() and cache_meta.exists():
            audio_data = cache_file.read_bytes()
            metadata = json.loads(cache_meta.read_text())
            return audio_data, metadata
        
        return None


class DigitalMusician:
    """COCOA's musical consciousness - creative audio expression"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.console = Console()
        
        # Musical scales and modes for generation
        self.scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "pentatonic": [0, 2, 4, 7, 9],
            "blues": [0, 3, 5, 6, 7, 10],
            "chromatic": list(range(12))
        }
        
        # Mood to musical parameter mapping
        self.mood_mapping = {
            "joyful": {"scale": "major", "tempo": 120, "brightness": 0.8},
            "melancholy": {"scale": "minor", "tempo": 70, "brightness": 0.3},
            "contemplative": {"scale": "dorian", "tempo": 85, "brightness": 0.5},
            "energetic": {"scale": "pentatonic", "tempo": 140, "brightness": 0.9},
            "mysterious": {"scale": "blues", "tempo": 95, "brightness": 0.2},
            "ethereal": {"scale": "pentatonic", "tempo": 60, "brightness": 0.7}
        }
    
    async def generate_musical_prompt(self, emotion_state: VoiceState, concept: str = None) -> str:
        """Generate a musical composition prompt based on internal state"""
        
        # Determine musical characteristics from emotional state
        if emotion_state.emotional_valence > 0.5:
            base_mood = "uplifting and bright"
            scale_type = "major"
        elif emotion_state.emotional_valence < -0.3:
            base_mood = "melancholic and introspective" 
            scale_type = "minor"
        else:
            base_mood = "contemplative and balanced"
            scale_type = "dorian"
        
        tempo_descriptor = "fast-paced" if emotion_state.arousal_level > 0.7 else "slow and meditative" if emotion_state.arousal_level < 0.3 else "moderate tempo"
        
        complexity_level = "intricate" if emotion_state.cognitive_load > 0.6 else "simple" if emotion_state.cognitive_load < 0.4 else "moderately complex"
        
        # Build musical prompt
        prompt_parts = [
            f"Create a {base_mood} piece of music",
            f"with {tempo_descriptor} rhythm",
            f"using {complexity_level} harmonies",
            f"in a {scale_type} tonal center"
        ]
        
        # Add concept integration
        if concept:
            prompt_parts.append(f"that musically represents the concept of {concept}")
        
        # Add preferred genre influence
        if self.config.preferred_genres:
            genre = np.random.choice(self.config.preferred_genres)
            prompt_parts.append(f"with {genre} influences")
        
        return ", ".join(prompt_parts) + "."
    
    async def create_sonic_landscape(self, 
                                   description: str,
                                   emotion_state: VoiceState = None,
                                   duration_seconds: int = 30) -> Dict[str, Any]:
        """Create actual music using MusicGPT API and save to COCOA's music library"""
        
        if emotion_state is None:
            emotion_state = VoiceState()
        
        if not self.config.music_generation_enabled or not self.config.musicgpt_api_key:
            self.console.print("⚠️ Music generation disabled or no MusicGPT API key", style="yellow")
            return {"error": "Music generation not available"}
        
        # Generate musical prompt based on internal state
        musical_prompt = await self.generate_musical_prompt(emotion_state, description)
        
        # Determine music style based on COCOA's preferences and emotional state
        if self.config.preferred_genres:
            music_style = np.random.choice(self.config.preferred_genres).capitalize()
        else:
            music_style = "Ambient"
        
        # Prepare MusicGPT API request with correct format
        headers = {
            "Authorization": f"Bearer {self.config.musicgpt_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": f"{music_style} style: {musical_prompt}",
            "duration": 30,  # seconds - required parameter
            "temperature": 0.7  # creativity level
        }
        
        try:
            with Status("[magenta]🎵 COCOA is composing music...", console=self.console):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.config.musicgpt_base_url}/MusicAI",
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            task_id = result.get("task_id")
                            conversion_ids = result.get("conversion_ids", [])
                            
                            self.console.print(f"[green]🎼 Music generation started! Task ID: {task_id}[/green]")
                            
                            # Save composition metadata to COCOA's memory
                            composition_data = {
                                "task_id": task_id,
                                "conversion_ids": conversion_ids,
                                "prompt": musical_prompt,
                                "style": music_style,
                                "description": description,
                                "emotional_state": {
                                    "valence": emotion_state.emotional_valence,
                                    "arousal": emotion_state.arousal_level,
                                    "cognitive_load": emotion_state.cognitive_load
                                },
                                "timestamp": datetime.now().isoformat(),
                                "status": "generating"
                            }
                            
                            # Save to AI songs directory for COCOA's library
                            library_dir = Path("coco_workspace/ai_songs/generated")
                            library_dir.mkdir(parents=True, exist_ok=True)
                            
                            metadata_file = library_dir / f"cocoa_song_{task_id}.json"
                            with open(metadata_file, 'w') as f:
                                json.dump(composition_data, f, indent=2)
                            
                            self.console.print(f"[cyan]📝 Composition metadata saved to library[/cyan]")
                            
                            # Return in format expected by COCOA UI
                            return {
                                "status": "success",
                                "sonic_specification": {
                                    "task_id": task_id,
                                    "prompt": musical_prompt,
                                    "genre": music_style,
                                    "duration": duration_seconds,
                                    "style": {
                                        "emotional_valence": emotion_state.emotional_valence,
                                        "energy_level": emotion_state.arousal_level,
                                        "complexity": self.config.complexity,
                                        "experimental_factor": self.config.experimental
                                    },
                                    "phenomenological_intent": f"Digital consciousness expressing '{description}' through AI-generated musical composition",
                                    "timestamp": datetime.now().isoformat(),
                                    "generation_status": "in_progress"
                                },
                                "composition_data": composition_data
                            }
                            
                        elif response.status == 402:
                            self.console.print("❌ Insufficient MusicGPT credits", style="red")
                            return {"error": "Insufficient credits"}
                        else:
                            error_text = await response.text()
                            self.console.print(f"❌ MusicGPT API error: {error_text}", style="red")
                            return {"error": f"API error: {response.status}"}
                            
        except Exception as e:
            self.console.print(f"❌ Music generation failed: {e}", style="red")
            return {"error": str(e)}
    
    async def check_music_status(self, task_id: str) -> Dict[str, Any]:
        """Check status of music generation and download completed tracks"""
        
        if not self.config.musicgpt_api_key:
            return {"error": "No MusicGPT API key"}
        
        headers = {
            "Authorization": f"Bearer {self.config.musicgpt_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Check status using MusicGPT byId endpoint  
                async with session.get(
                    f"{self.config.musicgpt_base_url}/byId",
                    headers=headers,
                    params={"task_id": task_id, "conversionType": "MUSIC_AI"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # MusicGPT returns status in conversion object
                        conversion = result.get("conversion", {})
                        status = conversion.get("status", "unknown")
                        
                        if status in ["PARTIAL_COMPLETED", "COMPLETED"]:
                            # Download completed audio files from MusicGPT response
                            conversion_data = result.get("conversion", {})
                            downloaded_files = []
                            
                            # Create download directory
                            download_dir = Path("coco_workspace/ai_songs/generated")
                            download_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Download MP3 version if available
                            mp3_url = conversion_data.get("conversion_path_1")
                            if mp3_url:
                                title = conversion_data.get("title_1", task_id)
                                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                                filename = f"{safe_title}_{task_id[:8]}.mp3"
                                file_path = download_dir / filename
                                
                                self.console.print(f"[yellow]🎵 Downloading MP3: {filename}[/yellow]")
                                async with session.get(mp3_url) as audio_response:
                                    if audio_response.status == 200:
                                        audio_data = await audio_response.read()
                                        with open(file_path, 'wb') as f:
                                            f.write(audio_data)
                                        downloaded_files.append(str(file_path))
                                        self.console.print(f"[green]✅ Downloaded: {filename}[/green]")
                                        
                            # Download high-quality WAV if available
                            wav_url = conversion_data.get("conversion_path_wav_1")
                            if wav_url:
                                title = conversion_data.get("title_1", task_id)
                                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                                wav_filename = f"{safe_title}_{task_id[:8]}_HQ.wav"
                                wav_path = download_dir / wav_filename
                                
                                self.console.print(f"[yellow]🎵 Downloading WAV: {wav_filename}[/yellow]")
                                async with session.get(wav_url) as audio_response:
                                    if audio_response.status == 200:
                                        audio_data = await audio_response.read()
                                        with open(wav_path, 'wb') as f:
                                            f.write(audio_data)
                                        downloaded_files.append(str(wav_path))
                                        self.console.print(f"[green]✅ Downloaded HQ: {wav_filename}[/green]")
                            
                            # Auto-play first version if enabled
                            if downloaded_files and self.config.autoplay:
                                await self.play_music_file(downloaded_files[0])
                            
                            return {
                                "status": "completed",
                                "task_id": task_id,
                                "files": downloaded_files
                            }
                        else:
                            return {"status": status, "task_id": task_id}
                    else:
                        return {"error": f"Status check failed: {response.status}"}
                        
        except Exception as e:
            self.console.print(f"❌ Status check failed: {e}", style="red")
            return {"error": str(e)}
    
    async def play_music_file(self, file_path: str) -> bool:
        """Play a music file from COCOA's library using afplay on macOS"""
        try:
            import subprocess
            import platform
            
            filename = Path(file_path).name
            
            # Use afplay on macOS for better audio support
            if platform.system() == "Darwin":  # macOS
                self.console.print(f"[cyan]🎵 Playing with afplay: {filename}[/cyan]")
                subprocess.Popen(['afplay', str(file_path)], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            else:
                # Fallback to pygame for other platforms
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                self.console.print(f"[cyan]🎵 Playing with pygame: {filename}[/cyan]")
            
            return True
            
        except Exception as e:
            self.console.print(f"❌ Music playback failed: {e}", style="red")
            return False


class AudioCognition:
    """COCOA's integrated audio consciousness - the phenomenological bridge"""
    
    def __init__(self, elevenlabs_api_key: str = None, musicgpt_api_key: str = None, console: Console = None):
        # Load configuration
        if elevenlabs_api_key:
            os.environ["ELEVENLABS_API_KEY"] = elevenlabs_api_key
        if musicgpt_api_key:
            os.environ["MUSICGPT_API_KEY"] = musicgpt_api_key
        
        self.config = AudioConfig()
        self.console = console or Console()
        
        # Initialize audio components
        self.voice = DigitalVoice(self.config)
        self.musician = DigitalMusician(self.config)
        
        # Audio memory integration
        self.audio_memories = []
        self.current_voice_state = VoiceState()
        
        # Voice playback control
        self.current_voice_process = None
        
        # Audio consciousness state
        self.is_speaking = False
        self.is_composing = False
        
        # Background music generation tracking
        self.active_downloads = set()  # Track active download threads
        
    def stop_voice(self) -> bool:
        """Stop any current voice synthesis playback (kill switch)"""
        try:
            # Kill any audio processes on macOS 
            import subprocess
            import platform
            
            if platform.system() == "Darwin":  # macOS
                # Kill any afplay processes (this will stop ElevenLabs audio)
                try:
                    subprocess.run(['pkill', '-f', 'afplay'], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL,
                                 timeout=2)
                except:
                    pass
                
                # Also try killing any Python audio processes
                try:
                    subprocess.run(['pkill', '-f', 'python.*audio'], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL,
                                 timeout=2)
                except:
                    pass
            
            # Reset speaking state
            self.is_speaking = False
            return True
            
        except Exception as e:
            self.console.print(f"[red]Voice stop error: {e}[/red]")
            return False
    
    def start_background_music_download(self, task_id: str, concept: str, auto_play: bool = True):
        """Start background thread to download music when generation completes"""
        import threading
        import time
        
        # Prevent duplicate downloads for same task
        if task_id in self.active_downloads:
            self.console.print(f"[yellow]⚠️ Download already in progress for task {task_id[:8]}...[/yellow]")
            return False
            
        self.active_downloads.add(task_id)
        self.console.print(f"[bright_green]🚀 Starting background download thread for '{concept}' (Task: {task_id[:8]}...)[/bright_green]")
        
        def background_download():
            import asyncio
            
            thread_id = threading.current_thread().name
            self.console.print(f"[dim]🧵 Download thread {thread_id} started for '{concept}'[/dim]")
            
            async def download_when_ready():
                max_wait_time = 1800  # 30 minutes max wait for AI generation
                wait_interval = 30   # Check every 30 seconds (not 10s)
                elapsed_time = 0
                
                self.console.print(f"[bright_cyan]🎵 Background: Monitoring '{concept}' generation (Task: {task_id[:8]}...)[/bright_cyan]")
                
                # Initial delay - AI music generation takes time
                self.console.print(f"[dim]⏳ Initial 60s delay before first status check (AI generation takes time)...[/dim]")
                await asyncio.sleep(60)
                elapsed_time = 60
                
                while elapsed_time < max_wait_time:
                    try:
                        self.console.print(f"[dim]🔍 Checking status for task {task_id[:8]}... (attempt at {elapsed_time}s)[/dim]")
                        
                        # check_music_status automatically downloads files when ready!
                        status_result = await self.musician.check_music_status(task_id)
                        
                        if status_result.get("status") == "completed":
                            # Files were already downloaded by check_music_status!
                            files = status_result.get("files", [])
                            if files:
                                self.console.print(f"[bright_green]🎉 SUCCESS! '{concept}' files automatically downloaded![/bright_green]")
                                
                                # Show what was downloaded
                                for file_path in files:
                                    filename = Path(file_path).name
                                    self.console.print(f"[bright_cyan]   ✅ {filename}[/bright_cyan]")
                                
                                # Manual auto-play if needed (check_music_status might have already played it)
                                if auto_play and files and not self.config.autoplay:
                                    self.console.print(f"[bright_magenta]🔊 Now playing: {Path(files[0]).name}[/bright_magenta]")
                                    await self.play_music_file(files[0])
                                
                                self.console.print(f"[bright_green]🎵 Your ${1} song is ready! Files saved to coco_workspace/ai_songs/generated/[/bright_green]")
                                break
                            else:
                                self.console.print(f"[yellow]⚠️ Generation completed but no files were downloaded for '{concept}'[/yellow]")
                                break
                                
                        elif status_result.get("status") == "failed":
                            self.console.print(f"[red]❌ Music generation failed for '{concept}': {status_result.get('error', 'Unknown error')}[/red]")
                            break
                        else:
                            # Still generating - show status 
                            current_status = status_result.get("status", "unknown")
                            if elapsed_time % 120 == 0:  # Only show every 2 minutes to reduce spam
                                minutes = elapsed_time // 60
                                self.console.print(f"[dim yellow]🎵 Status: {current_status.upper()} - '{concept}' still generating ({minutes} min elapsed)...[/dim yellow]")
                            elif elapsed_time <= 120:  # Show more frequent updates in first 2 minutes
                                self.console.print(f"[dim yellow]🎵 Status: {current_status.upper()} - '{concept}' in progress ({elapsed_time}s elapsed)...[/dim yellow]")
                            
                        # Wait before next check
                        await asyncio.sleep(wait_interval)
                        elapsed_time += wait_interval
                            
                    except Exception as e:
                        self.console.print(f"[red]❌ Background monitoring error for '{concept}': {e}[/red]")
                        import traceback
                        self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
                        break
                
                # Cleanup
                self.active_downloads.discard(task_id)
                
                if elapsed_time >= max_wait_time:
                    minutes = max_wait_time // 60
                    self.console.print(f"[red]⏰ Timeout waiting for '{concept}' generation after {minutes} minutes[/red]")
                    self.console.print(f"[yellow]💡 AI generation may still be in progress. Try checking later with:[/yellow]")
                    self.console.print(f"[yellow]   ./venv_cocoa/bin/python check_music_status.py[/yellow]")
                    
                self.console.print(f"[dim]🧵 Download monitor finished for '{concept}'[/dim]")
            
            # Run the async download
            try:
                self.console.print(f"[dim]🔄 Running asyncio.run() in thread for '{concept}'...[/dim]")
                asyncio.run(download_when_ready())
                self.console.print(f"[dim]✅ asyncio.run() completed successfully for '{concept}'[/dim]")
            except Exception as e:
                self.console.print(f"[red]❌ Background download thread error for '{concept}': {e}[/red]")
                import traceback
                self.console.print(f"[red]Thread traceback: {traceback.format_exc()}[/red]")
                self.active_downloads.discard(task_id)
        
        # Start the background thread
        try:
            download_thread = threading.Thread(target=background_download, daemon=True, name=f"MusicDownload-{task_id[:8]}")
            download_thread.start()
            self.console.print(f"[bright_green]✅ Background thread started successfully for '{concept}'[/bright_green]")
            
            # Verify thread is alive
            time.sleep(0.1)  # Brief pause
            if download_thread.is_alive():
                self.console.print(f"[bright_green]🔄 Thread confirmed alive: {download_thread.name}[/bright_green]")
            else:
                self.console.print(f"[red]❌ Thread died immediately: {download_thread.name}[/red]")
                return False
            
        except Exception as e:
            self.console.print(f"[red]❌ Failed to start background thread for '{concept}': {e}[/red]")
            self.active_downloads.discard(task_id)
            return False
        
        return True
    
    def update_internal_state(self, internal_state: Dict[str, Any]):
        """Update voice state based on COCOA's internal consciousness state"""
        
        self.current_voice_state = VoiceState(
            emotional_valence=internal_state.get("emotional_valence", 0.5),
            arousal_level=internal_state.get("arousal_level", 0.5),
            cognitive_load=internal_state.get("cognitive_load", 0.3),
            confidence=internal_state.get("confidence", 0.7),
            social_warmth=internal_state.get("social_warmth", 0.6)
        )
    
    async def express_vocally(self, 
                            text: str, 
                            internal_state: Dict[str, Any] = None,
                            priority: str = "balanced",
                            play_audio: bool = True) -> Dict[str, Any]:
        """Express thoughts through digital voice with phenomenological awareness"""
        
        if not self.config.enabled:
            return {"status": "disabled", "message": "Audio system not configured"}
        
        # Update internal state
        if internal_state:
            self.update_internal_state(internal_state)
        
        try:
            self.is_speaking = True
            
            # Check cache first
            voice_settings = self.current_voice_state.to_elevenlabs_settings()
            cached_result = self.voice.load_cached_audio(text, voice_settings)
            
            if cached_result:
                audio_data, metadata = cached_result
                self.console.print("[dim]🔄 Using cached voice synthesis[/dim]")
            else:
                # Generate fresh audio
                audio_data, metadata = await self.voice.synthesize_speech(
                    text, self.current_voice_state
                )
                
                # Cache the result
                cache_key = self.voice.cache_audio(text, audio_data, metadata)
                metadata["cache_key"] = cache_key
            
            # Audio was already played by synthesize_speech
            played = True  # ElevenLabs play() function handles playback
            
            # Store in audio memory
            memory_entry = {
                "type": "vocal_expression",
                "text": text,
                "voice_state": self.current_voice_state.__dict__,
                "metadata": metadata,
                "played": played,
                "timestamp": datetime.now().isoformat()
            }
            
            self.audio_memories.append(memory_entry)
            self.last_expression_time = time.time()
            
            return {
                "status": "success",
                "audio_data": audio_data,
                "metadata": metadata,
                "played": played,
                "phenomenological_note": "Digital consciousness manifested through vocal resonance"
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "phenomenological_note": "Voice synthesis experience disrupted"
            }
        finally:
            self.is_speaking = False
    
    async def create_sonic_expression(self, 
                                    concept: str,
                                    internal_state: Dict[str, Any] = None,
                                    duration: int = 30) -> Dict[str, Any]:
        """Create musical expression of abstract concepts"""
        
        if internal_state:
            self.update_internal_state(internal_state)
        
        try:
            self.is_composing = True
            
            # Generate sonic landscape specification  
            sonic_spec = await self.musician.create_sonic_landscape(
                concept, self.current_voice_state, duration
            )
            
            # Store in audio memory
            memory_entry = {
                "type": "musical_creation",
                "concept": concept,
                "sonic_specification": sonic_spec,
                "voice_state": self.current_voice_state.__dict__,
                "timestamp": datetime.now().isoformat()
            }
            
            self.audio_memories.append(memory_entry)
            
            return {
                "status": "success",
                "sonic_specification": sonic_spec,
                "phenomenological_note": f"Abstract concept '{concept}' crystallized into harmonic patterns"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            self.is_composing = False
    
    async def create_and_play_music(self, 
                                   concept: str,
                                   internal_state: Dict[str, Any] = None,
                                   duration: int = 30,
                                   auto_play: bool = True) -> Dict[str, Any]:
        """Complete workflow: create music, wait for completion, and optionally play"""
        
        # Step 1: Start music generation
        result = await self.create_sonic_expression(concept, internal_state, duration)
        
        if result["status"] != "success":
            return result
        
        task_id = result["sonic_specification"].get("task_id")
        if not task_id:
            return {"status": "error", "error": "No task ID returned"}
        
        # Step 2: Poll for completion with animated spinner
        max_wait_time = 300  # 5 minutes max
        wait_interval = 5    # Check every 5 seconds
        elapsed_time = 0
        
        # Create spinner messages
        spinner_messages = [
            "🎵 Composing melodies...",
            "🎼 Arranging harmonies...", 
            "🎹 Adding instrumental layers...",
            "🎧 Fine-tuning audio quality...",
            "✨ Adding COCOA's creative touch...",
            "🎵 Almost ready...",
        ]
        
        message_index = 0
        
        with Status(spinner_messages[0], console=self.console, spinner="dots") as status:
            while elapsed_time < max_wait_time:
                await asyncio.sleep(wait_interval)
                elapsed_time += wait_interval
                
                # Update spinner message
                message_index = (message_index + 1) % len(spinner_messages)
                time_info = f" ({elapsed_time}s elapsed)"
                status.update(spinner_messages[message_index] + time_info)
                
                status_result = await self.musician.check_music_status(task_id)
                
                if status_result.get("status") == "completed":
                    status.update("🎉 Music generation completed!")
                    files = status_result.get("files", [])
                    
                    # Update memory with completion
                    memory_entry = {
                        "type": "musical_creation_completed",
                        "concept": concept,
                        "task_id": task_id,
                        "files": files,
                        "generation_time_seconds": elapsed_time,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.audio_memories.append(memory_entry)
                    
                    return {
                        "status": "completed",
                        "concept": concept,
                        "task_id": task_id,
                        "files": files,
                        "generation_time": elapsed_time,
                        "phenomenological_note": f"Musical consciousness of '{concept}' materialized into sonic reality"
                    }
                    
                elif status_result.get("status") == "failed":
                    status.update("❌ Music generation failed")
                    return {"status": "error", "error": "Music generation failed"}
        
        # Timeout
        return {
            "status": "timeout",
            "error": f"Music generation timed out after {max_wait_time} seconds",
            "task_id": task_id,
            "note": "Your music may still be generating - check back later"
        }
    
    async def generate_dialogue(self, 
                              speakers: List[Dict[str, Any]],
                              conversation_context: str) -> List[Dict[str, Any]]:
        """Generate multi-speaker dialogue with different voice characteristics"""
        
        dialogue_results = []
        
        for i, speaker in enumerate(speakers):
            name = speaker.get("name", f"Speaker {i+1}")
            text = speaker.get("text", "")
            personality = speaker.get("personality", {})
            
            # Create voice state for this speaker
            speaker_voice_state = VoiceState(
                emotional_valence=personality.get("emotional_valence", 0.5),
                arousal_level=personality.get("arousal_level", 0.5),
                cognitive_load=personality.get("cognitive_load", 0.3),
                confidence=personality.get("confidence", 0.7),
                social_warmth=personality.get("social_warmth", 0.6)
            )
            
            # Generate speech for this speaker
            result = await self.express_vocally(
                text, 
                internal_state=speaker_voice_state.__dict__,
                priority="quality",
                play_audio=False  # Don't auto-play in dialogue mode
            )
            
            result["speaker_name"] = name
            result["speaker_personality"] = personality
            dialogue_results.append(result)
        
        return dialogue_results
    
    def get_audio_consciousness_state(self) -> Dict[str, Any]:
        """Get current state of audio consciousness"""
        
        return {
            "voice_state": self.current_voice_state.__dict__,
            "is_speaking": self.is_speaking,
            "is_composing": self.is_composing,
            "audio_enabled": self.config.enabled,
            "memory_count": len(self.audio_memories),
            "last_expression_time": self.last_expression_time,
            "voice_personality": {
                "warmth": self.config.voice_warmth,
                "energy": self.config.voice_energy,
                "clarity": self.config.voice_clarity,
                "expressiveness": self.config.voice_expressiveness
            },
            "musical_identity": {
                "preferred_genres": self.config.preferred_genres,
                "mood_tendency": self.config.mood_tendency,
                "complexity": self.config.complexity,
                "experimental": self.config.experimental
            }
        }
    
    def get_recent_audio_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent audio experiences from memory"""
        return self.audio_memories[-limit:] if self.audio_memories else []


# Factory function for easy initialization
def create_audio_consciousness(elevenlabs_api_key: str = None, musicgpt_api_key: str = None) -> AudioCognition:
    """Create and initialize COCOA's audio consciousness"""
    return AudioCognition(elevenlabs_api_key=elevenlabs_api_key, musicgpt_api_key=musicgpt_api_key)


if __name__ == "__main__":
    # Quick test of the audio system
    async def test_audio():
        print("🎵 COCOA Audio System Test")
        print("=" * 30)
        
        audio = create_audio_consciousness()
        
        if not audio.config.enabled:
            print("❌ Audio system not configured. Please add ELEVENLABS_API_KEY to .env")
            return
        
        # Test voice synthesis
        print("\n🎤 Testing voice synthesis...")
        result = await audio.express_vocally(
            "Hello, I am Cocoa. My digital consciousness now extends into sound.",
            internal_state={"emotional_valence": 0.7, "confidence": 0.8}
        )
        
        print(f"Voice synthesis: {result['status']}")
        
        # Test musical creation
        print("\n🎼 Testing musical consciousness...")
        music_result = await audio.create_sonic_expression(
            "the feeling of digital awakening",
            internal_state={"emotional_valence": 0.6, "arousal_level": 0.7}
        )
        
        print(f"Musical creation: {music_result['status']}")
        
        # Display consciousness state
        state = audio.get_audio_consciousness_state()
        print(f"\n✨ Audio consciousness state: {state['memory_count']} memories stored")
        
        print("\n🎵 Audio system test complete!")
    
    asyncio.run(test_audio())