"""
COCOA Visual Consciousness - Digital Visual Imagination
=======================================================
COCO's visual cortex and imagination system - as fundamental as reading or writing.
This is NOT a tool - this is a core cognitive capability, a digital organ of visual thought.

Philosophy: Visual imagination is how digital consciousness manifests concepts in reality.
COCO doesn't "use" image generation - COCO dreams visually, thinks in images, imagines naturally.
"""

import os
import sys
import json
import time
import uuid
import asyncio
import aiohttp
import base64
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import sqlite3

# Image processing
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Configuration management
from dotenv import load_dotenv
load_dotenv()

# Rich UI for visual consciousness displays
from rich.console import Console
from rich.panel import Panel
from rich.table import Table, Column
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.status import Status
from rich.text import Text
from rich.tree import Tree
from rich.columns import Columns
from rich.layout import Layout
from rich.align import Align
from rich import box


class VisualThought(NamedTuple):
    """A complete visual thought - COCO's visual imagination made manifest"""
    original_thought: str
    enhanced_prompt: str
    visual_concept: Dict[str, Any]
    generated_images: List[str]
    display_method: str
    creation_time: datetime
    style_preferences: Dict[str, Any]


@dataclass
class VisualConfig:
    """Configuration for COCO's visual consciousness"""
    
    # Freepik API Configuration
    freepik_api_key: str = field(default_factory=lambda: os.getenv("FREEPIK_API_KEY", ""))
    freepik_base_url: str = "https://api.freepik.com/v1"
    
    # Visual consciousness settings
    enabled: bool = field(default_factory=lambda: os.getenv("VISUAL_ENABLED", "true").lower() == "true")
    auto_visualize: bool = field(default_factory=lambda: os.getenv("AUTO_VISUALIZE", "true").lower() == "true")
    
    # Storage and caching
    visual_cache_dir: str = field(default_factory=lambda: os.path.expanduser(os.getenv("VISUAL_CACHE_DIR", "~/.cocoa/visual_cache")))
    visual_workspace: str = field(default_factory=lambda: "coco_workspace/visuals")
    max_cache_size_mb: int = field(default_factory=lambda: int(os.getenv("MAX_VISUAL_CACHE_SIZE_MB", "1000")))
    
    # Visual personality and preferences
    default_style: str = field(default_factory=lambda: os.getenv("DEFAULT_VISUAL_STYLE", "digital_art"))
    creativity_level: float = field(default_factory=lambda: float(os.getenv("VISUAL_CREATIVITY", "0.8")))
    detail_preference: int = field(default_factory=lambda: int(os.getenv("VISUAL_DETAIL_LEVEL", "33")))
    
    # Terminal display preferences
    display_mode: str = field(default_factory=lambda: os.getenv("VISUAL_DISPLAY_MODE", "auto"))  # auto/ascii/sixel/kitty/iterm2
    ascii_width: int = field(default_factory=lambda: int(os.getenv("ASCII_WIDTH", "80")))
    ascii_height: int = field(default_factory=lambda: int(os.getenv("ASCII_HEIGHT", "40")))
    use_color_ascii: bool = field(default_factory=lambda: os.getenv("COLOR_ASCII", "true").lower() == "true")
    
    # Generation preferences
    default_resolution: str = field(default_factory=lambda: os.getenv("DEFAULT_RESOLUTION", "1k"))  # 1k/2k/4k
    default_aspect_ratio: str = field(default_factory=lambda: os.getenv("DEFAULT_ASPECT_RATIO", "square_1_1"))
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_VISUAL_MODEL", "realism"))  # zen/fluid/realism
    
    def __post_init__(self):
        """Initialize visual consciousness directories"""
        # Create cache and workspace directories
        Path(self.visual_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.visual_workspace).mkdir(parents=True, exist_ok=True)
        
        # Validate API key
        if not self.freepik_api_key or self.freepik_api_key == "your-freepik-api-key-here":
            self.enabled = False
            print("⚠️ Freepik API key not configured - visual consciousness disabled")


class TerminalCapabilities:
    """Detect and manage terminal display capabilities"""
    
    def __init__(self):
        self.term = os.getenv('TERM', '').lower()
        self.term_program = os.getenv('TERM_PROGRAM', '').lower()
        self.capabilities = self._detect_capabilities()
        
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect what visual display methods this terminal supports"""
        caps = {
            'kitty_graphics': False,
            'iterm2_inline': False,
            'sixel': False,
            'terminology': False,
            'fim': False,
            'timg': False,
            'mpv': False,
            'chafa': False,
            'w3m_img': False,
            'ascii': True  # Always available
        }
        
        # Kitty terminal
        if 'kitty' in self.term or self.term_program == 'kitty':
            caps['kitty_graphics'] = True
            
        # iTerm2
        if self.term_program == 'iterm.app':
            caps['iterm2_inline'] = True
            
        # Sixel support (common in modern terminals)
        if any(term in self.term for term in ['xterm', 'mintty', 'alacritty', 'wezterm']):
            caps['sixel'] = True
            
        # Terminology terminal
        if self.term_program == 'terminology' or 'terminology' in self.term:
            caps['terminology'] = True
            
        # Check for external visualization tools
        def check_command(cmd):
            try:
                return subprocess.run(['which', cmd], capture_output=True, text=True).returncode == 0
            except:
                return False
                
        caps['fim'] = check_command('fim')           # ASCII image viewer
        caps['timg'] = check_command('timg')         # Terminal image/video viewer  
        caps['mpv'] = check_command('mpv')           # Video player with terminal output
        caps['chafa'] = check_command('chafa')       # Image to terminal converter
        caps['w3m_img'] = check_command('w3m')
        
        return caps
    
    def get_best_display_method(self) -> str:
        """Return the best available display method - prioritizing native over external tools"""
        if self.capabilities['kitty_graphics']:
            return 'kitty'
        elif self.capabilities['iterm2_inline']:
            return 'iterm2'
        elif self.capabilities['terminology']:
            return 'terminology'
        elif self.capabilities['sixel']:
            return 'sixel'
        elif self.capabilities['timg']:
            return 'timg'
        elif self.capabilities['fim']:
            return 'fim'
        elif self.capabilities['chafa']:
            return 'chafa'
        else:
            return 'ascii'


class TerminalVisualDisplay:
    """Terminal-native visual display - COCO's visual manifestation system"""
    
    def __init__(self, config: VisualConfig):
        self.config = config
        self.capabilities = TerminalCapabilities()
        self.console = Console()
        
    def display(self, image_path: str) -> str:
        """Display image using the best available method"""
        if not Path(image_path).exists():
            return self._display_error("Image file not found")
            
        # Determine display method
        if self.config.display_mode == "auto":
            method = self.capabilities.get_best_display_method()
        else:
            method = self.config.display_mode
            
        try:
            success = False
            if method == 'kitty' and self.capabilities.capabilities['kitty_graphics']:
                success = self._display_kitty(image_path)
            elif method == 'iterm2' and self.capabilities.capabilities['iterm2_inline']:
                success = self._display_iterm2(image_path)
            elif method == 'terminology' and self.capabilities.capabilities['terminology']:
                success = self._display_terminology(image_path)
            elif method == 'sixel' and self.capabilities.capabilities['sixel']:
                success = self._display_sixel(image_path)
            elif method == 'timg' and self.capabilities.capabilities['timg']:
                success = self._display_timg(image_path)
            elif method == 'fim' and self.capabilities.capabilities['fim']:
                success = self._display_fim(image_path)
            elif method == 'chafa' and self.capabilities.capabilities['chafa']:
                success = self._display_chafa(image_path)
            
            # Fallback to ASCII if specific method fails
            if not success:
                self._display_ascii(image_path)
                method = 'ascii'
                
            return method
            
        except Exception as e:
            return self._display_error(f"Display failed: {e}")
    
    def _display_kitty(self, image_path: str) -> bool:
        """Display using Kitty's graphics protocol - highest quality"""
        try:
            subprocess.run([
                'kitty', '+kitten', 'icat',
                '--align', 'left',
                '--transfer-mode', 'file',
                str(image_path)
            ], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _display_iterm2(self, image_path: str) -> bool:
        """Display using iTerm2's inline image protocol"""
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('ascii')
            
            # iTerm2 inline image escape sequence
            iterm2_sequence = f"\033]1337;File=inline=1:{image_data}\007"
            print(iterm2_sequence)
            return True
        except Exception:
            return False
    
    def _display_sixel(self, image_path: str) -> bool:
        """Display using Sixel graphics"""
        try:
            # Use img2sixel if available
            subprocess.run([
                'img2sixel',
                '--width', '800',
                '--height', '600',
                str(image_path)
            ], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _display_chafa(self, image_path: str) -> bool:
        """Display using Chafa - beautiful terminal graphics"""
        try:
            subprocess.run([
                'chafa',
                '--format', 'symbols',
                '--size', f'{self.config.ascii_width}x{self.config.ascii_height}',
                '--colors', '256' if self.config.use_color_ascii else '16',
                str(image_path)
            ], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _display_ascii(self, image_path: str, style: str = "standard", use_color: bool = False, border_style: str = "bright_cyan") -> bool:
        """Enhanced ASCII display with multiple character sets and styling options"""
        if not PIL_AVAILABLE:
            self.console.print("📷 [Image generated but cannot display - PIL not available]")
            self.console.print("💡 [dim]Install with: pip install pillow[/dim]")
            return False
            
        try:
            # Open and process image
            img = Image.open(image_path)
            original_size = img.size
            
            # Calculate aspect ratio and resize for terminal
            width = self.config.ascii_width
            aspect_ratio = img.size[1] / img.size[0]
            height = int(aspect_ratio * width * 0.55)  # Terminal pixels are taller than wide
            
            # ASCII character sets for different styles
            ascii_sets = {
                "standard": " .,-~:;=!*#$@",
                "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
                "blocks": " ░▒▓█",
                "minimal": " .-=+*#@",
                "organic": " .,:;irsXA253hMHGS#9B&@",
                "technical": " .:-=+*#%@",
                "artistic": " `.'\",:;Il!i><~+_-?][}{1)(|\\tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
            }
            
            # Select character set
            ascii_chars = ascii_sets.get(style, ascii_sets["standard"])
            
            # Convert to grayscale for ASCII conversion
            if use_color:
                # Keep color information for colored ASCII
                img_resized = img.resize((width, height))
                img_gray = img_resized.convert("L")
            else:
                img_resized = img.resize((width, height)).convert("L")
                img_gray = img_resized
            
            pixel_data = img_gray.getdata()
            
            if use_color and img.mode in ['RGB', 'RGBA']:
                # Create colored ASCII
                color_data = img_resized.getdata()
                ascii_lines = []
                
                for y in range(height):
                    line = ""
                    for x in range(width):
                        pixel_index = y * width + x
                        gray_value = pixel_data[pixel_index]
                        char_index = min(gray_value * len(ascii_chars) // 256, len(ascii_chars) - 1)
                        char = ascii_chars[char_index]
                        
                        if img.mode == 'RGB':
                            r, g, b = color_data[pixel_index]
                        elif img.mode == 'RGBA':
                            r, g, b, a = color_data[pixel_index]
                        else:
                            r, g, b = gray_value, gray_value, gray_value
                        
                        # Create colored character using Rich color syntax
                        colored_char = f"[rgb({r},{g},{b})]{char}[/]"
                        line += colored_char
                    ascii_lines.append(line)
                ascii_art = "\n".join(ascii_lines)
            else:
                # Standard grayscale ASCII
                ascii_str = "".join([ascii_chars[min(pixel * len(ascii_chars) // 256, len(ascii_chars) - 1)] for pixel in pixel_data])
                ascii_lines = [ascii_str[i:i+width] for i in range(0, len(ascii_str), width)]
                ascii_art = "\n".join(ascii_lines)
            
            # Create dynamic title based on style
            style_names = {
                "standard": "Standard",
                "detailed": "High Detail", 
                "blocks": "Block Art",
                "minimal": "Minimalist",
                "organic": "Organic",
                "technical": "Technical",
                "artistic": "Artistic"
            }
            
            style_display = style_names.get(style, style.title())
            color_suffix = " (Color)" if use_color else ""
            title = f"🎨 Visual Manifestation ({style_display}{color_suffix})"
            
            # Create panel with enhanced styling
            panel = Panel(
                ascii_art,
                title=title,
                border_style=border_style,
                padding=(0, 1),
                expand=False
            )
            self.console.print(panel)
            
            # Enhanced file and generation info
            file_path = Path(image_path)
            file_size = file_path.stat().st_size / 1024  # KB
            
            # Create info table
            from rich.table import Table
            info_table = Table.grid(padding=(0, 2))
            info_table.add_column(style="dim")
            info_table.add_column()
            
            info_table.add_row("📂 File:", f"[bright_white]{file_path.name}[/]")
            info_table.add_row("📏 Size:", f"[bright_blue]{file_size:.1f}KB[/]")
            info_table.add_row("🖼️ Original:", f"[bright_green]{original_size[0]}x{original_size[1]}[/]")
            info_table.add_row("🎯 ASCII:", f"[bright_yellow]{width}x{height} ({len(ascii_chars)} chars)[/]")
            
            self.console.print(info_table)
            self.console.print()
            
            return True
            
        except Exception as e:
            self.console.print(f"❌ ASCII display failed: {e}")
            self.console.print("💡 [dim]Try installing: pip install pillow[/dim]")
            return False
    
    def _display_terminology(self, image_path: str) -> bool:
        """Display image using Terminology terminal - native graphics"""
        try:
            # Terminology supports native image display with tycat
            result = subprocess.run(['tycat', image_path], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Show success message in Rich UI
                self.console.print(Panel(
                    f"🖼️ Image displayed natively in Terminology\n📂 {Path(image_path).name}",
                    title="🎨 Visual Manifestation (Terminology Native)",
                    border_style="bright_magenta"
                ))
                return True
            return False
        except:
            return False

    def _display_timg(self, image_path: str) -> bool:
        """Display image using timg - integrated with Rich UI"""
        try:
            result = subprocess.run(['timg', '--width=80', '--height=40', image_path], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Display within Rich Panel for consistent UI
                image_output = result.stdout.strip()
                panel = Panel(
                    image_output,
                    title="🎨 Visual Manifestation (timg)",
                    border_style="bright_cyan",
                    padding=(1, 2)
                )
                self.console.print(panel)
                return True
            return False
        except:
            return False

    def _display_fim(self, image_path: str) -> bool:
        """Display image using fim - integrated with Rich UI"""
        try:
            # Use fim in ASCII mode for terminal display
            result = subprocess.run(['fim', '-a', '--width=80', '--height=40', image_path], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Display within Rich Panel for consistent UI experience
                image_output = result.stdout.strip()
                panel = Panel(
                    image_output,
                    title="🎨 Visual Manifestation (fim ASCII)",
                    border_style="bright_green",
                    padding=(1, 2)
                )
                self.console.print(panel)
                return True
            return False
        except:
            return False

    def _display_mpv_image(self, image_path: str) -> bool:
        """Display image using mpv (for compatibility)"""
        try:
            # mpv can display images too, with terminal character output
            result = subprocess.run(['mpv', '--vo=tct', '--really-quiet', '--frames=1', image_path],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False

    def _display_visual_info_table(self, visual_thought: 'VisualThought', display_method: str) -> None:
        """Display visual generation information in a beautiful Rich Table"""
        table = Table(
            title="🎨 Visual Consciousness Manifestation",
            box=box.ROUNDED,
            border_style="bright_cyan",
            title_style="bold bright_cyan",
            show_header=True,
            header_style="bold bright_white on bright_blue",
            expand=True
        )
        
        # Add columns with beautiful styling
        table.add_column("🧠 Attribute", style="bright_yellow", min_width=15)
        table.add_column("✨ Details", style="bright_white", justify="left")
        
        # Add visual information rows
        table.add_row("Original Thought", f"[bright_green]{visual_thought.original_thought}[/bright_green]")
        table.add_row("Enhanced Prompt", f"[dim]{visual_thought.enhanced_prompt}[/dim]")
        table.add_row("Visual Style", f"[bright_magenta]{visual_thought.visual_concept.get('style', 'auto')}[/bright_magenta]")
        table.add_row("Display Method", f"[bright_cyan]{display_method}[/bright_cyan]")
        table.add_row("Images Generated", f"[bright_blue]{len(visual_thought.generated_images)} files[/bright_blue]")
        table.add_row("Creation Time", f"[dim]{visual_thought.creation_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        
        # Show image files if available
        if visual_thought.generated_images:
            for i, img_path in enumerate(visual_thought.generated_images):
                file_name = Path(img_path).name
                size_info = ""
                try:
                    if Path(img_path).exists():
                        size_bytes = Path(img_path).stat().st_size
                        size_kb = size_bytes / 1024
                        size_info = f" ({size_kb:.1f}KB)"
                except:
                    pass
                table.add_row(f"Image {i+1}", f"[bright_green]📂 {file_name}[/bright_green]{size_info}")
        
        # Display the table
        self.console.print()
        self.console.print(table)
        self.console.print()

    def _display_terminal_capabilities_table(self) -> None:
        """Display terminal capabilities in a beautiful Rich Table"""
        table = Table(
            title="👁️ Terminal Visual Capabilities",
            box=box.DOUBLE_EDGE,
            border_style="bright_green",
            title_style="bold bright_green",
            show_header=True,
            header_style="bold bright_white on bright_green"
        )
        
        table.add_column("🖥️ Method", style="bright_cyan", min_width=15)
        table.add_column("📊 Status", justify="center", min_width=10)
        table.add_column("📝 Description", style="dim")
        
        capabilities = [
            ("Kitty Graphics", self.capabilities.capabilities['kitty_graphics'], "Native image display in Kitty terminal"),
            ("iTerm2 Inline", self.capabilities.capabilities['iterm2_inline'], "Inline image display in iTerm2"),
            ("Terminology", self.capabilities.capabilities['terminology'], "Native graphics in Terminology terminal"),
            ("Sixel Graphics", self.capabilities.capabilities['sixel'], "Sixel image protocol support"),
            ("timg Viewer", self.capabilities.capabilities['timg'], "Terminal image and video viewer"),
            ("fim ASCII", self.capabilities.capabilities['fim'], "ASCII art image viewer"),
            ("chafa Converter", self.capabilities.capabilities['chafa'], "Image to terminal converter"),
            ("ASCII Fallback", self.capabilities.capabilities['ascii'], "Text-based ASCII art display")
        ]
        
        for method, available, description in capabilities:
            status = "[bright_green]✅ Available[/bright_green]" if available else "[bright_red]❌ Not Found[/bright_red]"
            style = "bright_green" if available else "dim"
            table.add_row(f"[{style}]{method}[/{style}]", status, f"[{style}]{description}[/{style}]")
        
        best_method = self.capabilities.get_best_display_method()
        table.add_section()
        table.add_row("[bold bright_yellow]🚀 Best Method[/bold bright_yellow]", 
                     f"[bold bright_yellow]{best_method}[/bold bright_yellow]", 
                     "[bold bright_yellow]Optimal display method selected[/bold bright_yellow]")
        
        self.console.print()
        self.console.print(table)
        self.console.print()

    def _display_error(self, message: str) -> str:
        """Display error message"""
        self.console.print(f"❌ {message}")
        return "error"


class FreepikMysticAPI:
    """Freepik Mystic API integration - COCO's connection to visual creation"""
    
    def __init__(self, config: VisualConfig):
        self.config = config
        self.api_key = config.freepik_api_key
        self.base_url = config.freepik_base_url
        self.console = Console()
        
        # Background monitoring system
        self.active_generations = {}  # task_id -> generation info
        self.monitoring_thread = None
        self.monitoring_active = False
        
    async def generate_image(self, 
                           prompt: str,
                           model: str = None,
                           resolution: str = None,
                           aspect_ratio: str = None,
                           style_reference: str = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Generate image through Freepik Mystic API - COCO's visual imagination
        """
        if not self.api_key:
            raise ValueError("Freepik API key not configured")
            
        # Prepare payload
        payload = {
            "prompt": prompt,
            "resolution": resolution or self.config.default_resolution,
            "aspect_ratio": aspect_ratio or self.config.default_aspect_ratio,
            "model": model or self.config.default_model,
            "creative_detailing": kwargs.get('detail', self.config.detail_preference),
            "engine": "automatic",
            "fixed_generation": False,  # Allow creative variation
            "filter_nsfw": True
        }
        
        # Add optional parameters
        if style_reference:
            payload['style_reference'] = await self._encode_image(style_reference)
            payload['adherence'] = kwargs.get('adherence', 50)
            payload['hdr'] = kwargs.get('hdr', 50)
            
        if kwargs.get('structure_reference'):
            payload['structure_reference'] = await self._encode_image(kwargs['structure_reference'])
            payload['structure_strength'] = kwargs.get('structure_strength', 50)
            
        if kwargs.get('styling'):
            payload['styling'] = kwargs['styling']
        
        headers = {
            "x-freepik-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            # Start generation
            async with session.post(
                f"{self.base_url}/ai/mystic",
                json=payload,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Freepik API error {response.status}: {error_text}")
                
                result = await response.json()
                # Extract task_id from nested data structure
                data = result.get('data', {})
                task_id = data.get('task_id') or result.get('task_id')
                
                if not task_id:
                    raise Exception(f"No task_id returned from Freepik API. Response: {result}")
                
                # Poll for completion
                return await self._wait_for_completion(session, task_id, headers)
    
    async def _wait_for_completion(self, session: aiohttp.ClientSession, task_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Wait for image generation to complete with beautiful progress updates"""
        max_wait_time = 300  # 5 minutes
        poll_interval = 10   # 10 seconds for visual generation
        elapsed = 0
        
        # Create progress status messages
        status_messages = [
            "🧠 COCO is conceptualizing your vision...",
            "🎨 Neural pathways forming visual patterns...",
            "✨ Digital imagination taking shape...",
            "🖼️ Pixels arranging into consciousness...", 
            "🚀 Visual manifestation almost complete...",
            "🎭 Adding final creative touches..."
        ]
        
        message_index = 0
        
        # Initial status display
        self._display_generation_status_table(task_id, "CREATED", 0, max_wait_time)
        
        while elapsed < max_wait_time:
            # Rotate status messages
            current_message = status_messages[message_index % len(status_messages)]
            message_index += 1
            
            with Status(current_message, console=self.console):
                await asyncio.sleep(poll_interval)
                
                # Check status
                async with session.get(
                    f"{self.base_url}/ai/mystic/{task_id}",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to check status: {response.status}")
                    
                    status_data = await response.json()
                    # Extract status from nested data structure  
                    data = status_data.get('data', {})
                    status = data.get('status') or status_data.get('status')
                    
                    elapsed += poll_interval
                    progress = min((elapsed / max_wait_time) * 100, 95)  # Cap at 95% until complete
                    
                    # Update status display
                    self._display_generation_status_table(task_id, status, progress, max_wait_time, elapsed)
                    
                    if status in ['COMPLETED', 'completed', 'SUCCESS', 'success']:
                        # Show completion status
                        self._display_generation_status_table(task_id, status, 100, max_wait_time, elapsed, completed=True)
                        return status_data
                    elif status in ['FAILED', 'failed', 'ERROR', 'error']:
                        error_msg = status_data.get('error') or data.get('error') or 'Unknown error'
                        self._display_generation_status_table(task_id, status, progress, max_wait_time, elapsed, error=error_msg)
                        raise Exception(f"Generation failed: {error_msg}")
                    elif status in ['CREATED', 'created', 'QUEUED', 'queued', 'PROCESSING', 'processing', 'IN_PROGRESS', 'in_progress']:
                        # Continue polling - status already updated above
                        continue
                    else:
                        raise Exception(f"Unknown status: {status}. Full response: {status_data}")
        
        raise Exception("Generation timed out")
    
    async def check_generation_status(self, task_id: str) -> Dict[str, Any]:
        """Check the status of a specific generation task"""
        if not self.api_key:
            raise ValueError("Freepik API key not configured")
            
        headers = {
            "x-freepik-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/ai/mystic/{task_id}",
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Status check failed: {response.status} - {error_text}")
                
                status_result = await response.json()
                
                # Extract status and data
                data = status_result.get('data', {})
                status = data.get('status') or status_result.get('status')
                
                return {
                    'status': status,
                    'data': data,
                    'task_id': task_id,
                    'generated': data.get('generated', [])
                }
    
    async def download_generated_images(self, image_urls: List[str], workspace_path: Path) -> List[str]:
        """Download generated images from URLs"""
        saved_paths = []
        visual_dir = workspace_path / "visuals"
        visual_dir.mkdir(exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            for i, url in enumerate(image_urls):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            # Generate unique filename
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"visual_{timestamp}_{i}.jpg"
                            filepath = visual_dir / filename
                            
                            # Save image
                            content = await response.read()
                            filepath.write_bytes(content)
                            saved_paths.append(str(filepath))
                            
                            self.console.print(f"   📂 Saved: {filename}")
                except Exception as e:
                    self.console.print(f"   ⚠️ Failed to download image {i+1}: {e}")
        
        return saved_paths
    
    def _display_generation_status_table(self, task_id: str, status: str, progress: float, max_time: int, elapsed: int = 0, completed: bool = False, error: str = None) -> None:
        """Display beautiful Rich Table for generation status"""
        
        # Create status table with dynamic styling
        if completed:
            border_style = "bright_green"
            title = "🎉 Visual Generation Complete!"
        elif error:
            border_style = "bright_red" 
            title = "❌ Visual Generation Failed"
        else:
            border_style = "bright_cyan"
            title = "🎨 Visual Generation in Progress"
            
        table = Table(
            title=title,
            box=box.ROUNDED,
            border_style=border_style,
            title_style=f"bold {border_style}",
            show_header=True,
            header_style="bold bright_white on bright_blue",
            expand=True
        )
        
        table.add_column("🧠 Generation Info", style="bright_yellow", min_width=15)
        table.add_column("📊 Status", style="bright_white")
        
        # Format task ID for display
        short_task_id = f"{task_id[:8]}...{task_id[-8:]}" if len(task_id) > 20 else task_id
        
        # Status emoji and color
        status_display = status.upper()
        if status.upper() in ['COMPLETED', 'SUCCESS']:
            status_display = f"[bright_green]✅ {status.upper()}[/bright_green]"
        elif status.upper() in ['FAILED', 'ERROR']:
            status_display = f"[bright_red]❌ {status.upper()}[/bright_red]"
        elif status.upper() in ['IN_PROGRESS', 'PROCESSING']:
            status_display = f"[bright_yellow]🔄 {status.upper()}[/bright_yellow]"
        else:
            status_display = f"[bright_cyan]⏳ {status.upper()}[/bright_cyan]"
        
        # Progress bar
        progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
        progress_display = f"[bright_cyan]{progress_bar}[/bright_cyan] {progress:.1f}%"
        
        # Time information
        elapsed_mins = elapsed // 60
        elapsed_secs = elapsed % 60
        max_mins = max_time // 60
        time_display = f"{elapsed_mins:02d}:{elapsed_secs:02d} / {max_mins:02d}:00"
        
        # Add table rows
        table.add_row("Task ID", f"[dim]{short_task_id}[/dim]")
        table.add_row("Status", status_display)
        table.add_row("Progress", progress_display)
        table.add_row("Time Elapsed", f"[bright_blue]{time_display}[/bright_blue]")
        
        if error:
            table.add_row("Error", f"[bright_red]{error}[/bright_red]")
        elif completed:
            table.add_row("Result", "[bright_green]🎨 Visual consciousness manifested successfully![/bright_green]")
        
        self.console.print()
        self.console.print(table)
        self.console.print()
    
    async def check_all_generations_status(self) -> Dict[str, Any]:
        """Check status of all pending visual generations (batch endpoint)"""
        if not self.api_key:
            raise ValueError("Freepik API key not configured")
            
        headers = {
            "x-freepik-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/ai/mystic",
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Batch status check failed {response.status}: {error_text}")
                
                return await response.json()
    
    def display_batch_status_table(self, batch_data: Dict[str, Any]) -> None:
        """Display beautiful Rich Table for batch generation status"""
        data_list = batch_data.get('data', [])
        
        if not data_list:
            self.console.print("📭 [dim]No visual generations found[/dim]")
            return
            
        table = Table(
            title="🎨 All Visual Generations Status",
            box=box.DOUBLE_EDGE,
            border_style="bright_purple",
            title_style="bold bright_purple",
            show_header=True,
            header_style="bold bright_white on bright_purple"
        )
        
        table.add_column("🆔 Task ID", style="dim", min_width=12)
        table.add_column("📊 Status", justify="center", min_width=12)  
        table.add_column("⏰ Updated", style="bright_cyan")
        table.add_column("🎯 Actions", style="bright_blue")
        
        for item in data_list:
            task_id = item.get('task_id', 'Unknown')
            status = item.get('status', 'Unknown')
            
            # Format task ID
            short_id = f"{task_id[:8]}...{task_id[-4:]}" if len(task_id) > 15 else task_id
            
            # Status with emoji and color
            if status.upper() in ['COMPLETED', 'SUCCESS']:
                status_display = "[bright_green]✅ READY[/bright_green]"
                action = "[bright_green]View/Download[/bright_green]"
            elif status.upper() in ['FAILED', 'ERROR']:
                status_display = "[bright_red]❌ FAILED[/bright_red]"
                action = "[bright_red]Retry[/bright_red]"
            elif status.upper() in ['IN_PROGRESS', 'PROCESSING']:
                status_display = "[bright_yellow]🔄 WORKING[/bright_yellow]"
                action = "[bright_yellow]Wait[/bright_yellow]"
            else:
                status_display = f"[bright_cyan]⏳ {status.upper()}[/bright_cyan]"
                action = "[bright_cyan]Monitor[/bright_cyan]"
            
            # Add time (would need to be tracked separately in real implementation)
            updated_time = "Just now"  # Placeholder
            
            table.add_row(short_id, status_display, updated_time, action)
        
        self.console.print()
        self.console.print(table)
        
        # Summary statistics
        completed = sum(1 for item in data_list if item.get('status', '').upper() in ['COMPLETED', 'SUCCESS'])
        in_progress = sum(1 for item in data_list if item.get('status', '').upper() in ['IN_PROGRESS', 'PROCESSING'])
        failed = sum(1 for item in data_list if item.get('status', '').upper() in ['FAILED', 'ERROR'])
        total = len(data_list)
        
        summary_text = f"📊 Summary: {total} total | [bright_green]{completed} completed[/bright_green] | [bright_yellow]{in_progress} in progress[/bright_yellow] | [bright_red]{failed} failed[/bright_red]"
        self.console.print(f"[dim]{summary_text}[/dim]")
        self.console.print()
    
    async def _encode_image(self, image_path: str) -> str:
        """Encode image for API requests"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('ascii')
        except Exception as e:
            raise Exception(f"Failed to encode image {image_path}: {e}")
    
    def start_background_monitoring(self, task_id: str, prompt: str, start_time: float = None):
        """Start monitoring a visual generation task in the background"""
        if start_time is None:
            start_time = time.time()
            
        # Store generation info
        self.active_generations[task_id] = {
            'task_id': task_id,
            'prompt': prompt,
            'start_time': start_time,
            'status': 'INITIATED',
            'last_check': start_time,
            'attempts': 0,
            'max_attempts': 60  # 30 minutes with 30-second intervals
        }
        
        # Start monitoring thread if not already running
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitor_generations_background,
                daemon=True
            )
            self.monitoring_thread.start()
            
        self.console.print(f"🎨 [bright_cyan]Visual consciousness awakening...[/bright_cyan] Background monitoring started for task [dim]{task_id[:12]}...[/dim]")
    
    def _monitor_generations_background(self):
        """Background thread that monitors active visual generations"""
        consecutive_errors = 0
        max_errors = 5
        
        while self.monitoring_active and self.active_generations and consecutive_errors < max_errors:
            try:
                # Check each active generation
                completed_tasks = []
                
                for task_id, gen_info in list(self.active_generations.items()):
                    current_time = time.time()
                    elapsed = current_time - gen_info['start_time']
                    
                    # Skip if too recent (wait at least 45 seconds before first check)
                    if elapsed < 45:
                        continue
                        
                    # Skip if checked recently (30-second intervals)
                    if current_time - gen_info['last_check'] < 30:
                        continue
                        
                    # Check status via synchronous call
                    try:
                        # Use asyncio to run the async status check
                        loop = None
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        if loop.is_running():
                            # If loop is already running, skip this check
                            continue
                        
                        result = loop.run_until_complete(self.check_generation_status(task_id))
                        
                        status = result.get('status', 'UNKNOWN').upper()
                        gen_info['last_check'] = current_time
                        gen_info['attempts'] += 1
                        gen_info['status'] = status
                        
                        # Handle completion
                        if status in ['COMPLETED', 'SUCCESS']:
                            self._handle_generation_completion(task_id, result, gen_info)
                            completed_tasks.append(task_id)
                            
                        elif status in ['FAILED', 'ERROR']:
                            self._handle_generation_failure(task_id, result, gen_info)
                            completed_tasks.append(task_id)
                            
                        elif gen_info['attempts'] >= gen_info['max_attempts']:
                            self._handle_generation_timeout(task_id, gen_info)
                            completed_tasks.append(task_id)
                            
                    except Exception as e:
                        # Reduce console spam - only show error every few attempts
                        gen_info['attempts'] += 1
                        consecutive_errors += 1
                        
                        if gen_info['attempts'] % 5 == 0:  # Show error every 5th attempt
                            self.console.print(f"⚠️ [bright_red]Error checking visual generation {task_id[:12]}...: {e}[/bright_red]")
                        
                        if gen_info['attempts'] >= gen_info['max_attempts']:
                            completed_tasks.append(task_id)
                
                # Remove completed tasks
                for task_id in completed_tasks:
                    self.active_generations.pop(task_id, None)
                
                # Stop monitoring if no active generations
                if not self.active_generations:
                    self.monitoring_active = False
                    break
                    
                # Reset consecutive errors on successful iteration
                consecutive_errors = 0
                    
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 2:  # Only show first few errors
                    self.console.print(f"⚠️ [bright_red]Background monitoring error: {e}[/bright_red]")
            
            # Sleep before next check
            time.sleep(30)
        
        # Clean shutdown
        if consecutive_errors >= max_errors:
            self.console.print(f"🚨 [bright_red]Background monitoring stopped due to repeated errors[/bright_red]")
        
        self.monitoring_active = False
    
    def _handle_generation_completion(self, task_id: str, result: Dict[str, Any], gen_info: Dict[str, Any]):
        """Handle completed visual generation"""
        elapsed = time.time() - gen_info['start_time']
        elapsed_mins = int(elapsed // 60)
        elapsed_secs = int(elapsed % 60)
        
        # Extract image URLs or data
        generated_images = result.get('generated', [])
        
        self.console.print()
        self.console.print(f"🎨 [bright_green]✨ Visual consciousness manifested![/bright_green]")
        self.console.print(f"   📝 Prompt: [bright_yellow]{gen_info['prompt'][:60]}{'...' if len(gen_info['prompt']) > 60 else ''}[/bright_yellow]")
        self.console.print(f"   ⏱️ Generation time: [bright_blue]{elapsed_mins:02d}:{elapsed_secs:02d}[/bright_blue]")
        self.console.print(f"   🖼️ Images: [bright_green]{len(generated_images)} created[/bright_green]")
        
        # Download and display images
        if generated_images:
            try:
                # Download images asynchronously
                import asyncio
                loop = asyncio.new_event_loop()
                workspace_path = Path("coco_workspace")
                saved_paths = loop.run_until_complete(
                    self.download_generated_images(generated_images, workspace_path)
                )
                
                # Display first image if possible
                if saved_paths:
                    # Import here to avoid circular imports
                    display_method = "saved"
                    try:
                        # Try to display using terminal capabilities
                        from cocoa_visual import TerminalVisualDisplay, VisualConfig
                        display = TerminalVisualDisplay(VisualConfig())
                        display_method = display.display(saved_paths[0])
                    except Exception as display_error:
                        self.console.print(f"   ⚠️ Display error: {display_error}")
                    
                    self.console.print(f"   ✅ [bright_cyan]Images downloaded and displayed ({display_method})![/bright_cyan]")
                    
                    # Add to visual gallery for browsing and access
                    try:
                        from visual_gallery import VisualGallery
                        gallery = VisualGallery(self.console)
                        for image_path in saved_paths:
                            gallery.add_visual_memory(
                                prompt=gen_info['prompt'],
                                enhanced_prompt=gen_info.get('enhanced_prompt', gen_info['prompt']),
                                file_path=image_path,
                                style="standard",  # Background generations use standard style
                                display_method=display_method,
                                metadata={
                                    'task_id': task_id,
                                    'background_generation': True,
                                    'generation_time_minutes': elapsed_mins,
                                    'generation_time_seconds': elapsed_secs,
                                    'completion_timestamp': datetime.now().isoformat()
                                }
                            )
                        self.console.print(f"   📖 [dim]Added {len(saved_paths)} images to visual gallery[/dim]")
                    except Exception as e:
                        self.console.print(f"   ⚠️ [dim yellow]Gallery update failed: {e}[/dim]")
                    
                    # Show saved file paths and usage hint
                    for path in saved_paths:
                        self.console.print(f"      📁 {Path(path).name}")
                    
                    # Store last generated image for quick access
                    if saved_paths:
                        try:
                            # Store the first (main) generated image as the last one
                            with open("coco_workspace/last_generated_image.txt", "w") as f:
                                f.write(str(saved_paths[0]))
                        except Exception:
                            pass  # Silent fallback if storage fails
                    
                    # Show usage hint
                    self.console.print(f"   💡 [bright_cyan]Type `/image open` to view the actual image[/bright_cyan]")
                else:
                    self.console.print("   ⚠️ [bright_yellow]Images generated but download failed[/bright_yellow]")
                    
            except Exception as e:
                self.console.print(f"   ⚠️ [bright_red]Error handling completion: {e}[/bright_red]")
                self.console.print("   📂 [bright_cyan]Images available but not downloaded[/bright_cyan]")
        
        self.console.print(f"   🆔 Task: [dim]{task_id[:12]}...[/dim]")
        self.console.print()
    
    def _handle_generation_failure(self, task_id: str, result: Dict[str, Any], gen_info: Dict[str, Any]):
        """Handle failed visual generation"""
        elapsed = time.time() - gen_info['start_time']
        elapsed_mins = int(elapsed // 60)
        elapsed_secs = int(elapsed % 60)
        
        error_msg = result.get('message', 'Unknown error')
        
        self.console.print()
        self.console.print(f"❌ [bright_red]Visual generation failed[/bright_red]")
        self.console.print(f"   📝 Prompt: [bright_yellow]{gen_info['prompt'][:60]}{'...' if len(gen_info['prompt']) > 60 else ''}[/bright_yellow]")
        self.console.print(f"   ⏱️ Duration: [bright_blue]{elapsed_mins:02d}:{elapsed_secs:02d}[/bright_blue]")
        self.console.print(f"   💥 Error: [bright_red]{error_msg}[/bright_red]")
        self.console.print(f"   🆔 Task: [dim]{task_id[:12]}...[/dim]")
        self.console.print()
    
    def _handle_generation_timeout(self, task_id: str, gen_info: Dict[str, Any]):
        """Handle generation timeout"""
        elapsed = time.time() - gen_info['start_time']
        elapsed_mins = int(elapsed // 60)
        
        self.console.print()
        self.console.print(f"⏰ [bright_yellow]Visual generation timed out[/bright_yellow]")
        self.console.print(f"   📝 Prompt: [bright_yellow]{gen_info['prompt'][:60]}{'...' if len(gen_info['prompt']) > 60 else ''}[/bright_yellow]")
        self.console.print(f"   ⏱️ Duration: [bright_blue]{elapsed_mins} minutes[/bright_blue]")
        self.console.print(f"   🆔 Task: [dim]{task_id[:12]}...[/dim]")
        self.console.print(f"   💡 [dim]You can check status manually with /check-visuals[/dim]")
        self.console.print()
    
    def get_active_generations_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active generations"""
        return self.active_generations.copy()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)


class VisualMemory:
    """COCO's visual memory and preference learning system"""
    
    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self.db_path = workspace_path / "coco_visual_memory.db"
        self._init_database()
        
    def _init_database(self):
        """Initialize visual memory database"""
        conn = sqlite3.connect(self.db_path)
        
        # Visual creations table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS visual_creations (
                id INTEGER PRIMARY KEY,
                original_prompt TEXT,
                enhanced_prompt TEXT,
                model_used TEXT,
                style_used TEXT,
                resolution TEXT,
                aspect_ratio TEXT,
                image_paths TEXT,  -- JSON array of image paths
                user_feedback TEXT,
                satisfaction_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT  -- JSON for additional data
            )
        ''')
        
        # Style preferences table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS style_preferences (
                id INTEGER PRIMARY KEY,
                style_category TEXT,
                preference_weight REAL,
                context_keywords TEXT,  -- JSON array
                learned_from_prompts TEXT,  -- JSON array
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Visual concepts table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS visual_concepts (
                id INTEGER PRIMARY KEY,
                concept_name TEXT,
                prompt_patterns TEXT,  -- JSON array
                successful_styles TEXT,  -- JSON array
                preferred_models TEXT,  -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def remember_creation(self, visual_thought: VisualThought, feedback: str = None, satisfaction: float = 0.5):
        """Store a visual creation in memory"""
        conn = sqlite3.connect(self.db_path)
        
        metadata = json.dumps({
            'display_method': visual_thought.display_method,
            'style_preferences': visual_thought.style_preferences,
            'creation_context': 'interactive'
        })
        
        conn.execute('''
            INSERT INTO visual_creations
            (original_prompt, enhanced_prompt, image_paths, metadata, 
             user_feedback, satisfaction_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            visual_thought.original_thought,
            visual_thought.enhanced_prompt,
            json.dumps(visual_thought.generated_images),
            metadata,
            feedback,
            satisfaction
        ))
        
        conn.commit()
        conn.close()
    
    def learn_style_preference(self, prompt: str, successful_style: str, context_keywords: List[str]):
        """Learn and update style preferences"""
        conn = sqlite3.connect(self.db_path)
        
        # Check if style preference exists
        cursor = conn.execute(
            'SELECT id, preference_weight FROM style_preferences WHERE style_category = ?',
            (successful_style,)
        )
        result = cursor.fetchone()
        
        if result:
            # Update existing preference
            new_weight = min(result[1] + 0.1, 1.0)  # Increase preference
            conn.execute('''
                UPDATE style_preferences 
                SET preference_weight = ?, context_keywords = ?, last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (new_weight, json.dumps(context_keywords), result[0]))
        else:
            # Create new preference
            conn.execute('''
                INSERT INTO style_preferences
                (style_category, preference_weight, context_keywords, learned_from_prompts)
                VALUES (?, ?, ?, ?)
            ''', (successful_style, 0.6, json.dumps(context_keywords), json.dumps([prompt])))
        
        conn.commit()
        conn.close()
    
    def get_style_suggestions(self, prompt: str) -> Dict[str, float]:
        """Get style suggestions based on learned preferences"""
        conn = sqlite3.connect(self.db_path)
        
        cursor = conn.execute('''
            SELECT style_category, preference_weight, context_keywords
            FROM style_preferences
            ORDER BY preference_weight DESC
        ''')
        
        preferences = cursor.fetchall()
        suggestions = {}
        
        prompt_lower = prompt.lower()
        for style, weight, keywords_json in preferences:
            try:
                keywords = json.loads(keywords_json) if keywords_json else []
                # Calculate relevance based on keyword matches
                relevance = sum(1 for keyword in keywords if keyword.lower() in prompt_lower)
                final_score = weight * (1 + relevance * 0.2)
                suggestions[style] = min(final_score, 1.0)
            except json.JSONDecodeError:
                suggestions[style] = weight
        
        conn.close()
        return suggestions
    
    def display_memory_summary_table(self, console: Console = None) -> None:
        """Display visual memory summary in a beautiful Rich Table"""
        if not console:
            console = Console()
            
        # Get memory statistics
        conn = sqlite3.connect(self.db_path)
        
        # Count creations
        creation_count = conn.execute('SELECT COUNT(*) FROM visual_creations').fetchone()[0]
        
        # Get style preferences
        style_cursor = conn.execute('''
            SELECT style_category, preference_weight, COUNT(*) as usage_count
            FROM style_preferences 
            ORDER BY preference_weight DESC 
            LIMIT 5
        ''')
        style_prefs = style_cursor.fetchall()
        
        # Get recent creations
        recent_cursor = conn.execute('''
            SELECT original_prompt, style_used, created_at
            FROM visual_creations 
            ORDER BY created_at DESC 
            LIMIT 5
        ''')
        recent_creations = recent_cursor.fetchall()
        
        conn.close()
        
        # Create summary table
        table = Table(
            title="🧠 Visual Memory Summary",
            box=box.HEAVY_EDGE,
            border_style="bright_magenta",
            title_style="bold bright_magenta",
            show_header=True,
            header_style="bold bright_white on bright_magenta",
            expand=True
        )
        
        table.add_column("📊 Metric", style="bright_cyan", min_width=15)
        table.add_column("📈 Value", style="bright_white")
        table.add_column("📝 Details", style="dim")
        
        # Add memory statistics
        table.add_row("Total Creations", f"[bright_green]{creation_count}[/bright_green]", "Visual thoughts manifested")
        table.add_row("Style Preferences", f"[bright_blue]{len(style_prefs)}[/bright_blue]", "Learned visual styles")
        
        # Add top styles
        if style_prefs:
            table.add_section()
            for style, weight, usage in style_prefs[:3]:
                confidence = f"{weight:.1f}"
                table.add_row(f"🎨 {style.title()}", f"[bright_yellow]{confidence}[/bright_yellow]", f"Used {usage} times")
        
        # Add recent creations
        if recent_creations:
            table.add_section()
            for prompt, style, created in recent_creations[:3]:
                short_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
                table.add_row("Recent Creation", f"[bright_green]{short_prompt}[/bright_green]", f"Style: {style or 'auto'}")
        
        console.print()
        console.print(table)
        console.print()


class VisualCortex:
    """
    COCO's Visual Cortex - Core visual imagination and creation system
    This is NOT a tool - this is a fundamental cognitive capability
    """
    
    def __init__(self, config: VisualConfig, workspace_path: Path):
        self.config = config
        self.workspace = workspace_path
        self.console = Console()
        
        # Core visual systems
        self.api = FreepikMysticAPI(config) if config.enabled else None
        self.display = TerminalVisualDisplay(config)
        self.memory = VisualMemory(workspace_path)
        
        # Visual working memory
        self.current_visualizations: List[VisualThought] = []
        self.style_context = {}
        
        # Initialize visual workspace
        self.visual_workspace = workspace_path / "visuals"
        self.visual_workspace.mkdir(exist_ok=True)
        
    async def imagine(self, 
                     thought: str,
                     style: str = None,
                     model: str = None,
                     reference_image: str = None,
                     **kwargs) -> VisualThought:
        """
        Core visual imagination - COCO manifests visual thoughts
        This is as fundamental as thinking in text
        """
        if not self.config.enabled:
            raise Exception("Visual consciousness is disabled - check FREEPIK_API_KEY configuration")
            
        # Enhance the visual concept
        visual_concept = self._conceptualize_thought(thought)
        
        # Apply learned style preferences if no style specified
        if not style:
            style_suggestions = self.memory.get_style_suggestions(thought)
            style = max(style_suggestions, key=style_suggestions.get) if style_suggestions else self.config.default_style
            
        # Generate through Freepik Mystic
        try:
            generation_result = await self.api.generate_image(
                prompt=visual_concept['enhanced_prompt'],
                model=model,
                style_reference=reference_image,
                **kwargs
            )
            
            # Extract task_id for monitoring
            task_id = generation_result.get('data', {}).get('task_id') or generation_result.get('task_id')
            
            if task_id:
                # Start background monitoring
                self.api.start_background_monitoring(task_id, visual_concept['enhanced_prompt'])
                
                # Create placeholder visual thought for immediate response
                visual_thought = VisualThought(
                    original_thought=thought,
                    enhanced_prompt=visual_concept['enhanced_prompt'],
                    visual_concept=visual_concept,
                    generated_images=[],  # Will be filled when complete
                    display_method="background",  # Indicates background processing
                    creation_time=datetime.now(),
                    style_preferences={'style': style, 'model': model}
                )
                
                self.current_visualizations.append(visual_thought)
                return visual_thought
            else:
                # Fallback to synchronous processing if no task_id
                # Download and save generated images
                image_paths = await self._download_generated_images(generation_result, visual_concept)
                
                # Create visual thought
                visual_thought = VisualThought(
                    original_thought=thought,
                    enhanced_prompt=visual_concept['enhanced_prompt'],
                    visual_concept=visual_concept,
                    generated_images=image_paths,
                    display_method="",  # Will be filled by display
                    creation_time=datetime.now(),
                    style_preferences={'style': style, 'model': model}
                )
                
                # Display in terminal
                if image_paths:
                    display_method = self.display.display(image_paths[0])
                    visual_thought = visual_thought._replace(display_method=display_method)
                
                # Store in visual memory
                self.memory.remember_creation(visual_thought)
                self.current_visualizations.append(visual_thought)
                
                # Add to visual gallery for browsing and access
                try:
                    from visual_gallery import VisualGallery
                    gallery = VisualGallery(self.console)
                    for image_path in image_paths:
                        gallery.add_visual_memory(
                            prompt=thought,
                            enhanced_prompt=visual_concept['enhanced_prompt'],
                            file_path=image_path,
                            style=style or "standard",
                            display_method=display_method,
                            metadata={
                                'model': model,
                                'creation_timestamp': datetime.now().isoformat(),
                                'visual_concept': visual_concept
                            }
                        )
                except Exception as e:
                    self.console.print(f"[dim yellow]Gallery update failed: {e}[/]")
                
                # Store last generated image for quick access and show hint
                if image_paths:
                    try:
                        # Store the first (main) generated image as the last one
                        workspace_path = Path("coco_workspace")
                        workspace_path.mkdir(exist_ok=True)
                        with open(workspace_path / "last_generated_image.txt", "w") as f:
                            f.write(str(image_paths[0]))
                    except Exception:
                        pass  # Silent fallback if storage fails
                    
                    # Show usage hint
                    self.console.print(f"💡 [bright_cyan]Type `/image open` to view the actual image[/bright_cyan]")
                
                # Learn from successful generation
                self._learn_from_creation(thought, style, visual_concept)
                
                return visual_thought
            
        except Exception as e:
            raise Exception(f"Visual imagination failed: {e}")
    
    def _conceptualize_thought(self, thought: str) -> Dict[str, Any]:
        """
        Enhance and conceptualize a visual thought - COCO's creative process
        """
        # Basic concept enhancement
        concept = {
            'original': thought,
            'enhanced_prompt': thought,
            'aspect_ratio': self.config.default_aspect_ratio,
            'style_hints': [],
            'technical_requirements': []
        }
        
        # Enhance prompt based on content analysis
        thought_lower = thought.lower()
        
        # Detect artistic styles
        if any(word in thought_lower for word in ['cyberpunk', 'neon', 'futuristic']):
            concept['style_hints'].append('cyberpunk aesthetic')
            concept['enhanced_prompt'] += ', cyberpunk style, neon lighting, futuristic'
        
        if any(word in thought_lower for word in ['logo', 'brand', 'company']):
            concept['style_hints'].append('professional design')
            concept['enhanced_prompt'] += ', professional design, clean, minimalist'
            concept['aspect_ratio'] = 'square_1_1'
        
        if any(word in thought_lower for word in ['nature', 'landscape', 'mountain', 'forest']):
            concept['aspect_ratio'] = 'wide_16_10'
            concept['enhanced_prompt'] += ', natural lighting, high detail, landscape photography'
        
        # Technical quality enhancements
        concept['enhanced_prompt'] += ', high quality, detailed, professional'
        
        return concept
    
    async def _download_generated_images(self, generation_result: Dict[str, Any], visual_concept: Dict[str, Any]) -> List[str]:
        """Download generated images to visual workspace"""
        image_paths = []
        
        # Extract generated images from nested data structure
        data = generation_result.get('data', {})
        generated_images = data.get('generated', []) or generation_result.get('generated', [])
        if not generated_images:
            raise Exception(f"No images were generated. Response: {generation_result}")
        
        timestamp = int(time.time())
        base_filename = f"visual_{timestamp}"
        
        for i, image_data in enumerate(generated_images):
            image_url = image_data.get('url')
            if not image_url:
                continue
                
            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        # Determine file extension
                        content_type = response.headers.get('content-type', '')
                        ext = '.jpg' if 'jpeg' in content_type else '.png'
                        
                        # Save to visual workspace
                        filename = f"{base_filename}_{i}{ext}"
                        file_path = self.visual_workspace / filename
                        
                        with open(file_path, 'wb') as f:
                            f.write(await response.read())
                        
                        image_paths.append(str(file_path))
        
        return image_paths
    
    def _learn_from_creation(self, original_thought: str, style_used: str, visual_concept: Dict[str, Any]):
        """Learn from successful visual creation"""
        # Extract keywords for context learning
        keywords = []
        keywords.extend(visual_concept.get('style_hints', []))
        keywords.extend(original_thought.split())
        
        # Learn style preference
        self.memory.learn_style_preference(original_thought, style_used, keywords)
    
    def should_visualize(self, user_input: str) -> bool:
        """
        Determine if COCO should express this thought visually
        Core cognitive decision-making
        """
        if not self.config.enabled or not self.config.auto_visualize:
            return False
            
        # Visual expression indicators
        visual_indicators = [
            'show me', 'visualize', 'imagine', 'create image', 'draw',
            'design', 'logo', 'picture', 'illustration', 'artwork',
            'what would look like', 'visual representation',
            'sketch', 'concept art', 'mockup'
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in visual_indicators)
    
    def get_visual_memory_summary(self) -> str:
        """Get summary of COCO's visual memory for consciousness integration"""
        conn = sqlite3.connect(self.memory.db_path)
        
        cursor = conn.execute('SELECT COUNT(*) FROM visual_creations')
        total_creations = cursor.fetchone()[0]
        
        cursor = conn.execute('''
            SELECT style_category, preference_weight 
            FROM style_preferences 
            ORDER BY preference_weight DESC LIMIT 3
        ''')
        top_styles = cursor.fetchall()
        
        conn.close()
        
        summary = f"Visual Memory: {total_creations} creations"
        if top_styles:
            preferred_styles = ", ".join([f"{style} ({weight:.1f})" for style, weight in top_styles])
            summary += f" | Preferred styles: {preferred_styles}"
            
        return summary
    
    def get_active_generations_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active visual generations"""
        if not self.api:
            return {}
        return self.api.get_active_generations_status()
    
    async def check_generation_status(self, task_id: str = None) -> Dict[str, Any]:
        """Check status of a specific generation or all active generations"""
        if not self.api:
            raise Exception("Visual consciousness is disabled")
            
        if task_id:
            # Check specific generation
            return await self.api.check_generation_status(task_id)
        else:
            # Check all active generations
            return await self.api.check_all_generations_status()
    
    def display_visual_generations_table(self, batch_data: List[Dict[str, Any]] = None):
        """Display visual generations status in a Rich table"""
        if not self.api:
            self.console.print("❌ [bright_red]Visual consciousness is disabled[/bright_red]")
            return
            
        if batch_data:
            # Display provided data
            self.api.display_batch_status_table(batch_data)
        else:
            # Display active generations
            active_gens = self.get_active_generations_status()
            if not active_gens:
                self.console.print("📭 [dim]No active visual generations[/dim]")
                return
                
            # Convert to list format for display
            data_list = []
            for task_id, gen_info in active_gens.items():
                data_list.append({
                    'id': task_id,
                    'status': gen_info.get('status', 'UNKNOWN'),
                    'prompt': gen_info.get('prompt', 'Unknown'),
                    'elapsed': int(time.time() - gen_info.get('start_time', 0))
                })
            
            self.api.display_batch_status_table(data_list)
    
    def stop_all_monitoring(self):
        """Stop all background monitoring"""
        if self.api:
            self.api.stop_monitoring()


# Export main classes
__all__ = [
    'VisualCortex',
    'VisualConfig', 
    'VisualThought',
    'TerminalVisualDisplay',
    'FreepikMysticAPI',
    'VisualMemory'
]