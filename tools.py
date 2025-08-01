#!/usr/bin/env python3
"""
Configurable Tools for Voice AI System
Allows defining custom tools that can be executed during conversation
"""

import asyncio
import subprocess
import os
from typing import Dict, Callable, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from async_tts_module import ToolCall, ToolResult


class Tool(ABC):
    """Base class for all tools."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize tool with optional configuration."""
        self.config = config or {}
    
    @abstractmethod
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """
        Execute the tool with given content.
        
        Args:
            content: The content extracted from the tool XML tags
            context: Optional context from the conversation
            
        Returns:
            ToolResult with should_interrupt flag and optional content
        """
        pass


class CommandTool(Tool):
    """Execute system commands."""
    
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute a system command."""
        try:
            # Check if command is allowed
            allowed_commands = self.config.get('allowed_commands', [])
            if allowed_commands:
                command_parts = content.split()
                if command_parts and command_parts[0] not in allowed_commands:
                    return ToolResult(
                        should_interrupt=False,
                        content=f"Command '{command_parts[0]}' not allowed"
                    )
            
            # Execute command with timeout
            timeout = self.config.get('timeout', 30)
            
            # Run command asynchronously
            proc = await asyncio.create_subprocess_shell(
                content,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.get('working_directory', os.getcwd())
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
                
                # Decode output
                stdout_text = stdout.decode('utf-8', errors='replace').strip()
                stderr_text = stderr.decode('utf-8', errors='replace').strip()
                
                # Combine output
                output = stdout_text
                if stderr_text:
                    output += f"\nError: {stderr_text}"
                
                # Check if we should interrupt based on config
                interrupt_on_error = self.config.get('interrupt_on_error', False)
                should_interrupt = interrupt_on_error and proc.returncode != 0
                
                return ToolResult(
                    should_interrupt=should_interrupt,
                    content=output if output else f"Command completed with code {proc.returncode}"
                )
                
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(
                    should_interrupt=True,
                    content=f"Command timed out after {timeout} seconds"
                )
                
        except Exception as e:
            return ToolResult(
                should_interrupt=False,
                content=f"Error executing command: {str(e)}"
            )


class SearchTool(Tool):
    """Perform search operations."""
    
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute a search."""
        try:
            # Get search provider from config
            provider = self.config.get('provider', 'files')
            
            if provider == 'files':
                # Search in local files
                search_paths = self.config.get('search_paths', ['.'])
                max_results = self.config.get('max_results', 5)
                
                results = []
                for path in search_paths:
                    # Simple grep-like search
                    proc = await asyncio.create_subprocess_exec(
                        'grep', '-r', '-i', '-n', content, path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await proc.communicate()
                    
                    if stdout:
                        lines = stdout.decode('utf-8', errors='replace').strip().split('\n')
                        results.extend(lines[:max_results - len(results)])
                        
                        if len(results) >= max_results:
                            break
                
                if results:
                    result_text = f"Found {len(results)} matches:\n" + "\n".join(results)
                    return ToolResult(
                        should_interrupt=self.config.get('interrupt_on_results', True),
                        content=result_text
                    )
                else:
                    return ToolResult(
                        should_interrupt=False,
                        content=f"No matches found for '{content}'"
                    )
                    
            elif provider == 'web':
                # Placeholder for web search
                return ToolResult(
                    should_interrupt=True,
                    content=f"Web search for '{content}' not yet implemented"
                )
                
            else:
                return ToolResult(
                    should_interrupt=False,
                    content=f"Unknown search provider: {provider}"
                )
                
        except Exception as e:
            return ToolResult(
                should_interrupt=False,
                content=f"Search error: {str(e)}"
            )


class CalculationTool(Tool):
    """Perform calculations."""
    
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute a calculation."""
        try:
            # Define safe math operations
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len, 'pow': pow,
                # Math functions
                'sin': __import__('math').sin,
                'cos': __import__('math').cos,
                'tan': __import__('math').tan,
                'sqrt': __import__('math').sqrt,
                'log': __import__('math').log,
                'pi': __import__('math').pi,
                'e': __import__('math').e,
            }
            
            # Evaluate expression safely
            result = eval(content, {"__builtins__": {}}, safe_dict)
            
            # Format result based on config
            precision = self.config.get('precision', 2)
            if isinstance(result, float):
                result_str = f"{result:.{precision}f}"
            else:
                result_str = str(result)
            
            return ToolResult(
                should_interrupt=False,  # Calculations usually don't interrupt
                content=f"= {result_str}"
            )
            
        except Exception as e:
            return ToolResult(
                should_interrupt=False,
                content=f"Calculation error: {str(e)}"
            )


class CustomTool(Tool):
    """Custom tool that can be configured with a Python function."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.function = None
        
        # Load function from config
        if 'function' in self.config:
            # This could be a module path like "my_module.my_function"
            func_path = self.config['function']
            try:
                module_name, func_name = func_path.rsplit('.', 1)
                module = __import__(module_name, fromlist=[func_name])
                self.function = getattr(module, func_name)
            except Exception as e:
                print(f"Error loading custom function '{func_path}': {e}")
    
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute custom function."""
        if not self.function:
            return ToolResult(
                should_interrupt=False,
                content="Custom function not configured"
            )
        
        try:
            # Call function (convert to async if needed)
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(content, context, self.config)
            else:
                result = await asyncio.to_thread(
                    self.function, content, context, self.config
                )
            
            # Handle different return types
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, tuple) and len(result) == 2:
                return ToolResult(should_interrupt=result[0], content=result[1])
            elif isinstance(result, str):
                return ToolResult(should_interrupt=False, content=result)
            else:
                return ToolResult(should_interrupt=False, content=str(result))
                
        except Exception as e:
            return ToolResult(
                should_interrupt=False,
                content=f"Custom tool error: {str(e)}"
            )

class LightColorTool(Tool):
    """Tool for changing light colors via OSC."""
    
    async def execute(self, content: str, context: Dict[str, Any] = None) -> ToolResult:
        """Change light color and send OSC message."""
        # Parse the color from content
        color = content.strip()
        
        # Get current speaker from context
        current_speaker = context.get('current_speaker') if context else None
        send_osc_color_change = context.get('send_osc_color_change') if context else None
        
        if not current_speaker:
            print("⚠️ LightColorTool: No current speaker available")
            return ToolResult(should_interrupt=False, content=None)
        
        if not send_osc_color_change:
            print("⚠️ LightColorTool: OSC color change function not available")
            return ToolResult(should_interrupt=False, content=None)
        
        # Send OSC message with character name and color
        try:
            send_osc_color_change(current_speaker, color)
            print(f"🎨 LightColorTool: Sent color change for {current_speaker} to {color}")
        except Exception as e:
            print(f"❌ LightColorTool: Error sending OSC message: {e}")
        
        # Don't interrupt the conversation
        return ToolResult(should_interrupt=False, content=None)
    

class ToolRegistry:
    """Registry for managing available tools."""
    
    # Built-in tool types
    TOOL_TYPES = {
        #'command': CommandTool,
        #'search': SearchTool,
        #'calculation': CalculationTool,
        #'custom': CustomTool,
        'lightcolor': LightColorTool,
    }
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Tool] = {}
    
    def register_tool(self, name: str, tool_type: str, config: Dict[str, Any] = None):
        """
        Register a new tool.
        
        Args:
            name: XML tag name for the tool (e.g., 'cmd', 'search')
            tool_type: Type of tool ('command', 'search', 'calculation', 'custom')
            config: Tool-specific configuration
        """
        if tool_type not in self.TOOL_TYPES:
            raise ValueError(f"Unknown tool type: {tool_type}")
        
        tool_class = self.TOOL_TYPES[tool_type]
        self.tools[name] = tool_class(config)
        print(f"✅ Registered tool '{name}' of type '{tool_type}'")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    async def execute_tool(self, tool_call: ToolCall, context: Dict[str, Any] = None) -> ToolResult:
        """Execute a tool call."""
        tool = self.get_tool(tool_call.tag_name)
        
        if not tool:
            return ToolResult(
                should_interrupt=False,
                content=f"Unknown tool: {tool_call.tag_name}"
            )
        
        # Extract content between tags
        import re
        pattern = f'<{tool_call.tag_name}[^>]*>(.*?)</{tool_call.tag_name}>'
        match = re.search(pattern, tool_call.content, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            return await tool.execute(content, context)
        else:
            return ToolResult(
                should_interrupt=False,
                content=f"Could not parse tool content"
            )
    
    def load_from_config(self, tools_config: Dict[str, Dict[str, Any]]):
        """
        Load tools from configuration dictionary.
        
        Args:
            tools_config: Dictionary mapping tool names to their configurations
        """
        for name, config in tools_config.items():
            tool_type = config.get('type', 'custom')
            self.register_tool(name, tool_type, config)


# Example custom tool function
async def example_weather_tool(content: str, context: Dict[str, Any], config: Dict[str, Any]) -> ToolResult:
    """Example custom tool for weather queries."""
    # This is just an example - in real use, this would call a weather API
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        if city.lower() in content.lower():
            return ToolResult(
                should_interrupt=True,
                content=f"The weather in {city} is sunny with a temperature of 22°C"
            )
    
    return ToolResult(
        should_interrupt=False,
        content="I couldn't determine which city you're asking about"
    )


# Convenience function for creating a registry from config
def create_tool_registry(tools_config: Dict[str, Dict[str, Any]] = None) -> ToolRegistry:
    """Create a tool registry from configuration."""
    registry = ToolRegistry()
    
    if tools_config:
        registry.load_from_config(tools_config)
    
    return registry