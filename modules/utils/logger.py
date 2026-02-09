"""
Enhanced Logging Utility for OCL Framework
Provides comprehensive logging with levels, timestamps, and debugging support.
"""
import sys
from datetime import datetime
from typing import Optional
from enum import Enum


class LogLevel(Enum):
    """Log levels for controlling verbosity."""
    DEBUG = 0    # Detailed debugging information
    INFO = 1     # General information
    STEP = 2     # Major step completed
    SUCCESS = 3  # Success message
    WARNING = 4  # Warning message
    ERROR = 5    # Error message


class FrameworkLogger:
    """Enhanced logger for OCL framework with structured output."""
    
    def __init__(self, verbose: bool = True, debug: bool = False, name: str = "Framework"):
        """
        Initialize logger.
        
        Args:
            verbose: Print INFO and above
            debug: Print DEBUG and above
            name: Logger name prefix
        """
        self.verbose = verbose
        self.debug_enabled = debug  # Renamed to avoid shadowing debug() method
        self.name = name
        self.indent_level = 0
        
        # Set minimum level
        if debug:
            self.min_level = LogLevel.DEBUG
        elif verbose:
            self.min_level = LogLevel.INFO
        else:
            self.min_level = LogLevel.WARNING
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged."""
        return level.value >= self.min_level.value
    
    def _format_message(self, level: LogLevel, message: str, prefix: str = "") -> str:
        """Format log message with timestamp and level."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        indent = "  " * self.indent_level
        
        # Plain text prefixes (no symbols)
        icons = {
            LogLevel.DEBUG: "",
            LogLevel.INFO: "",
            LogLevel.STEP: "",
            LogLevel.SUCCESS: "",
            LogLevel.WARNING: "",
            LogLevel.ERROR: ""
        }
        
        icon = icons.get(level, "")
        
        if level == LogLevel.DEBUG:
            return f"[{timestamp}] {indent}{icon}DEBUG: {prefix}{message}"
        else:
            return f"[{timestamp}] {indent}{icon}{prefix}{message}"
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        if self._should_log(LogLevel.DEBUG):
            print(self._format_message(LogLevel.DEBUG, message), **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        if self._should_log(LogLevel.INFO):
            print(self._format_message(LogLevel.INFO, message), **kwargs)
    
    def step(self, message: str, **kwargs):
        """Log major step."""
        if self._should_log(LogLevel.STEP):
            print(self._format_message(LogLevel.STEP, message), **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message."""
        if self._should_log(LogLevel.SUCCESS):
            print(self._format_message(LogLevel.SUCCESS, message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        if self._should_log(LogLevel.WARNING):
            print(self._format_message(LogLevel.WARNING, message), **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        if self._should_log(LogLevel.ERROR):
            print(self._format_message(LogLevel.ERROR, message), file=sys.stderr, **kwargs)
    
    def section(self, title: str, char: str = "=", width: int = 70):
        """Print section header."""
        if self._should_log(LogLevel.INFO):
            print(f"\n{char * width}")
            print(title)
            print(f"{char * width}")
    
    def subsection(self, title: str, char: str = "-", width: int = 70):
        """Print subsection header."""
        if self._should_log(LogLevel.INFO):
            print(f"\n{char * width}")
            print(title)
            print(f"{char * width}")
    
    def progress(self, current: int, total: int, item: str = "item"):
        """Log progress."""
        if self._should_log(LogLevel.INFO):
            percentage = (current / total * 100) if total > 0 else 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {current}/{total} ({percentage:.1f}%) {item}s")
    
    def indent(self):
        """Increase indentation level."""
        self.indent_level += 1
    
    def dedent(self):
        """Decrease indentation level."""
        self.indent_level = max(0, self.indent_level - 1)
    
    def statistics(self, stats: dict, title: str = "Statistics"):
        """Print statistics dictionary."""
        if self._should_log(LogLevel.INFO):
            print(f"\n{'='*70}")
            print(f"{title}")
            print(f"{'='*70}")
            for key, value in stats.items():
                key_formatted = key.replace('_', ' ').title()
                print(f"  {key_formatted}: {value}")
            print(f"{'='*70}\n")


# Global logger instance
_global_logger: Optional[FrameworkLogger] = None


def get_logger(name: str = "Framework", verbose: bool = True, debug: bool = False) -> FrameworkLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = FrameworkLogger(verbose=verbose, debug=debug, name=name)
    return _global_logger


def set_logger(logger: FrameworkLogger):
    """Set global logger instance."""
    global _global_logger
    _global_logger = logger
