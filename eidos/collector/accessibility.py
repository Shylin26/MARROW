from AppKit import NSWorkspace
from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCopyAttributeValue,
    kAXFocusedWindowAttribute,
    kAXTitleAttribute
)
import logging

class AccessibilityCollector:
    def __init__(self):
        
        self.workspace = NSWorkspace.sharedWorkspace()

    def get_info(self):
        """Returns the active app name and window title."""
        try:
            
            frontmost_app = self.workspace.frontmostApplication()
            if not frontmost_app:
                return {"app_name": "Unknown", "window_title": "Unknown"}

            app_name = frontmost_app.localizedName()
            pid = frontmost_app.processIdentifier()

            
            app_element = AXUIElementCreateApplication(pid)
            
            
            error, window_element = AXUIElementCopyAttributeValue(
                app_element, kAXFocusedWindowAttribute, None
            )
            
            if error != 0 or not window_element:
                return {"app_name": app_name, "window_title": ""}

            
            error, title = AXUIElementCopyAttributeValue(
                window_element, kAXTitleAttribute, None
            )
            
            if error != 0 or not title:
                return {"app_name": app_name, "window_title": ""}

            return {
                "app_name": app_name,
                "window_title": title
            }
        except Exception as e:
            logging.error(f"Accessibility Error: {e}")
            return {"app_name": "Error", "window_title": ""}
