#!/usr/bin/env python3
"""
Telegrip VR Monitor - Independent VR control information monitoring script
Simplified version for LeMelon VR teleoperation
"""

# Standard library imports
import os
import sys
import asyncio
import logging
import threading
import http.server
import ssl
import socket

# Set the absolute path to the telegrip folder
TELEGRIP_PATH = "/home/owen/telegrip"


def setup_telegrip_environment():
    """Setup telegrip environment"""
    # Add telegrip path to Python path
    if TELEGRIP_PATH not in sys.path:
        sys.path.insert(0, TELEGRIP_PATH)

    # Set working directory
    os.chdir(TELEGRIP_PATH)

    # Set environment variables
    env_pythonpath = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = f"{TELEGRIP_PATH}:{env_pythonpath}"


def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Connect to a remote address to determine the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            # Fallback: get hostname IP
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            # Final fallback
            return "localhost"


def find_available_port(start_port=8443, max_attempts=10):
    """
    Find an available port starting from start_port.
    Prevents 'Address already in use' errors.
    """
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                logger.info(f"✅ Found available HTTPS port: {port}")
                return port
        except OSError as e:
            logger.debug(f"Port {port} unavailable: {e}")
            continue
    raise RuntimeError(
        f"❌ No available ports found in range {start_port}-{start_port+max_attempts-1}. "
        f"Try: sudo lsof -ti:{start_port} | xargs kill -9"
    )


def import_telegrip_modules():
    """Import telegrip modules without robot interface dependencies"""
    try:
        # Import modules directly to avoid triggering robot interface imports
        from telegrip.config import TelegripConfig
        from telegrip.inputs.vr_ws_server import VRWebSocketServer
        from telegrip.inputs.base import ControlGoal, ControlMode
        return TelegripConfig, VRWebSocketServer, ControlGoal, ControlMode
    except ImportError as e:
        print(f"Error importing telegrip modules: {e}")
        print(f"Make sure TELEGRIP_PATH is correct: {TELEGRIP_PATH}")
        print(f"Also ensure required packages are installed (websockets, etc.)")
        return None, None, None, None


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleAPIHandler(http.server.BaseHTTPRequestHandler):
    """Simplified HTTP request handler, only provides basic web services"""

    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        try:
            super().end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
                ssl.SSLError):
            pass

    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        """Override to reduce HTTP request logging noise."""
        pass  # Disable default HTTP logging

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self.serve_file('web-ui/index.html', 'text/html')
        elif self.path.endswith('.css'):
            self.serve_file(f'web-ui{self.path}', 'text/css')
        elif self.path.endswith('.js'):
            self.serve_file(f'web-ui{self.path}', 'application/javascript')
        elif self.path.endswith('.ico'):
            self.serve_file(self.path[1:], 'image/x-icon')
        elif self.path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            if self.path.endswith(('.jpg', '.jpeg')):
                content_type = 'image/jpeg'
            elif self.path.endswith('.png'):
                content_type = 'image/png'
            else:
                content_type = 'image/gif'
            self.serve_file(f'web-ui{self.path}', content_type)
        else:
            self.send_error(404, "Not found")

    def serve_file(self, filename, content_type):
        """Serve a file with the given content type."""
        try:
            web_root = getattr(self.server, 'web_root_path', TELEGRIP_PATH)
            file_path = os.path.join(web_root, filename)

            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, f"File not found: {filename}")
        except Exception as e:
            print(f"Error serving file {filename}: {e}")
            self.send_error(500, "Internal server error")


class SimpleHTTPSServer:
    """Simplified HTTPS server for providing web interface"""

    def __init__(self, config):
        self.config = config
        self.httpd = None
        self.server_thread = None
        self.web_root_path = TELEGRIP_PATH

    async def start(self):
        """Start the HTTPS server."""
        try:
            self.httpd = http.server.HTTPServer(
                (self.config.host_ip, self.config.https_port),
                SimpleAPIHandler
            )

            self.httpd.web_root_path = self.web_root_path

            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain('cert.pem', 'key.pem')
            self.httpd.socket = context.wrap_socket(
                self.httpd.socket, server_side=True
            )

            self.server_thread = threading.Thread(
                target=self.httpd.serve_forever, daemon=True
            )
            self.server_thread.start()

            print(f"🌐 HTTPS server started on "
                  f"{self.config.host_ip}:{self.config.https_port}")

        except Exception as e:
            print(f"❌ Failed to start HTTPS server: {e}")
            raise

    async def stop(self):
        """Stop the HTTPS server."""
        if self.httpd:
            self.httpd.shutdown()
            if self.server_thread:
                self.server_thread.join(timeout=5)
            print("🌐 HTTPS server stopped")


class VRMonitor:
    """VR control information monitor"""

    def __init__(self):
        self.config = None
        self.vr_server = None
        self.https_server = None
        self.is_running = False
        self.latest_goal = None
        self.left_goal = None
        self.right_goal = None
        self.headset_goal = None
        self._goal_lock = threading.Lock()

    def initialize(self):
        """Initialize VR monitor"""
        print("🔧 Initializing Telegrip VR Monitor...")

        setup_telegrip_environment()

        (TelegripConfig, VRWebSocketServer, ControlGoal, ControlMode
         ) = import_telegrip_modules()
        if TelegripConfig is None:
            print("❌ Failed to import telegrip modules")
            return False

        self.config = TelegripConfig()
        self.config.enable_vr = True
        self.config.enable_keyboard = False
        self.config.enable_https = True
        
        # Find available port to prevent "Address already in use" errors
        try:
            available_port = find_available_port(start_port=8443, max_attempts=10)
            self.config.https_port = available_port
            logger.info(f"🌐 Using HTTPS port: {available_port}")
        except RuntimeError as e:
            print(f"❌ Port allocation failed: {e}")
            return False

        self.command_queue = asyncio.Queue()

        try:
            self.vr_server = VRWebSocketServer(
                command_queue=self.command_queue,
                config=self.config
            )
        except Exception as e:
            print(f"❌ Failed to create VR WebSocket server: {e}")
            return False

        try:
            self.https_server = SimpleHTTPSServer(self.config)
        except Exception as e:
            print(f"❌ Failed to create HTTPS server: {e}")
            return False

        print("✅ Telegrip VR Monitor initialized successfully")
        return True

    async def start_monitoring(self):
        """Start monitoring VR control information"""
        print("🚀 Starting VR Monitor...")

        # Only initialize if not already initialized
        if self.vr_server is None:
            if not self.initialize():
                print("❌ Failed to initialize VR monitor")
                return

        try:
            await self.https_server.start()
            await self.vr_server.start()

            self.is_running = True
            print("✅ VR Monitor is now running")

            if self.config.host_ip == "0.0.0.0":
                host_display = get_local_ip()
            else:
                host_display = self.config.host_ip
            print("📱 Open your VR headset browser and navigate to:")
            print(f"   https://{host_display}:{self.config.https_port}")
            print("🎯 Press Ctrl+C to stop monitoring")
            print()

            await self.monitor_commands()

        except KeyboardInterrupt:
            print("\n⏹️  Stopping VR monitor...")
        except Exception as e:
            print(f"❌ Error in VR monitor: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            await self.stop_monitoring()

    async def monitor_commands(self):
        """Monitor commands from VR controllers"""
        print("📊 Monitoring VR control commands...")

        while self.is_running:
            try:
                goal = await asyncio.wait_for(
                    self.command_queue.get(), timeout=1.0
                )

                with self._goal_lock:
                    if goal.arm == "left":
                        self.left_goal = goal
                    elif goal.arm == "right":
                        self.right_goal = goal
                    elif goal.arm == "headset" or goal.arm == "base":
                        # Merge base/headset goals
                        if self.headset_goal is None:
                            self.headset_goal = goal
                        else:
                            # Preserve angular velocity if not in new goal
                            prev_angular = getattr(self.headset_goal, 'base_angular_velocity', None)
                            self.headset_goal = goal
                            new_angular = getattr(goal, 'base_angular_velocity', None)
                            if new_angular is None and prev_angular is not None:
                                self.headset_goal.base_angular_velocity = prev_angular

                    self.latest_goal = goal

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"❌ Error processing command: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")

    def get_latest_goal_nowait(self, arm=None):
        """Return the latest VR control goal if available, else None."""
        with self._goal_lock:
            if arm == "left":
                return self.left_goal
            elif arm == "right":
                return self.right_goal
            elif arm == "headset":
                return self.headset_goal
            else:
                return {
                    "left": self.left_goal,
                    "right": self.right_goal,
                    "headset": self.headset_goal,
                }

    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False

        if self.vr_server:
            await self.vr_server.stop()

        if self.https_server:
            await self.https_server.stop()

        print("✅ VR Monitor stopped")


def main():
    """Main function"""
    print("🎮 Telegrip VR Monitor - Telegrip VR Control Information Monitor")
    print("=" * 60)

    if not os.path.exists(TELEGRIP_PATH):
        print(f"❌ Telegrip path does not exist: {TELEGRIP_PATH}")
        print("Please update TELEGRIP_PATH in the script")
        return

    monitor = VRMonitor()

    try:
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        print("\n👋 Telegrip VR Monitor stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
