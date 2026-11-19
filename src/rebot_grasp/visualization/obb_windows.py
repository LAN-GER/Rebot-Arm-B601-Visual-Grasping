"""OBB diagnostic views backed by the shared point-cloud windows."""
from rebot_grasp.visualization.cloud_windows import CloudDiagnosticWindows, cloud_display_data, diagnostic_image

OBBDiagnosticWindows = CloudDiagnosticWindows
__all__ = ["OBBDiagnosticWindows", "cloud_display_data", "diagnostic_image"]
