# Currently for Windows (250114)
import ctypes

def get_screen_resolution():
    """return resolution (width, height)."""
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()  # Ignore DPI scaling (essential)
    width = int(user32.GetSystemMetrics(0))
    height = int(user32.GetSystemMetrics(1))
    return width, height

def get_dpi_scaling():
    """Return current DPI scaling ratio"""
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2) # Windows 10 version 1607 or more
    except AttributeError:
        pass # Ignore if "shcore.SetProcessDpiAwareness" not exists (For old version Windows)
    hdc = user32.GetDC(0)
    dpi_x = gdi32.GetDeviceCaps(hdc, 88)  # HORZRES
    user32.ReleaseDC(0, hdc)
    scaling_factor = dpi_x / 96.0  # Default DPI: 96
    return scaling_factor

# if __name__ == "__main__":
#     width, height = get_screen_resolution()
#     scaling = get_dpi_scaling()
#     print(f"현재 화면 해상도: {width} x {height}")
#     print(f"DPI 스케일링: {scaling:.2f}")

#     # DPI 스케일링을 고려한 논리적 해상도 계산
#     logical_width = int(width / scaling)
#     logical_height = int(height / scaling)
#     print(f"논리적 해상도 (DPI 고려): {logical_width} x {logical_height}")