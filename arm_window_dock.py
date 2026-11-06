"""Dock the Tk panel against the visible frame of this process's GLFW viewer."""
import ctypes as ct
from ctypes import wintypes as wt
import os

user = ct.WinDLL("user32")
dwm = ct.WinDLL("dwmapi")


def api(lib, name, restype, *args):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, restype
    return fn


class MonitorInfo(ct.Structure):
    _fields_ = [("size", wt.DWORD), ("monitor", wt.RECT), ("work", wt.RECT), ("flags", wt.DWORD)]


find = api(user, "FindWindowExW", wt.HWND, wt.HWND, wt.HWND, wt.LPCWSTR, wt.LPCWSTR)
pid_for = api(user, "GetWindowThreadProcessId", wt.DWORD, wt.HWND, ct.POINTER(wt.DWORD))
ancestor = api(user, "GetAncestor", wt.HWND, wt.HWND, wt.UINT)
rect_for = api(user, "GetWindowRect", wt.BOOL, wt.HWND, ct.POINTER(wt.RECT))
visible_rect = api(dwm, "DwmGetWindowAttribute", ct.c_long, wt.HWND, wt.DWORD, ct.c_void_p, wt.DWORD)
monitor_for = api(user, "MonitorFromWindow", wt.HANDLE, wt.HWND, wt.DWORD)
monitor_info = api(user, "GetMonitorInfoW", wt.BOOL, wt.HANDLE, ct.POINTER(MonitorInfo))
move = api(user, "SetWindowPos", wt.BOOL, wt.HWND, wt.HWND, ct.c_int, ct.c_int, ct.c_int, ct.c_int, wt.UINT)
dpi = api(user, "SetThreadDpiAwarenessContext", wt.HANDLE, wt.HANDLE)
minimized = api(user, "IsIconic", wt.BOOL, wt.HWND)
maximized = api(user, "IsZoomed", wt.BOOL, wt.HWND)
show = api(user, "ShowWindow", wt.BOOL, wt.HWND, ct.c_int)
set_title = api(user,"SetWindowTextW",wt.BOOL,wt.HWND,wt.LPCWSTR)
message = api(user,"SendMessageW",ct.c_ssize_t,wt.HWND,wt.UINT,wt.WPARAM,wt.LPARAM)


def viewer_window():
    hwnd = None
    while True:
        hwnd = find(None, hwnd, "GLFW30", None)
        if not hwnd:
            return None
        pid = wt.DWORD()
        pid_for(hwnd, ct.byref(pid))
        if pid.value == os.getpid():
            return hwnd


def bounds(hwnd):
    outer, frame = wt.RECT(), wt.RECT()
    rect_for(hwnd, ct.byref(outer))
    if visible_rect(hwnd, 9, ct.byref(frame), ct.sizeof(frame)) != 0:
        frame = outer
    return outer, frame


def place(hwnd, outer, frame, x, y, w, h):
    if (frame.left, frame.top, frame.right-frame.left, frame.bottom-frame.top) != (x,y,w,h):
        move(hwnd, None, x+outer.left-frame.left, y+outer.top-frame.top,
             w+outer.right-outer.left-frame.right+frame.left,
             h+outer.bottom-outer.top-frame.bottom+frame.top, 0x14)  # no focus or z-order change


def dock(root, title=None):
    viewer = viewer_window()
    if not viewer or minimized(viewer) or not root.winfo_ismapped():
        return
    panel = ancestor(root.winfo_id(), 2)
    if not panel or minimized(panel):
        return
    if title:
        set_title(viewer,title)
        for size in (0,1):
            icon = message(panel,0x7F,size,0)  # WM_GETICON
            if icon:
                message(viewer,0x80,size,icon)  # WM_SETICON
    previous = dpi(-4)  # DWM reports physical pixels, including on scaled monitors.
    try:
        info = MonitorInfo(size=ct.sizeof(MonitorInfo))
        if not monitor_info(monitor_for(viewer, 2), ct.byref(info)):
            return
        area = info.work
        maxim = maximized(viewer)
        if maxim:
            show(viewer, 9)
        outer, frame = bounds(viewer)
        po, pf = bounds(panel)
        width = min(pf.right-pf.left, (area.right-area.left)*2//3)
        available = area.right-area.left-width
        vw = available if maxim else min(frame.right-frame.left, available)
        vh = area.bottom-area.top if maxim else min(frame.bottom-frame.top, area.bottom-area.top)
        x = area.left if maxim else max(area.left,min(frame.left,area.right-width-vw))
        y = area.top if maxim else max(area.top,min(frame.top,area.bottom-vh))
        place(viewer,outer,frame,x,y,vw,vh)
        place(panel,po,pf,x+vw,y,width,vh)
    finally:
        if previous:
            dpi(previous)
