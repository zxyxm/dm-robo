"""Interactive simulation and read-only CAN twin for Assembly 4."""
from contextlib import closing
from queue import Empty, SimpleQueue
from threading import Event, Thread
import time
import tkinter as tk
from tkinter import ttk

import mujoco
from mujoco import viewer
import numpy as np
from serial.tools.list_ports import comports

from arm4_fk import load_model
from arm4_twin import read_positions, update_pose
from dm_debug import CanBus, Joint, U2Can, ids


def stream(stop, events, interface, channel, rate, slaves, masters, zero, signs):
    try:
        bus = U2Can(channel, rate) if interface == "u2can" else CanBus(interface, channel, rate)
        with closing(bus):
            joints = [Joint(s, m) for s, m in zip(slaves, masters)]
            while not stop.is_set():
                values = read_positions(bus, joints)
                events.put(("sample", signs * (values - zero)))
                stop.wait(.04)
    except Exception as exc:
        events.put(("error", str(exc)))
    finally:
        events.put(("closed", None))


class Panel:
    def __init__(self, root):
        self.root = root
        root.title("达妙六轴机械臂 · 孪生操作面板")
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.view = None
        self.worker = None
        self.stop = Event()
        self.events = SimpleQueue()
        self.mode = "sim"
        self.q = np.zeros(6)
        self.running = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="仿真模式 · 可拖动关节或播放演示")
        self.pose = tk.StringVar()
        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="装配体 4   /   六轴操作面板", font=("Microsoft YaHei UI", 14)).pack(anchor="w")
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=8)
        self.connect_btn = ttk.Button(bar, text="连接硬件跟随", command=self.connect)
        self.connect_btn.pack(side="left")
        ttk.Button(bar, text="断开并切回仿真", command=self.simulation).pack(side="left", padx=5)
        ttk.Button(bar, text="打开三维视图", command=self.open_view).pack(side="left")
        self.settings = ttk.LabelFrame(outer, text="CAN 连接与标定", padding=8)
        self.settings.pack(fill="x")
        self.fields = {}
        defaults = [("接口", "u2can"), ("端口", ""), ("通信速率", "921600"),
                    ("电机 ID", "1,2,3,4,5,6"), ("反馈 ID", "0x11,0x12,0x13,0x14,0x15,0x16"),
                    ("STEP 零位读数（°）", ""), ("方向（±1）", "1,1,1,1,1,1")]
        for i, (label, default) in enumerate(defaults):
            ttk.Label(self.settings, text=label).grid(row=i, column=0, sticky="w")
            opt = ttk.Combobox(self.settings, width=38) if i < 2 else ttk.Entry(self.settings, width=40)
            opt.grid(row=i, column=1, sticky="ew", pady=1)
            opt.insert(0, default)
            self.fields[label] = opt
        self.fields["接口"].configure(values=("u2can", "pcan", "socketcan"), state="readonly")
        self.fields["接口"].bind("<<ComboboxSelected>>", lambda _: self.refresh())
        ttk.Button(self.settings, text="刷新端口", command=self.refresh).grid(row=1, column=2, padx=4)
        self.refresh()
        ttk.Label(outer, textvariable=self.status, wraplength=500).pack(anchor="w", pady=6)
        self.play = ttk.Checkbutton(outer, text="播放仿真演示", variable=self.running)
        self.play.pack(anchor="w")
        self.rows = []
        for i in range(6):
            row = ttk.Frame(outer)
            row.pack(fill="x")
            ttk.Label(row, text=f"J{i+1}").pack(side="left")
            control = ttk.Scale(row, from_=-180, to=180, length=290)
            control.set(0)
            control.pack(side="left", padx=8, fill="x", expand=True)
            val = ttk.Label(row, width=10)
            val.pack(side="right")
            self.rows.append((control, val))
        ttk.Label(outer, textvariable=self.pose).pack(anchor="w", pady=6)
        self.t0 = time.monotonic()
        self.open_view()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    def refresh(self):
        channel = self.fields["端口"]
        interface = self.fields["接口"].get()
        values = [p.device for p in comports()] if interface == "u2can" else ["PCAN_USBBUS1" if interface == "pcan" else "can0"]
        channel.configure(values=values)
        if channel.get() not in values:
            channel.set("" if interface == "u2can" else values[0])
        rate = self.fields["通信速率"]
        rate.delete(0, "end")
        rate.insert(0, "921600" if interface == "u2can" else "1000000")

    def open_view(self):
        if self.view is None or not self.view.is_running():
            self.view = viewer.launch_passive(self.model, self.data)
            self.view.cam.lookat[:] = [0, 0, .16]
            self.view.cam.distance = .72
            self.view.cam.azimuth = 20
            self.view.cam.elevation = -17

    def connect(self):
        if self.worker is not None:
            return
        f = {k: v.get().strip() for k, v in self.fields.items()}
        try:
            zero = np.array([float(v) for v in f["STEP 零位读数（°）"].split(",")])
            signs = np.array([int(v) for v in f["方向（±1）"].split(",")])
            slaves, masters = ids(f["电机 ID"]), ids(f["反馈 ID"])
            rate = int(f["通信速率"])
            if (not f["端口"] or rate <= 0 or zero.size != 6 or not np.isfinite(zero).all()
                    or signs.size != 6 or not np.isin(signs, [-1, 1]).all() or set(slaves) & set(masters)):
                raise ValueError("检查端口、速率、六轴零位、方向和 ID")
        except Exception as exc:
            self.status.set(f"未连接 · 请填写实际端口与六轴标定值（英文逗号分隔）。{exc}")
            return
        self.mode = "hardware"
        self.running.set(False)
        self.status.set("连接中 · 等待六轴有效反馈")
        self.connect_btn.state(["disabled"])
        self.play.state(["disabled"])
        self.stop = Event()
        self.worker = Thread(target=stream, args=(self.stop, self.events, f["接口"], f["端口"], rate,
                             slaves, masters, np.radians(zero), signs), daemon=True)
        self.worker.start()

    def simulation(self):
        self.stop.set()
        self.mode = "sim"
        self.play.state(["!disabled"])
        self.status.set("仿真模式 · 滑块只控制模型")

    def tick(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except Empty:
                break
            if kind == "closed":
                self.worker = None
                self.connect_btn.state(["!disabled"])
            elif self.mode == "hardware":
                if kind == "error":
                    self.mode = "offline"
                    self.status.set(f"硬件离线 · 视图已冻结。{value}")
                else:
                    self.q = value
                    self.status.set("硬件跟随 · 六轴反馈正常（只读）")
        manual = self.mode == "sim" and not self.running.get()
        if manual:
            self.q = np.radians([scale.get() for scale, _ in self.rows])
        elif self.mode == "sim":
            t = time.monotonic() - self.t0
            self.q = np.radians([28,18,25,20,18,40]) * np.sin(t*np.array([.6,.42,.55,.73,.8,.95]))
        for (scale, label), angle in zip(self.rows, np.degrees(self.q)):
            scale.state(["!disabled" if manual else "disabled"])
            if not manual:
                scale.set(float(np.clip(angle, -180, 180)))
            label.configure(text=f"{angle: .1f}°")
        xyz, rpy = update_pose(self.model, self.data, self.q, 0, 1)
        self.pose.set("XYZ (mm)   " + "   ".join(f"{v:.2f}" for v in xyz) +
                      "\nRPY (°)       " + "   ".join(f"{v:.2f}" for v in rpy))
        if self.view and self.view.is_running():
            self.view.sync()
        self.root.after(40, self.tick)

    def close(self):
        self.stop.set()
        if self.worker and self.worker.is_alive():
            self.root.after(50, self.close)
            return
        if self.view:
            self.view.close()
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    Panel(window)
    window.mainloop()
