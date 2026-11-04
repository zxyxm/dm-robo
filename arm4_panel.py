"""Interactive simulation and read-only CAN twin for Assembly 4."""
from contextlib import closing
from queue import Empty, SimpleQueue
from threading import Event, Thread
import time
import argparse
import tkinter as tk
from tkinter import ttk

import mujoco
from mujoco import viewer
import numpy as np
from scipy.spatial.transform import Rotation
from serial.tools.list_ports import comports

from arm4_writing import pen_trajectory, draw_ink
from arm4_fk import load_model
from arm5_guides import guides, draw_coordinates
from arm5_scan import AXES, CENTER, ROOT, load_passes, parallel_sections
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
    def __init__(self, root, ultrasound=True):
        self.root = root
        root.title("达妙六轴机械臂 · 孪生操作面板")
        self.ultrasound = ultrasound
        self.passes = load_passes() if ultrasound else {}
        self.model = load_model(ROOT/"arm5.xml") if ultrasound else load_model()
        self.data = mujoco.MjData(self.model)
        self.view = None
        self.worker = None
        self.stop = Event()
        self.events = SimpleQueue()
        self.mode = "sim"
        self.writing = None
        self.finished_message = None
        self.ink = []
        self.q = np.zeros(6)
        self.running = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="仿真模式 · 可拖动关节或播放演示")
        self.pose = tk.StringVar()
        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text=("装配体 5   /   倒挂超声扫查" if ultrasound else "装配体 4   /   六轴操作面板"), font=("Microsoft YaHei UI", 14)).pack(anchor="w")
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
        if not ultrasound:
            self.play.pack(anchor="w")
        else:
            self.section = ttk.Combobox(outer, values=list(self.passes), state="readonly", width=30)
            self.section.current(next(i for i,k in enumerate(self.passes) if k.startswith("Y=250")))
            self.section.pack(anchor="w")
            self.slice_axis = 1
            self.position_mm = [250.,250.]
            self.spacing = tk.StringVar(value="2")
            self.slice_label = tk.StringVar()
            self.show_guides = tk.BooleanVar(value=True)
            tools = ttk.LabelFrame(outer,text="盒坐标系 · 等距断面",padding=5)
            tools.pack(fill="x")
            for col,(name,action) in enumerate((
                    ("XZ 断面",lambda:self.select_slice(1)),("YZ 断面",lambda:self.select_slice(0)),
                    ("上一面 −",lambda:self.shift_slice(-1)),("下一面 +",lambda:self.shift_slice(1)))):
                ttk.Button(tools,text=name,command=action).grid(row=0,column=col)
            ttk.Label(tools,text="间距 (mm)").grid(row=1,column=0)
            ttk.Combobox(tools,textvariable=self.spacing,values=("1","2","5","10"),state="readonly",width=5).grid(row=1,column=1)
            self.chosen_btn = ttk.Button(tools,text="扫选中面",command=lambda:self.start_selected(False))
            self.chosen_btn.grid(row=1,column=2)
            self.grid_btn = ttk.Button(tools,text="扫等距多面",command=lambda:self.start_selected(True))
            self.grid_btn.grid(row=1,column=3)
            ttk.Label(tools,textvariable=self.slice_label).grid(row=2,column=0,columnspan=3,sticky="w")
            ttk.Checkbutton(tools,text="断面网格",variable=self.show_guides).grid(row=2,column=3)
            self.select_slice(1)
            self.multi_btn = ttk.Button(outer, text="按当前探头角度 · 扫多个切面", command=self.start_parallel)
            self.multi_btn.pack(anchor="w")
        self.write_btn = ttk.Button(outer, text="仿真扫描所选截面区间" if ultrasound else "仿真写 8", command=self.start_task)
        self.write_btn.pack(anchor="w")
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
        if ultrasound:
            for (slider,_),q in zip(self.rows,np.degrees(self.passes[self.section.get()][0])):
                slider.set(float((q+180)%360-180))
        self.t0 = time.monotonic()
        self.open_view()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    def select_slice(self, axis):
        self.slice_axis = axis
        label, coordinate = ("YZ", "X") if axis == 0 else ("XZ", "Y")
        self.slice_label.set(f"{label} 断面   {coordinate} = {self.position_mm[axis]:g} mm")

    def shift_slice(self, sign):
        axis = self.slice_axis
        value = self.position_mm[axis] + sign*float(self.spacing.get())
        if abs(value/1000-CENTER[axis]) >= AXES[axis]-1e-10:
            self.status.set("已到半椭球边缘，无法再移动一个完整间距")
            return
        self.position_mm[axis] = value
        self.select_slice(axis)

    def start_selected(self, multiple):
        if self.mode != "sim":
            return
        offsets = (0,2,4,6,8,-2,-4,-6,-8) if multiple else (0,)
        offsets = tuple(v*float(self.spacing.get())/2 for v in offsets)
        self.ink.clear()
        self.open_view()
        self.running.set(False)
        self.writing = self.parallel_frames(self.q.copy(), (self.slice_axis,self.position_mm[self.slice_axis]), offsets)

    def start_parallel(self):
        if self.mode != "sim":
            return
        self.open_view()
        self.ink.clear()
        self.running.set(False)
        self.writing = self.parallel_frames(self.q.copy())

    def parallel_frames(self, captured, plane=None, offsets=(0,2,4,6,8,-2,-4,-6,-8)):
        completed = 0
        for offset, poses in parallel_sections(self.model, captured, plane=plane, offsets=offsets):
            if not poses:
                continue
            completed += 1
            self.ink.append(None)
            location = f"相对当前截面 {offset:+g} mm" if plane is None else f"{'XY'[plane[0]]} = {plane[1]+offset:g} mm"
            self.status.set(f"仿真分段扫查 · 第 {completed} 面 · {location}")
            for q in poses:
                yield q, True
        self.finished_message = f"多切面扫查结束 · 已扫 {completed}/{len(offsets)} 面"

    def start_task(self):
        if self.mode != "sim":
            return
        self.open_view()
        if self.ultrasound:
            self.running.set(False)
            self.ink.clear()
            self.writing = ((q,True) for q in self.passes[self.section.get()])
            self.status.set("仿真定位到本段起点并扫查 · " + self.section.get())
            return
        self.view.cam.azimuth = -130
        self.view.cam.elevation = -10
        self.view.cam.lookat[:] = [0, -.13, .20]
        self.view.cam.distance = .57
        self.running.set(False)
        self.play.state(["disabled"])
        self.ink.clear()
        self.writing = pen_trajectory(self.model)
        self.status.set("仿真写字 · 从 STEP 零位起笔")

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
            self.view.cam.lookat[:] = [.25,.25,.14] if self.ultrasound else [0, 0, .16]
            self.view.cam.distance = 1.12 if self.ultrasound else .72
            self.view.cam.azimuth = 120 if self.ultrasound else 20
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
        self.writing = None
        self.ink.clear()
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
        self.writing = None
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
        self.write_btn.state(["!disabled" if self.mode == "sim" else "disabled"])
        if self.ultrasound:
            for button in (self.multi_btn,self.chosen_btn,self.grid_btn):
                button.state(["!disabled" if self.mode == "sim" else "disabled"])
        ink = False
        manual = self.mode == "sim" and not self.running.get() and self.writing is None
        if manual:
            self.q = np.radians([scale.get() for scale, _ in self.rows])
        elif self.writing is not None:
            try:
                self.q, ink = next(self.writing)
            except StopIteration:
                self.writing = None
                self.play.state(["!disabled"])
                for (scale, _), angle in zip(self.rows, np.degrees(self.q)):
                    scale.state(["!disabled"])
                    scale.set(float((angle+180)%360-180))
                self.status.set(self.finished_message or ("本段扫查结束 · 可选择其他截面区间" if self.ultrasound else "数字 8 已写完 · 笔尖已抬起"))
                self.finished_message = None
        elif self.mode == "sim":
            t = time.monotonic() - self.t0
            self.q = np.radians([28,18,25,20,18,40]) * np.sin(t*np.array([.6,.42,.55,.73,.8,.95]))
        for (scale, label), angle in zip(self.rows, np.degrees(self.q)):
            scale.state(["!disabled" if manual else "disabled"])
            if not manual:
                scale.set(float(np.clip(angle, -180, 180)))
            label.configure(text=f"{angle: .1f}°")
        xyz, rpy = update_pose(self.model, self.data, self.q, 0, 1)
        if self.ultrasound:
            site = self.data.site("acoustic")
            xyz = site.xpos * 1000
            rpy = Rotation.from_matrix(site.xmat.reshape(3,3)).as_euler("xyz",degrees=True)
        self.pose.set(("探头中心 XYZ (mm)   " if self.ultrasound else "XYZ (mm)   ") + "   ".join(f"{v:.2f}" for v in xyz) +
                      "\nRPY (°)       " + "   ".join(f"{v:.2f}" for v in rpy))
        if ink:
            self.ink.append(self.data.site("acoustic" if self.ultrasound else "pen_tip").xpos.copy())
        if self.view and self.view.is_running():
            with self.view.lock():
                scene = self.view.user_scn
                draw_ink(scene, self.ink, radius=.00045 if self.ultrasound else .001)
                if self.ultrasound:
                    lines = guides(self.slice_axis, self.position_mm[self.slice_axis],float(self.spacing.get())) if self.show_guides.get() else ()
                    draw_coordinates(scene, lines)
                if self.ultrasound and ink and scene.ngeom < scene.maxgeom:
                    beam = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(beam,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),
                                       np.zeros(3),np.eye(3).ravel(),np.array([.1,.9,.65,1],np.float32))
                    tip = self.data.site("acoustic")
                    stop = tip.xpos + .025*tip.xmat.reshape(3,3)[:,2]
                    mujoco.mjv_makeConnector(beam,mujoco.mjtGeom.mjGEOM_CAPSULE,.0015,*tip.xpos,*stop)
                    scene.ngeom += 1
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--writing", action="store_true", help="Open the previous Assembly 4 writing scene")
    args = parser.parse_args()
    window = tk.Tk()
    Panel(window, ultrasound=not args.writing)
    window.mainloop()
