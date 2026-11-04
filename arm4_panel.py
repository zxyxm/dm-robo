"""Interactive simulation and read-only CAN twin for Assembly 4."""
from contextlib import closing
from queue import Empty, SimpleQueue
from threading import Event, Thread
import time
import argparse
import sys

if sys.platform == "win32":
    from arm_window_dock import dock
import tkinter as tk
from tkinter import ttk, font

import mujoco
from mujoco import viewer
import numpy as np
from scipy.spatial.transform import Rotation
from serial.tools.list_ports import comports

from arm4_writing import pen_trajectory, draw_ink
from arm4_fk import load_model
from arm5_motion import motion_frames
from arm5_guides import guides, draw_coordinates
from arm5_scan import AXES, CENTER, ROOT, Workspace, load_passes, surface_runs, surface_start
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
        for name in ("TkDefaultFont","TkTextFont","TkMenuFont"):
            font.nametofont(name).configure(family="Microsoft YaHei UI",size=13)
        style = ttk.Style(root)
        style.configure(".",font=("Microsoft YaHei UI",13))
        root.option_add("*TCombobox*Listbox.font",("Microsoft YaHei UI",13))
        self.cancel_plan = Event()
        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text=("装配体 5   /   倒挂超声扫查" if ultrasound else "装配体 4   /   六轴操作面板"), font=("Microsoft YaHei UI", 17)).pack(anchor="w")
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=8)
        self.connect_btn = ttk.Button(bar, text="连接硬件跟随", command=self.connect)
        self.connect_btn.pack(side="left")
        ttk.Button(bar, text="断开并切回仿真", command=self.simulation).pack(side="left", padx=5)
        ttk.Button(bar, text="打开三维视图", command=self.open_view).pack(side="left")
        tabs = ttk.Notebook(outer)
        tabs.pack(fill="both",expand=True)
        controls = ttk.Frame(tabs,padding=4)
        hardware = ttk.Frame(tabs,padding=4)
        joints = ttk.Frame(tabs,padding=4)
        tabs.add(controls,text="扫查操作" if ultrasound else "写字操作")
        tabs.add(joints,text="六轴角度")
        tabs.add(hardware,text="CAN 设置")
        self.settings = ttk.LabelFrame(hardware, text="CAN 连接与标定", padding=8)
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
        self.play = ttk.Checkbutton(controls, text="播放仿真演示", variable=self.running)
        if not ultrasound:
            self.play.pack(anchor="w")
        else:
            self.slice_axis = 1
            self.position_mm = [250.,250.]
            self.spacing = tk.StringVar(value="2")
            self.slice_label = tk.StringVar()
            tools = ttk.LabelFrame(controls,text="盒坐标系 · 等距断面",padding=5)
            tools.pack(fill="x")
            for col,(name,action) in enumerate((
                    ("XZ 断面",lambda:self.select_slice(1)),("YZ 断面",lambda:self.select_slice(0)),
                    ("上一面 −",lambda:self.shift_slice(-1)),("下一面 +",lambda:self.shift_slice(1)))):
                ttk.Button(tools,text=name,command=action).grid(row=0,column=col)
            ttk.Label(tools,text="间距 (mm)").grid(row=1,column=0)
            ttk.Combobox(tools,textvariable=self.spacing,values=("1","2","5","10"),state="readonly",width=5).grid(row=1,column=1)
            self.write_btn = ttk.Button(tools,text="扫描橙色线",command=self.start_selected)
            self.write_btn.grid(row=1,column=2,columnspan=2,sticky="ew")
            ttk.Label(tools,textvariable=self.slice_label).grid(row=2,column=0,columnspan=2,sticky="w")
            self.plane_slider = ttk.Scale(tools)
            self.plane_slider.grid(row=2,column=2,columnspan=2,sticky="ew")
            self.coords = [tk.StringVar(value=v) for v in ("250","250","50")]
            self.extents = [tk.StringVar(value="180") for _ in range(2)]
            fields = ttk.Frame(tools)
            fields.grid(row=3,column=0,columnspan=4,sticky="w")
            for i,(label,var) in enumerate(zip(("X","Y","Z","左弧°","右弧°"),self.coords+self.extents)):
                ttk.Label(fields,text=label).grid(row=0,column=2*i)
                ttk.Entry(fields,textvariable=var,width=5).grid(row=0,column=2*i+1)
            ttk.Button(tools,text="贴合起点 (mm)",command=self.apply_start).grid(row=4,column=0,columnspan=2)
            self.arc_angles = [90.,90.]
            self.lateral_slider = ttk.Scale(tools,command=self.slide_start)
            self.lateral_slider.grid(row=5,column=1,columnspan=2,sticky="ew")
            ttk.Button(tools,text="← 起点",command=lambda:self.nudge_start(-1)).grid(row=5,column=0)
            ttk.Button(tools,text="起点 →",command=lambda:self.nudge_start(1)).grid(row=5,column=3)
            self.select_slice(1)
        if not ultrasound:
            self.write_btn = ttk.Button(controls,text="仿真写 8",command=self.start_task)
            self.write_btn.pack(anchor="w")
        self.rows = []
        for i in range(6):
            row = ttk.Frame(joints)
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
            # Start near the marked apex using an already screened central-slice pose.
            central = next(poses for label,poses in self.passes.items() if label.startswith("Y=250"))
            def apex_distance(q):
                self.data.qpos[:] = q
                mujoco.mj_kinematics(self.model,self.data)
                return np.linalg.norm(self.data.site("acoustic").xpos-self.start_mm/1000)
            home = min(central,key=apex_distance)
            for (slider,_),q in zip(self.rows,np.degrees(home)):
                slider.set(float((q+180)%360-180))
        self.next_dock = 0.
        self.t0 = time.monotonic()
        self.open_view()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    def select_slice(self, axis):
        self.slice_axis = axis
        label, coordinate = ("YZ", "X") if axis == 0 else ("XZ", "Y")
        self.set_arc(self.arc_angles[axis])
        self.slice_label.set(f"{label} 断面   {coordinate} = {self.position_mm[axis]:g} mm")

    def slide_plane(self,value):
        self.position_mm[self.slice_axis] = float(value)
        self.select_slice(self.slice_axis)

    def set_arc(self, angle):
        axis = self.slice_axis
        fixed = (self.position_mm[axis]/1000-CENTER[axis])/AXES[axis]
        scale = np.sqrt(max(0.,1-fixed**2))
        reach = AXES[1-axis]*scale*1000
        center = CENTER[1-axis]*1000
        point = CENTER.copy()*1000
        point[axis] = self.position_mm[axis]
        point[1-axis] += reach*np.cos(np.radians(angle))
        point[2] += AXES[2]*scale*1000*np.sin(np.radians(angle))
        self.cancel_plan.set()
        if self.writing is not None:
            self.writing = None
            for (slider,_),q in zip(self.rows,self.q):
                slider.set((np.degrees(q)+180)%360-180)
            self.status.set("位置已更新 · 点击扫描橙色线重新开始")
        self.ink.clear()
        self.plane_slider.configure(command="",from_=(CENTER[axis]-AXES[axis])*1000+.01,
                                    to=(CENTER[axis]+AXES[axis])*1000-.01)
        self.plane_slider.set(self.position_mm[axis])
        self.plane_slider.configure(command=self.slide_plane)
        self.arc_angles[axis] = angle
        self.start_mm = point
        for var,value in zip(self.coords,point):
            var.set(f"{value:.5f}")
        self.lateral_slider.configure(from_=center-reach,to=center+reach,command="")
        self.lateral_slider.set(point[1-axis])
        self.lateral_slider.configure(command=self.slide_start)

    def slide_start(self, value):
        lo,hi = (float(self.lateral_slider.cget(k)) for k in ("from","to"))
        angle = np.degrees(np.arccos(np.clip((2*float(value)-lo-hi)/(hi-lo),-1,1)))
        self.set_arc(angle)

    def nudge_start(self, sign):
        self.slide_start(self.lateral_slider.get()+sign*float(self.spacing.get()))

    def shift_slice(self, sign):
        axis = self.slice_axis
        value = self.position_mm[axis] + sign*float(self.spacing.get())
        if abs(value/1000-CENTER[axis]) >= AXES[axis]-1e-10:
            self.status.set("已到半椭球边缘，无法再移动一个完整间距")
            return
        self.position_mm[axis] = value
        self.select_slice(axis)

    def apply_start(self):
        try:
            point,angle = surface_start(self.slice_axis,[float(v.get()) for v in self.coords])
        except ValueError as exc:
            self.status.set(f"起点无效 · {exc}")
            return None
        self.start_mm = point
        self.position_mm[self.slice_axis] = point[self.slice_axis]
        self.set_arc(angle)
        self.slice_label.set(f"{('YZ','XZ')[self.slice_axis]} 断面   {'XY'[self.slice_axis]} = {point[self.slice_axis]:g} mm")
        self.status.set("绿色标记为曲面起点 · 扫描时检查可达性")
        return angle

    def selected_runs(self,axis,position,angle,left,right,initial,cancel):
        work = Workspace(self.model)
        for poses in surface_runs(work,axis,position,angle,left,right,initial):
            if cancel.is_set():
                return
            yield f"{'XY'[axis]} = {position:g} mm", poses

    def play_plan(self, make_runs):
        self.cancel_plan.set()
        cancel = self.cancel_plan = Event()
        queue = SimpleQueue()
        initial = self.q.copy()
        def solve():
            try:
                for label,poses in make_runs(initial,cancel):
                    if cancel.is_set():
                        return
                    if len(poses):
                        queue.put((label,poses))
                queue.put((None,None))
            except Exception as exc:
                queue.put((str(exc),None))
        Thread(target=solve,daemon=True).start()
        self.finished_message = None
        self.status.set("正在后台求解 · 等待可达轨迹")
        def playback():
            current = initial
            count = 0
            while True:
                try:
                    label,poses = queue.get_nowait()
                except Empty:
                    yield current,False
                    continue
                if poses is None:
                    self.finished_message = ("求解失败 · "+label if label else f"扫描结束 · 已播放 {count} 个可达段")
                    return
                self.ink.append(None)
                self.status.set("仿真转位 · "+label)
                path = np.unwrap(np.vstack([current,poses]),axis=0)
                for current in motion_frames(path[:2]):
                    yield current,False
                self.status.set("贴面扫描 · "+label)
                for current in motion_frames(path[1:]):
                    yield current,True
                count += 1
        self.writing = playback()

    def start_selected(self):
        if self.mode != "sim":
            return
        angle = self.apply_start()
        if angle is None:
            return
        try:
            left,right = [float(v.get()) for v in self.extents]
            if not (0 <= left <= 180 and 0 <= right <= 180):
                raise ValueError
        except ValueError:
            self.status.set("左右弧范围应为 0 至 180 度")
            return
        self.ink.clear()
        self.open_view()
        self.running.set(False)
        axis = self.slice_axis
        position = self.position_mm[axis]
        self.play_plan(lambda initial,cancel:self.selected_runs(axis,position,angle,left,right,initial,cancel))

    def start_task(self):
        if self.mode != "sim":
            return
        self.open_view()
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
            self.next_dock = 0.
            self.view = viewer.launch_passive(self.model, self.data)
            self.view.cam.lookat[:] = [.25,.25,.14] if self.ultrasound else [0, 0, .16]
            self.view.cam.distance = 1.12 if self.ultrasound else .72
            self.view.cam.azimuth = -82 if self.ultrasound else 20
            self.view.cam.elevation = -21

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
        self.cancel_plan.set()
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
        self.cancel_plan.set()
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
        if self.ultrasound:
            base = self.data.body("mount")
            local = base.xmat.reshape(3,3).T @ (site.xpos-base.xpos) * 1000
            self.pose.set(self.pose.get()+"\n基座 XYZ (mm)   "+"   ".join(f"{v:.2f}" for v in local))
        if ink:
            point = self.data.site("acoustic" if self.ultrasound else "pen_tip").xpos.copy()
            if not self.ink or self.ink[-1] is None or np.linalg.norm(point-self.ink[-1]) >= .0008:
                self.ink.append(point)
        if self.view and self.view.is_running():
            if sys.platform == "win32" and time.monotonic() >= self.next_dock:
                dock(self.root)
                self.next_dock = time.monotonic() + .15
            with self.view.lock():
                scene = self.view.user_scn
                draw_ink(scene, self.ink, radius=.00045 if self.ultrasound else .001)
                if self.ultrasound:
                    lines = guides(self.slice_axis, self.position_mm[self.slice_axis])
                    draw_coordinates(scene, lines, self.data.body("mount"), getattr(self,"start_mm",None))
                if self.ultrasound and ink and scene.ngeom < scene.maxgeom:
                    beam = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(beam,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),
                                       np.zeros(3),np.eye(3).ravel(),np.array([.1,.9,.65,1],np.float32))
                    tip = self.data.site("acoustic")
                    stop = tip.xpos + .025*tip.xmat.reshape(3,3)[:,2]
                    mujoco.mjv_makeConnector(beam,mujoco.mjtGeom.mjGEOM_CAPSULE,.0015,*tip.xpos,*stop)
                    scene.ngeom += 1
            self.view.sync()
        self.root.after(25 if self.ultrasound else 40, self.tick)

    def close(self):
        self.cancel_plan.set()
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
