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
from arm5_force import ProbeControl, contact, estimate_force, jacobians
from arm4_twin import read_feedback, update_pose
from dm_debug import CanBus, Joint, U2Can, ids


def stream(stop, events, interface, channel, rate, slaves, masters, zero, signs):
    try:
        bus = U2Can(channel, rate) if interface == "u2can" else CanBus(interface, channel, rate)
        with closing(bus):
            joints = [Joint(s, m) for s, m in zip(slaves, masters)]
            while not stop.is_set():
                pos,vel,tau = read_feedback(bus,joints)
                events.put(("sample",(signs*(pos-zero),signs*vel,signs*tau,time.monotonic())))
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
        if ultrasound:
            forces = ttk.Frame(tabs,padding=8)
            tabs.add(forces,text="力控 / 力矩")
            self.load_text = tk.StringVar()
            self.force_on = tk.BooleanVar(value=False)
            self.force_goal = tk.DoubleVar(value=2.)
            self.force_widget = ttk.Checkbutton(forces,text="仿真法向恒力（起点就位后生效）",variable=self.force_on)
            self.force_widget.pack(anchor="w")
            ttk.Label(forces,text="仿真目标法向力 (N)").pack(anchor="w")
            self.goal_widget = ttk.Combobox(forces,textvariable=self.force_goal,values=(0,1,2,3,4,5),state="readonly",width=6)
            self.goal_widget.pack(anchor="w")
            self.zero_btn = ttk.Button(forces,text="实机空载静止时置零",command=self.zero_force)
            self.zero_btn.pack(anchor="w",pady=4)
            ttk.Label(forces,textvariable=self.load_text,justify="left").pack(anchor="w")
            self.force_control = ProbeControl()
            self.tau = None
            self.feedback_at = 0.
            self.force_bias = np.zeros(6)
            self.zeroed = False
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
            self.init_btn = ttk.Button(tools,text="初始化位置",command=self.initialize_scan)
            self.init_btn.grid(row=4,column=0,columnspan=2,sticky="ew")
            self.write_btn = ttk.Button(tools,text="自动扫描",command=self.start_selected)
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
            self.track = None
            self.auto_targets = []
            ttk.Label(tools,text="手动扫描  ← 左").grid(row=6,column=0)
            self.scrub_position = tk.DoubleVar()
            self.scrub = ttk.Scale(tools,variable=self.scrub_position,command=self.manual_scan)
            self.scrub.grid(row=6,column=1,columnspan=2,sticky="ew")
            ttk.Label(tools,text="右 →").grid(row=6,column=3)
            for var in self.coords+self.extents:
                var.trace_add("write",lambda *_:self.invalidate_scan())

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
        self.invalidate_scan()
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

    def invalidate_scan(self):
        self.cancel_plan.set()
        self.track = None
        self.auto_targets = []
        if self.writing is not None:
            self.writing = None
            for (slider,_),q in zip(self.rows,self.q):
                slider.set((np.degrees(q)+180)%360-180)
            self.status.set("设置已变更 · 请重新初始化")
        self.scrub.state(["disabled"])
        self.write_btn.state(["disabled"])

    def manual_scan(self,value):
        if self.track is not None and self.mode == "sim":
            self.automatic = False
            self.auto_targets.clear()
            self.target = float(value)
            self.status.set("手动扫描 · 拖动滑条沿橙线左右移动")

    def start_selected(self):
        if self.track is not None and self.mode == "sim":
            self.automatic = True
            self.auto_targets = [len(self.track)-1,0,self.origin_index]
            self.target = self.auto_targets.pop(0)
            self.status.set("自动扫描 · 沿可达弧段往返")

    def initialize_scan(self):
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
        initial = self.q.copy()
        cancel = self.cancel_plan = Event()
        result = SimpleQueue()
        self.status.set("初始化中 · 求解起点及相连的可达弧段")
        def plan():
            try:
                work = Workspace(self.model)
                def side(l,r):
                    poses = next(surface_runs(work,axis,position,angle,l,r,initial,connected=True),[])
                    return np.asarray(poses)
                home = side(0,0)
                if not len(home):
                    raise ValueError("所选起点不可达，请移动橙线或调整起点")
                sides = []
                for l,r in ((left,0),(0,right)):
                    if cancel.is_set():
                        return
                    poses = side(l,r)
                    sides.append(np.vstack([home,list(motion_frames(poses))]) if len(poses)>1 else home)
                result.put((np.vstack([sides[0][::-1],sides[1][1:]]),len(sides[0])-1))
            except Exception as exc:
                result.put((None,str(exc)))
        Thread(target=plan,daemon=True).start()
        def control():
            while result.empty():
                yield initial,False
            path,origin = result.get()
            if path is None:
                self.finished_message = origin
                return
            for q in motion_frames([initial,path[origin]]):
                yield q,False
            self.track = path
            self.origin_index = origin
            current = self.target = float(origin)
            self.scrub.configure(from_=0,to=max(1,len(path)-1))
            self.scrub_position.set(current)
            if len(path)>1:
                self.scrub.state(["!disabled"])
                self.write_btn.state(["!disabled"])
            self.status.set("起点已就位 · 可自动扫描或拖动手动扫描" if len(path)>1 else "起点已就位 · 当前范围仅有一个可达点")
            self.automatic = False
            speed = 0.
            while True:
                distance = self.target-current
                desired = np.sign(distance)*min(.65,np.sqrt(.03*abs(distance)))
                speed += np.clip(desired-speed,-.015,.015)
                step = speed
                if step*distance >= 0 and abs(step)>=abs(distance):
                    step,speed = distance,0.
                current = float(np.clip(current+step,0,len(path)-1))
                if self.automatic:
                    self.scrub_position.set(current)
                if current == self.target and self.automatic:
                    if self.auto_targets:
                        self.target = self.auto_targets.pop(0)
                    else:
                        self.automatic = False
                        self.status.set("自动扫描完成 · 已返回起点")
                lo = int(current)
                q = path[lo]+(current-lo)*(path[min(lo+1,len(path)-1)]-path[lo])
                yield q,abs(step)>1e-8
        self.writing = control()

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
        if self.ultrasound:
            self.invalidate_scan()
        if self.ultrasound:
            self.force_on.set(False)
            self.tau = None
            self.force_bias[:] = 0.
            self.zeroed = False
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
        if self.ultrasound:
            self.invalidate_scan()
        self.cancel_plan.set()
        self.stop.set()
        self.writing = None
        self.mode = "sim"
        self.play.state(["!disabled"])
        self.status.set("仿真模式 · 滑块只控制模型")

    def fresh_static(self):
        return (self.mode=="hardware" and self.tau is not None and
                time.monotonic()-self.feedback_at < .5 and np.max(abs(self.velocity)) < .08)

    def zero_force(self):
        if self.fresh_static():
            self.force_bias = self.data.qfrc_bias.copy()-self.tau
            self.zeroed = True

    def display_forces(self):
        sim = self.mode=="sim"
        self.force_widget.state(["!disabled" if sim else "disabled"])
        self.goal_widget.state(["!disabled" if sim else "disabled"])
        self.zero_btn.state(["!disabled" if self.fresh_static() else "disabled"])
        linear,_ = jacobians(self.model,self.data)
        force = None
        if sim:
            normal,magnitude = contact(self.data.site("acoustic").xpos)
            force = normal*magnitude
            tau = self.data.qfrc_bias-linear.T@force
            source = "仿真 · 准静态力矩 / 弹性接触模型"
        elif self.mode=="hardware" and self.tau is not None and time.monotonic()-self.feedback_at < .5:
            tau = self.tau
            source = "CAN 电机反馈 · 末端力准静态估算"
            if self.fresh_static():
                force = estimate_force(linear,self.data.qfrc_bias,tau,self.force_bias)
                source += "（已置零）" if self.zeroed else "（未置零）"
        else:
            self.load_text.set("等待新鲜的六轴反馈")
            return
        lines = [source,"关节力矩（N·m）"]
        lines.extend("     ".join(f"J{i+1}  {tau[i]:+.2f}" for i in row) for row in ((0,1,2),(3,4,5)))
        if force is None:
            lines.append("末端力不可估 · 关节运动或姿态接近奇异")
        else:
            lines += ["盒坐标 · 环境作用于探头（N）",
                      "   ".join(f"F{k} {v:+.2f}" for k,v in zip("xyz",force)),
                      f"合力 {np.linalg.norm(force):.2f} N"]
            if sim:
                lines.append(f"仿真法向接触力 {magnitude:.2f} N")
        self.load_text.set("\n".join(lines))

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
                    self.q,self.velocity,self.tau,self.feedback_at = value
                    self.status.set("硬件跟随 · 六轴反馈正常（只读）")
        if self.ultrasound:
            self.init_btn.state(["!disabled" if self.mode == "sim" else "disabled"])
        else:
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
        if self.ultrasound:
            if self.mode=="sim" and self.track is not None and self.force_on.get():
                self.q = self.force_control.correct(self.model,self.data,self.q,self.force_goal.get(),.025)
            else:
                self.force_control.offset = self.force_control.previous_force = 0.
        for (scale, label), angle in zip(self.rows, np.degrees(self.q)):
            scale.state(["!disabled" if manual else "disabled"])
            if not manual:
                scale.set(float(np.clip(angle, -180, 180)))
            label.configure(text=f"{angle: .1f}°")
        xyz, rpy = update_pose(self.model, self.data, self.q, 0, 1)
        if self.ultrasound:
            self.display_forces()
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
