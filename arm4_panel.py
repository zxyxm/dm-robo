"""Interactive simulation and read-only CAN twin for Assembly 4."""
from contextlib import closing
from copy import copy
from queue import Empty, SimpleQueue
from threading import Event, Thread
import time
import sqlite3
from pathlib import Path
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
from arm5_motion import SPEED, motion_frames, ceiling_park_frames
from arm5_coverage import optimize_mount, BLUE, draw_area, sample_section, solve_area
from arm5_guides import box_mm, guides, draw_coordinates
from arm5_scan import AXES, CENTER, ROOT, Workspace, surface_runs, surface_start
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
        root.title("声弧 · 机械臂控制面板")
        logo = tk.PhotoImage(file=str(ROOT/"shenghu.png"))
        self.logo = logo.subsample(4)
        root.iconphoto(True,self.logo,logo)
        self.ultrasound = ultrasound
        self.radii = AXES.copy()
        self.center = CENTER.copy()
        self.mount_stop = Event()
        self.model = load_model(ROOT/"arm5.xml") if ultrasound else load_model()
        self.data = mujoco.MjData(self.model)
        self.view = None
        self.closing = False
        self.worker = None
        self.stop = Event()
        self.events = SimpleQueue()
        self.mode = "sim"
        self.writing = None
        self.parking = False
        self.finished_message = None
        self.ink = []
        self.q = np.zeros(6)
        self.velocity = np.zeros(6)
        self.speed_faults = {}
        self.running = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="仿真模式 · 可拖动关节或播放演示")
        self.pose = tk.StringVar()
        for name in ("TkDefaultFont","TkTextFont","TkMenuFont"):
            font.nametofont(name).configure(family="Microsoft YaHei UI",size=17)
        style = ttk.Style(root)
        style.configure(".",font=("Microsoft YaHei UI",17))
        root.option_add("*TCombobox*Listbox.font",("Microsoft YaHei UI",17))
        root.option_add("*TEntry.font","TkTextFont")
        root.option_add("*TCombobox.font","TkTextFont")
        self.cancel_plan = Event()
        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer,image=self.logo,compound="left",text=("  声弧 · 机械臂超声扫查" if ultrasound else "  声弧 · 六轴操作面板"),font=("Microsoft YaHei UI",21)).pack(anchor="w")
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=8)
        self.connect_btn = ttk.Button(bar, text="连接硬件跟随", command=self.connect)
        self.connect_btn.pack(side="left")
        ttk.Button(bar, text="断开并切回仿真", command=self.simulation).pack(side="left", padx=5)
        ttk.Button(bar, text="打开三维视图", command=self.open_view).pack(side="left")
        if ultrasound:
            self.park_btn = ttk.Button(bar,text="一键收臂",command=self.park_arm)
            self.park_btn.pack(side="left",padx=5)
        tabs = ttk.Notebook(outer)
        tabs.pack(fill="both",expand=True)
        controls = ttk.Frame(tabs,padding=4)
        hardware = ttk.Frame(tabs,padding=4)
        joints = ttk.Frame(tabs,padding=4)
        tabs.add(controls,text="扫查操作" if ultrasound else "写字操作")
        tabs.add(joints,text="六轴角度")
        log_tab = ttk.Frame(tabs,padding=4)
        tabs.add(log_tab,text="速度日志")
        self.speed_log = tk.Text(log_tab,height=10,width=48,wrap="word",state="disabled")
        self.speed_log.pack(fill="both",expand=True)
        self.speed_text = tk.StringVar()
        ttk.Label(outer,textvariable=self.speed_text).pack(anchor="w")
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
            self.slice_axis = 0
            self.position_mm = [250.,250.]
            self.spacing = tk.StringVar(value="2")
            self.slice_label = tk.StringVar()
            tools = ttk.LabelFrame(controls,text="盒坐标系 · 等距断面",padding=5)
            tools.pack(fill="x")
            for col in range(4):
                tools.columnconfigure(col,weight=1)
            for col,(name,action) in enumerate((
                    ("XZ 断面",lambda:self.select_slice(1)),("YZ 断面",lambda:self.select_slice(0)),
                    ("上一面 −",lambda:self.shift_slice(-1)),("下一面 +",lambda:self.shift_slice(1)))):
                ttk.Button(tools,text=name,command=action).grid(row=0,column=col)
            ttk.Label(tools,text="间距 (mm)").grid(row=1,column=0)
            ttk.Combobox(tools,textvariable=self.spacing,values=("1","2","5","10"),state="readonly",width=5).grid(row=1,column=1)
            self.init_btn = ttk.Button(tools,text="初始化到负向极限",command=self.initialize_scan)
            self.init_btn.grid(row=6,column=3,sticky="ew",padx=5,pady=5)
            self.write_btn = ttk.Button(tools,text="自动扫描",command=self.start_selected)
            self.write_btn.grid(row=8,column=1,columnspan=3,sticky="ew",padx=5,pady=3)
            ttk.Label(tools,text="自动模式").grid(row=8,column=0,sticky="w")
            ttk.Label(tools,textvariable=self.slice_label).grid(row=2,column=0,columnspan=2,sticky="w",padx=7,pady=8)
            self.plane_slider = ttk.Scale(tools)
            self.plane_slider.grid(row=2,column=2,columnspan=2,sticky="ew",padx=10)
            self.start_label = tk.StringVar()
            ttk.Label(tools,textvariable=self.start_label).grid(row=6,column=0,columnspan=3,sticky="w",padx=7)
            self.track = None
            self.auto_targets = []
            ttk.Label(tools,text="调整起点位置").grid(row=5,column=0,sticky="w",padx=7)
            manual_row = ttk.Frame(tools)
            manual_row.grid(row=7,column=0,columnspan=4,sticky="ew",pady=3)
            manual_row.columnconfigure(2,weight=1)
            ttk.Label(manual_row,text="手动模式").grid(row=0,column=0,sticky="w",padx=(0,12))
            self.scrub_position = tk.DoubleVar()
            self.scrub = ttk.Scale(manual_row,variable=self.scrub_position,command=self.manual_scan)
            self.scrub.grid(row=0,column=2,sticky="ew",padx=8)
            self.scan_steps = []
            for col,label,sign in ((1,"−",-1),(3,"+",1)):
                button = ttk.Button(manual_row,text=label,width=3,
                                    command=lambda d=sign:self.step_scan(d))
                button.grid(row=0,column=col)
                self.scan_steps.append(button)

            self.arc_angles = [90.,90.]
            self.lateral_slider = ttk.Scale(tools,command=self.slide_start)
            self.lateral_slider.grid(row=5,column=2,sticky="ew")
            ttk.Button(tools,text="← 起点",command=lambda:self.nudge_start(-1)).grid(row=5,column=1)
            ttk.Button(tools,text="起点 →",command=lambda:self.nudge_start(1)).grid(row=5,column=3)
            self.coverage_text = tk.StringVar()
            ttk.Label(tools,textvariable=self.coverage_text).grid(row=9,column=0,columnspan=4,sticky="w")
            self.add_size_controls(controls)
            progress_box = ttk.LabelFrame(controls,text="蓝色可扫区 · 求解日志",padding=5)
            progress_box.pack(fill="x",pady=4)
            self.area_message = tk.StringVar()
            ttk.Label(progress_box,textvariable=self.area_message,foreground="#1656a6").pack(anchor="w")
            self.area_bar = ttk.Progressbar(progress_box,maximum=100)
            self.area_bar.pack(fill="x")
            self.area_log = tk.Text(progress_box,height=3,width=42,state="disabled",font="TkTextFont")
            self.area_log.pack(fill="x")
            self.solve_requested = False
            self.area_message.set("◆ 等待设置尺寸并求解")
            self.area_cache = {}
            self.area_stop = Event()
            self.slice_stop = Event()
            self.coverage_results = SimpleQueue()
            self.preview_job = None
            self.preview_key = None
            self.area_axis = None
            self.blue_lines = []
            self.select_slice(self.slice_axis)
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
            self.preview_work = Workspace(self.model)
            self.follow_slice()
            self.q = np.radians([slider.get() for slider,_ in self.rows])
        update_pose(self.model,self.data,self.q,0,1)
        self.last_motion = time.monotonic()
        self.next_dock = 0.
        self.t0 = time.monotonic()
        self.open_view()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    def select_slice(self, axis):
        if axis != 0:
            self.status.set("工具 TZ 锁定世界 +X，扫查使用 YZ 断面")
            return
        self.slice_axis = axis
        label, coordinate = ("YZ", "X") if axis == 0 else ("XZ", "Y")
        self.set_arc(self.arc_angles[axis])
        self.slice_label.set(f"{label} 断面   {coordinate} = {self.position_mm[axis]:g} mm")
        if hasattr(self, "preview_work") and self.mode == "sim":
            self.follow_slice()

    def follow_slice(self):
        axis = self.slice_axis
        poses = next(surface_runs(self.preview_work, axis, self.position_mm[axis],
                                  self.arc_angles[axis], 0, 0, self.q), [])
        if not poses:
            for (slider,_),angle in zip(self.rows,np.degrees(self.q)):
                slider.set(float((angle+180)%360-180))
            self.status.set("当前断面的标记点不可达 · 探头保持上次姿态")
            return
        target = (poses[0]+np.pi)%(2*np.pi)-np.pi
        self.preview_target = target.copy()
        self.running.set(False)
        for (slider,_),angle in zip(self.rows,np.degrees(target)):
            slider.set(angle)
        self.status.set("探头限速跟随当前断面")

    def slide_plane(self,value):
        self.position_mm[self.slice_axis] = float(value)
        self.select_slice(self.slice_axis)

    def set_arc(self, angle):
        axis = self.slice_axis
        fixed = (self.position_mm[axis]/1000-CENTER[axis])/self.radii[axis]
        scale = np.sqrt(max(0.,1-fixed**2))
        reach = self.radii[1-axis]*scale*1000
        center = CENTER[1-axis]*1000
        point = self.center.copy()*1000
        point[axis] = self.position_mm[axis]
        point[1-axis] += reach*np.cos(np.radians(angle))
        point[2] += self.radii[2]*scale*1000*np.sin(np.radians(angle))
        self.invalidate_scan()
        self.ink.clear()
        self.plane_slider.configure(command="",from_=(CENTER[axis]-self.radii[axis])*1000+.01,
                                    to=(CENTER[axis]+self.radii[axis])*1000-.01)
        self.plane_slider.set(self.position_mm[axis])
        self.plane_slider.configure(command=self.slide_plane)
        self.arc_angles[axis] = angle
        self.start_mm = point
        self.start_label.set("起点（mm）  " + "   ".join(f"{axis} {value:.2f}" for axis,value in zip("XYZ",box_mm(point/1000))))
        self.lateral_slider.configure(from_=center-reach,to=center+reach,command="")
        self.lateral_slider.set(point[1-axis])
        self.lateral_slider.configure(command=self.slide_start)
        self.refresh_coverage()

    def add_size_controls(self,parent):
        box = ttk.LabelFrame(parent,text="椭球尺寸（mm）",padding=6)
        box.pack(fill="x",pady=4)
        self.geometry_text = tk.StringVar(value="盒顶 Z = 0 mm · 椭球底面居中 (250, -250, -300) mm")
        ttk.Label(box,textvariable=self.geometry_text).grid(row=0,columnspan=3,sticky="w")
        fields = []
        for i,(label,value) in enumerate(zip(("X 全长","Y 全宽","Z 高度"),self.radii*[2000,2000,1000])):
            box.columnconfigure(i,weight=1,uniform="dimension")
            ttk.Label(box,text=label).grid(row=1,column=i,sticky="w",padx=5)
            field = tk.StringVar(value=f"{value:g}")
            ttk.Entry(box,textvariable=field,width=10).grid(row=2,column=i,sticky="ew",padx=5)
            fields.append(field)
        self.height_field = tk.StringVar(value="300")
        ttk.Label(box,text="盒子高度（mm）").grid(row=3,column=0,sticky="w",padx=5)
        ttk.Entry(box,textvariable=self.height_field,width=10).grid(row=3,column=1,sticky="ew",padx=5)
        self.mount_xy = tuple(self.model.body("mount").pos[:2])
        self.recipe_mount = self.mount_xy
        error = tk.StringVar()
        def parameters():
            sizes = [float(field.get()) for field in fields]
            height = float(self.height_field.get())
            self.validate_dimensions(sizes,height)
            return sizes,height
        def submit(solve=True):
            try:
                sizes,height = parameters()
                self.solve_dimensions(sizes,height,self.recipe_mount,solve=solve)
                error.set("")
            except (ValueError,RuntimeError) as exc:
                error.set(str(exc))
        ttk.Button(box,text="确定椭球和盒子参数",command=lambda:submit(False)).grid(row=4,column=0,sticky="ew",padx=5,pady=3)
        ttk.Button(box,text="应用并求解",command=submit).grid(row=4,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(box,text="优化基座安装位置",command=self.start_mount_search).grid(row=5,column=0,sticky="ew",padx=5)
        self.mount_message = tk.StringVar(value="按已应用参数比较顶面安装点")
        ttk.Label(box,textvariable=self.mount_message).grid(row=5,column=1,columnspan=2,sticky="w")
        db_path = Path.home()/".shenghu"/"recipes.sqlite3"
        name = tk.StringVar()
        picker = ttk.Combobox(box,textvariable=name,width=12)
        picker.grid(row=6,column=0,sticky="ew",padx=5,pady=4)
        def recipes(action):
            try:
                db_path.parent.mkdir(exist_ok=True)
                with sqlite3.connect(db_path) as db:
                    db.execute("CREATE TABLE IF NOT EXISTS recipe(name TEXT PRIMARY KEY,x REAL,y REAL,z REAL,h REAL,bx REAL,by REAL)")
                    key = name.get().strip()
                    if action=="save":
                        if not key:
                            raise ValueError("请输入配方名称")
                        sizes,height = parameters()
                        db.execute("INSERT OR REPLACE INTO recipe VALUES(?,?,?,?,?,?,?)",(key,*sizes,height,*self.recipe_mount))
                    elif action=="load":
                        row = db.execute("SELECT x,y,z,h,bx,by FROM recipe WHERE name=?",(key,)).fetchone()
                        if row is None:
                            raise ValueError("请选择已有配方")
                        for field,v in zip([*fields,self.height_field],row[:4]):
                            field.set(f"{v:g}")
                        self.recipe_mount = tuple(row[4:])
                    picker.configure(values=[r[0] for r in db.execute("SELECT name FROM recipe ORDER BY name")])
                if action!="list":
                    error.set("配方已保存" if action=="save" else "配方已载入，点击应用并求解")
            except (ValueError,OSError,sqlite3.Error) as exc:
                error.set(str(exc))
        recipes("list")
        ttk.Button(box,text="保存配方",command=lambda:recipes("save")).grid(row=6,column=1,sticky="ew",padx=5)
        ttk.Button(box,text="载入配方",command=lambda:recipes("load")).grid(row=6,column=2,sticky="ew",padx=5)
        ttk.Label(box,textvariable=error,foreground="#8b4420",wraplength=240).grid(row=4,column=2,sticky="w")

    @staticmethod
    def validate_dimensions(sizes,height):
        sizes = np.asarray(sizes,dtype=float)
        if not np.isfinite(height) or height<=0:
            raise ValueError("盒高必须是正数")
        if not np.isfinite(sizes).all() or np.any(sizes <= [1,1,.5]) or np.any(sizes > [500,500,height]):
            raise ValueError("椭球长宽须在 (1,500] mm，高度须大于 0.5 mm 且不超过盒高")

    def start_mount_search(self):
        if self.mode!="sim":
            self.mount_message.set("请切回仿真模式再优化")
            return
        self.mount_stop.set()
        stop = self.mount_stop = Event()
        model,results = self.model,self.coverage_results
        self.mount_message.set("正在比较安装点 · TZ 固定朝 +X")
        def search():
            def progress(i,total,xy,area):
                results.put(("mount_progress",stop,f"{i}/{total} 个安装点 · 最佳面积 {area:.0f} mm²"))
            try:
                result = optimize_mount(model,stop,progress)
                if result is not None:
                    results.put(("mount_done",stop,result))
            except Exception as exc:
                results.put(("mount_progress",stop,"优化失败 · "+str(exc)))
        Thread(target=search,daemon=True).start()

    def solve_dimensions(self,sizes,height=None,mount_xy=None,*,solve=True):
        if self.mode != "sim":
            raise ValueError("请切回仿真模式后设置尺寸")
        height = 300-self.center[2]*1000 if height is None else height
        mount_xy = self.mount_xy if mount_xy is None else tuple(mount_xy)
        self.validate_dimensions(sizes,height)
        sizes = np.asarray(sizes,dtype=float)
        floor = .3-height/1000
        radii = sizes/[2000,2000,1000]
        changed = not np.array_equal(self.radii,radii) or abs(floor-self.center[2])>1e-10 or mount_xy!=self.mount_xy
        if not changed and not solve:
            self.status.set("椭球和盒子参数已确定")
            return
        if changed:
            model = load_model(ROOT/"arm5.xml",phantom_scale=radii/AXES,box_height=height/1000,mount_xy=mount_xy)
            data = mujoco.MjData(model)
            work = Workspace(model)
        self.mount_stop.set()
        self.mount_xy = self.recipe_mount = mount_xy
        self.geometry_text.set(f"盒顶 Z = 0 mm · 椭球底面居中 (250, -250, {-height:g}) mm")
        self.mount_message.set(f"安装中心 ({mount_xy[0]*1000:g}, {box_mm([*mount_xy,.3])[1]:g}, 0) mm")
        self.area_stop.set()
        self.slice_stop.set()
        self.invalidate_scan()
        self.coverage_results = SimpleQueue()
        self.area_cache.clear()
        self.preview_key = self.area_axis = None
        if changed:
            self.model,self.data,self.preview_work,self.radii = model,data,work,radii
            self.center = work.center.copy()
            update_pose(model,data,self.q,0,1)
            self.position_mm = list(CENTER[:2]*1000)
            self.arc_angles = [90.,90.]
            if self.view and self.view.is_running():
                self.refresh_view_geometry()
        self.solve_requested = solve
        if not solve:
            self.area_message.set("◆ 参数已确定 · 点击“应用并求解”生成蓝色面")
            self.area_bar.configure(value=0)
        self.log_area("尺寸（mm） " + " × ".join(f"{v:g}" for v in sizes))
        self.select_slice(self.slice_axis)

    def refresh_coverage(self):
        axis = self.slice_axis
        position = self.position_mm[axis]
        model, results = self.model, self.coverage_results
        key = (axis,position)
        if key == self.preview_key:
            return
        self.preview_key = key
        if self.solve_requested and axis != self.area_axis:
            self.area_stop.set()
            self.area_stop = Event()
            self.area_axis = axis
            if axis not in self.area_cache:
                self.area_started = time.monotonic()
                self.area_message.set("◆ 正在生成蓝色面 · 正在估算剩余时间")
                self.area_bar.configure(value=0)
                self.log_area("开始逐层检查可达姿态")
                def area(stop):
                    try:
                        def progress(done,total):
                            results.put(("progress",axis,(done,total,time.monotonic())))
                        patches = solve_area(model,axis,stop,progress)
                        if not stop.is_set():
                            results.put(("area",axis,patches))
                    except Exception as exc:
                        if not stop.is_set():
                            results.put(("area_error",axis,str(exc)))
                Thread(target=area,args=(self.area_stop,),daemon=True).start()
        self.slice_stop.set()
        stop = self.slice_stop = Event()
        self.blue_lines = []
        self.coverage_text.set("正在求解当前断面 · 蓝色表示可扫")
        if self.preview_job is not None:
            self.root.after_cancel(self.preview_job)
        def section():
            try:
                row = sample_section(Workspace(model),axis,position,2,stop)
                if not stop.is_set():
                    lines = [(a,b,.0008,BLUE) for (a,ok),(b,other) in zip(row,row[1:]) if ok and other]
                    results.put(("section",key,(lines,sum(ok for _,ok in row),len(row))))
            except Exception as exc:
                results.put(("error",key,str(exc)))
        def start():
            self.preview_job = None
            Thread(target=section,daemon=True).start()
        self.preview_job = self.root.after(220,start)

    def log_area(self,message):
        self.area_log.configure(state="normal")
        self.area_log.insert("end",time.strftime("%H:%M:%S  ")+message+"\n")
        self.area_log.see("end")
        self.area_log.configure(state="disabled")

    def poll_coverage(self):
        while not self.coverage_results.empty():
            kind,key,result = self.coverage_results.get_nowait()
            if kind in ("mount_progress","mount_done"):
                if key.is_set() or self.mode!="sim":
                    continue
                if kind=="mount_progress":
                    self.mount_message.set(result)
                else:
                    xy,area = result
                    if xy!=self.mount_xy or not self.solve_requested:
                        self.solve_dimensions(self.radii*[2000,2000,1000],mount_xy=xy)
                    if area>0:
                        self.mount_message.set(f"网格最佳 ({xy[0]*1000:g}, {box_mm([*xy,.3])[1]:g}, 0) mm · {area:.1f} mm²")
                    else:
                        self.mount_message.set("所有候选安装点均未形成蓝色面片，保留当前基座位置")
                    self.log_area(self.mount_message.get())
                continue
            if kind in ("progress","area","area_error") and key!=self.area_axis:
                continue
            if kind=="progress":
                done,total,stamp = result
                eta = (stamp-self.area_started)*(total-done)/done
                self.area_bar.configure(value=100*done/total)
                self.area_message.set(f"◆ 已求解 {done}/{total} 层 · 预计还需 {eta:.1f} 秒")
                if done==1 or done%5==0:
                    self.log_area(f"已完成 {done}/{total} 层，预计剩余 {eta:.1f} 秒")
            elif kind=="area_error":
                self.area_message.set("蓝色面生成失败")
                self.log_area("ERROR "+result)
            elif kind=="area":
                self.area_cache[key] = result
                self.area_message.set(f"◆ 蓝色面已生成 · {len(result)} 个三角面片")
                self.log_area(f"生成完成，耗时 {time.monotonic()-self.area_started:.1f} 秒")
            elif key==self.preview_key:
                if kind=="error":
                    self.coverage_text.set("可扫区求解失败 · "+result)
                else:
                    self.blue_lines,count,total = result
                    self.coverage_text.set(f"蓝色为可扫区 · 当前断面 {count}/{total} 个采样点可达")

    def slide_start(self, value):
        lo,hi = (float(self.lateral_slider.cget(k)) for k in ("from","to"))
        angle = np.degrees(np.arccos(np.clip((2*float(value)-lo-hi)/(hi-lo),-1,1)))
        self.set_arc(angle)
        if self.mode == "sim":
            self.follow_slice()

    def nudge_start(self, sign):
        self.slide_start(self.lateral_slider.get()+sign*float(self.spacing.get()))

    def shift_slice(self, sign):
        axis = self.slice_axis
        value = self.position_mm[axis] + sign*float(self.spacing.get())
        if abs(value/1000-CENTER[axis]) >= self.radii[axis]-1e-10:
            self.status.set("已到半椭球边缘，无法再移动一个完整间距")
            return
        self.position_mm[axis] = value
        self.select_slice(axis)

    def invalidate_scan(self):
        self.parking = False
        self.finished_message = None
        self.cancel_plan.set()
        self.track = None
        self.pending_scan = None
        self.auto_targets = []
        if self.writing is not None:
            self.writing = None
            for (slider,_),q in zip(self.rows,self.q):
                slider.set((np.degrees(q)+180)%360-180)
            self.status.set("设置已变更 · 可直接扫描")
        self.scrub.configure(from_=0,to=100)
        self.scrub_position.set(50)

    def park_arm(self):
        if self.mode!="sim" or self.parking:
            return
        self.invalidate_scan()
        self.running.set(False)
        self.parking = True
        self.ink.clear()
        self.status.set("收臂中 · 正在检查通往盒顶 +Y 侧的路径")
        initial,model = self.q.copy(),self.model
        stop = self.cancel_plan = Event()
        result = SimpleQueue()
        def plan():
            try:
                frames = ceiling_park_frames(Workspace(model),initial,stop)
                result.put((frames,"未找到可通过碰撞检查的收臂路径，保持当前姿态"))
            except Exception as exc:
                result.put(([],f"收臂规划失败 · {exc}"))
        Thread(target=plan,daemon=True).start()
        def playback():
            while result.empty():
                yield initial,False
            frames,message = result.get()
            if frames:
                self.status.set("正在限速收臂至盒顶 +Y 侧")
                for q in frames:
                    yield q,False
                    while np.max(abs((self.q-q+np.pi)%(2*np.pi)-np.pi))>1e-7:
                        yield q,False
                message = "收臂完成 · 已停放在盒顶 +Y 侧"
            self.finished_message = message
            self.parking = False
        self.writing = playback()

    def step_scan(self,direction):
        if self.mode == "sim" and self.track is None:
            prior = self.pending_scan
            if prior and prior[0]=="manual":
                self.scrub.set(float(np.clip(self.scrub.get()+direction,0,100)))
            else:
                count = prior[1] if prior and prior[0]=="step" else 0
                self.request_scan("step",count+direction)
            return
        if self.track is not None and self.mode == "sim":
            last = len(self.track)-1
            step = max(1,round(last/100))
            self.scrub.set(float(np.clip(self.scrub.get()+direction*step,0,last)))

    def manual_scan(self,value):
        if self.mode == "sim" and self.track is None:
            self.request_scan("manual",float(value)/100)
            self.scrub_position.set(float(value))
            return
        if self.track is not None and self.mode == "sim":
            self.automatic = False
            self.auto_targets.clear()
            self.target = float(value)
            self.status.set("手动扫描 · 拖动滑条沿橙线左右移动")

    def start_selected(self):
        if self.mode == "sim" and self.track is None:
            self.request_scan("auto",0)
            return
        if self.track is not None and self.mode == "sim":
            self.automatic = True
            self.auto_targets = [len(self.track)-1,0]
            self.target = self.auto_targets.pop(0)
            self.status.set("自动扫描 · 沿可达弧段往返")

    def request_scan(self,mode,value):
        if self.pending_scan is None:
            self.initialize_scan((mode,value))
        else:
            self.pending_scan = (mode,value)

    def initialize_scan(self,request=("limit",0)):
        if self.mode != "sim":
            return
        self.invalidate_scan()
        self.pending_scan = request
        angle = self.arc_angles[self.slice_axis]
        self.ink.clear()
        self.open_view()
        self.running.set(False)
        axis = self.slice_axis
        position = self.position_mm[axis]
        initial = self.q.copy()
        cancel = self.cancel_plan = Event()
        result = SimpleQueue()
        self.status.set("正在求解当前起点相连的可扫弧段")
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
                for l,r in ((180,0),(0,180)):
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
                self.pending_scan = None
                self.finished_message = origin
                return
            for q in motion_frames([initial,path[origin]]):
                yield q,False
            index = origin
            if self.pending_scan[0]=="limit":
                for index in range(origin-1,-1,-1):
                    if self.pending_scan[0]!="limit":
                        index += 1
                        break
                    yield path[index],False
            while np.max(abs((self.q-path[index]+np.pi)%(2*np.pi)-np.pi)) > 1e-4:
                yield path[index],False
            self.track = path
            current = self.target = float(index)
            self.scrub.configure(from_=0,to=max(1,len(path)-1))
            self.scrub_position.set(current)
            self.automatic = False
            mode,value = self.pending_scan
            self.pending_scan = None
            if len(path)==1:
                self.status.set("当前起点仅有一个可达点，请调整起点或断面")
            elif mode=="auto":
                self.start_selected()
            elif mode in ("manual","step"):
                last = len(path)-1
                target = value*last if mode=="manual" else current+value*max(1,round(last/100))
                self.scrub.set(float(np.clip(target,0,last)))
            else:
                _,limit_angle = surface_start(axis,self.data.site("acoustic").xpos*1000,self.radii,self.center)
                self.status.set(f"已到负向极限 {min(0.,angle-limit_angle):.1f}° · 可直接扫描")
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
                        self.status.set("自动扫描完成 · 已返回负向极限")
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

    def refresh_view_geometry(self):
        # All parameter recipes use the same meshes and topology.
        source,display = self.model,self.display_model
        with self.view.lock():
            for name in ("body_pos","site_pos","geom_pos","geom_quat","geom_size","geom_rbound",
                         "mesh_vert","mesh_normal","mesh_face","mesh_facenormal"):
                getattr(display,name)[:] = getattr(source,name)
            self.display_data.qpos[:] = self.q
            mujoco.mj_forward(display,self.display_data)
            self.view.user_scn.ngeom = 0
        self.view.update_mesh(display.mesh("phantom").id)
        self.view.sync()

    def open_view(self):
        if self.view is None or not self.view.is_running():
            self.next_dock = 0.
            if self.ultrasound:
                self.display_model = copy(self.model)
                self.display_data = mujoco.MjData(self.display_model)
                self.display_data.qpos[:] = self.q
                mujoco.mj_forward(self.display_model,self.display_data)
                self.view = viewer.launch_passive(self.display_model,self.display_data)
            else:
                self.view = viewer.launch_passive(self.model,self.data)
            self.view.cam.lookat[:] = [.25,.25,.14] if self.ultrasound else [0, 0, .16]
            self.view.cam.distance = 1.12 if self.ultrasound else .72
            self.view.cam.azimuth = 180 if self.ultrasound else 20
            self.view.cam.elevation = -4 if self.ultrasound else -21

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
        self.mount_stop.set()
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
            normal,magnitude = contact(self.data.site("acoustic").xpos,self.radii,self.center)
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

    def check_speed(self, source, velocity):
        over = np.abs(velocity) > SPEED + 1e-7
        new = over & ~self.speed_faults.get(source,np.zeros(6,dtype=bool))
        self.speed_faults[source] = over
        if np.any(new):
            entries = [f"J{i+1} {np.degrees(velocity[i]):+.1f}°/s（限值 {np.degrees(SPEED[i]):.0f}°/s）"
                       for i in np.flatnonzero(new)]
            self.last_fault = f"ERROR {source}超速 " + "，".join(entries)
            self.speed_log.configure(state="normal")
            self.speed_log.insert("end",time.strftime("%H:%M:%S ")+self.last_fault+"\n")
            if int(self.speed_log.index("end-1c").split(".")[0]) > 501:
                self.speed_log.delete("1.0","2.0")
            self.speed_log.see("end")
            self.speed_log.configure(state="disabled")

    def tick(self):
        if self.closing:
            return
        if self.view is not None and not self.view.is_running():
            self.close()
            return
        previous = self.q.copy()
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
                    self.check_speed("CAN 反馈",self.velocity)
                    self.status.set("硬件跟随 · 六轴反馈正常（只读）")
        if self.ultrasound:
            self.poll_coverage()
            self.init_btn.state(["!disabled" if self.mode == "sim" else "disabled"])
            self.park_btn.state(["!disabled" if self.mode=="sim" and not self.parking else "disabled"])
            enabled = self.mode=="sim" and (self.track is None or len(self.track)>1)
            for control in (self.scrub,*self.scan_steps,self.write_btn):
                control.state(["!disabled" if enabled else "disabled"])
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
        now = time.monotonic()
        dt = max(now-self.last_motion,1e-9)
        self.last_motion = now
        if self.mode == "sim":
            delta = (self.q-previous+np.pi)%(2*np.pi)-np.pi
            self.check_speed("仿真请求（已限速）",delta/dt)
            # Keep a stalled UI from spending its entire missed time in one frame.
            ratio = max(1.,np.max(abs(delta)/SPEED)/min(dt,.04))
            self.q = previous + delta/ratio
            if self.ultrasound:
                following = manual and hasattr(self,"preview_target") and np.allclose(
                    (previous+delta-self.preview_target+np.pi)%(2*np.pi)-np.pi,0.,atol=1e-8)
                if not self.parking and (following or self.writing is not None):
                    # Before scanning, a manually jogged arm may need to approach alignment.
                    self.preview_work.d.qpos[:] = previous
                    mujoco.mj_kinematics(self.model,self.preview_work.d)
                    axis = self.preview_work.d.site("flange").xmat.reshape(3,3)[:,2]
                    if self.track is not None or np.linalg.norm(axis-[1,0,0]) < 1e-6:
                        self.q = self.preview_work.flange_locked_step(previous,delta/ratio,SPEED*min(dt,.04))
            self.velocity = (self.q-previous)/dt
        fresh = self.mode=="sim" or (self.mode=="hardware" and now-getattr(self,"feedback_at",0)<.5)
        speeds = "  ".join(f"J{i+1} {v:+.1f}" for i,v in enumerate(np.degrees(self.velocity)))
        self.speed_text.set("轴速 °/s   "+(speeds if fresh else "等待反馈")+
                            ("\n超速详情见「速度日志」" if hasattr(self,"last_fault") else ""))
        for (scale, label), angle in zip(self.rows, np.degrees(self.q)):
            scale.state(["!disabled" if manual else "disabled"])
            if not manual:
                scale.set(float((angle+180)%360-180))
            label.configure(text=f"{angle: .1f}°")
        xyz, rpy = update_pose(self.model, self.data, self.q, 0, 1)
        if self.ultrasound:
            self.display_forces()
            site = self.data.site("acoustic")
            xyz = box_mm(site.xpos)
            rpy = Rotation.from_matrix(site.xmat.reshape(3,3)).as_euler("xyz",degrees=True)
        self.pose.set(("探头盒坐标 XYZ (mm)   " if self.ultrasound else "XYZ (mm)   ") + "   ".join(f"{v:.2f}" for v in xyz) +
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
                dock(self.root, ("声弧 · 超声扫查三维视图" if self.ultrasound else "声弧 · 六轴三维视图")
                     if self.next_dock == 0. or not getattr(self,"viewer_branded",False) else None)
                self.viewer_branded = bool(self.root.winfo_ismapped())
                self.next_dock = time.monotonic() + .15
            with self.view.lock():
                if self.ultrasound:
                    self.display_data.qpos[:] = self.q
                    mujoco.mj_forward(self.display_model,self.display_data)
                scene = self.view.user_scn
                draw_ink(scene, self.ink, radius=.00045 if self.ultrasound else .001)
                if self.ultrasound:
                    lines = guides(self.slice_axis, self.position_mm[self.slice_axis],tuple(self.radii),tuple(self.center))
                    draw_coordinates(scene, lines+self.blue_lines, self.data.body("mount"),
                                     getattr(self,"start_mm",None), self.data.site("flange"))
                    draw_area(scene,self.area_cache.get(self.slice_axis,()))
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
        if self.closing:
            return
        self.closing = True
        self.mount_stop.set()
        if self.ultrasound:
            self.area_stop.set()
            self.slice_stop.set()
            if self.preview_job is not None:
                self.root.after_cancel(self.preview_job)
                self.preview_job = None
        self.cancel_plan.set()
        self.stop.set()
        self.finish_close()

    def finish_close(self):
        if self.worker and self.worker.is_alive():
            self.root.after(50, self.finish_close)
            return
        if self.view and self.view.is_running():
            self.view.close()
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--writing", action="store_true", help="Open the previous Assembly 4 writing scene")
    args = parser.parse_args()
    window = tk.Tk()
    Panel(window, ultrasound=not args.writing)
    window.mainloop()
