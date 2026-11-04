# 装配体 4 · 六轴机械臂正运动学

`arm4.xml` 是由 STEP 装配姿态建立的 MuJoCo 六转动关节模型。`arm4_fk.py` 输出 J6 法兰中心的位置、欧拉角、四元数和齐次变换，并用独立的旋转矩阵计算核对 MuJoCo 的结果。

## 运行

在 Python 3.10 环境下，从仓库根目录执行：

```sh
python -m pip install mujoco==3.1.6 numpy scipy
python arm4_fk.py
python arm4_fk.py 30 -20 35 10 -15 45
```

六个输入依次为 J1 至 J6 相对 STEP 保存姿态的角度，单位为度，正方向按 XML 中各关节轴的右手定则。省略参数使用全零姿态。

## 坐标与核对值

基坐标系原点为底板上表面中心，坐标轴与 STEP 装配坐标系平行，+Z 向上。末端为 3507 输出侧端面中心，+Z 沿输出轴向外，+X 取电机 CAD 局部 +X。RPY 的旋转矩阵为 Rz(yaw) Ry(pitch) Rx(roll)，四元数顺序为 (w,x,y,z)。

| 关节角（度） | 法兰位置 XYZ（mm） | RPY（度） |
|---|---|---|
| 0, 0, 0, 0, 0, 0 | 0.00, −134.62, 241.94 | 91.516, 0, 0 |
| 30, −20, 35, 10, −15, 45 | 16.84, 25.30, 398.14 | 11.744, 18.358, −82.724 |

模型使用简化连杆，适用于法兰位姿计算。实机关节限位、编码器零点、质量惯量、驱动参数和碰撞外形尚未标定。

## 六轴达妙 CAN 调试

`dm_debug.py` 是独立的 Python 3.10+ 命令行工具。J1～J3 为 4340、J4～J5 为 4310、J6 为 3507，编码量程从每台驱动器读取。默认电机 ID 为 0x01～0x06，反馈 Master ID 为 0x11～0x16；用达妙上位机逐台配置为唯一 ID，也可用 `--ids` 和 `--masters` 分别传六个逗号分隔的 ID。使用经典 CAN 标准帧，并保证只有一个程序控制总线。

Windows 达妙 U2CAN 串口版请安装 pyserial，将 COM5 改成实际端口。串口默认 921600，可用 `--baud` 更改；CAN 侧波特率由转接器和驱动器预先配成一致。原生 CAN 使用 python-can，例如 Linux 下已配置的 can0 使用 `--interface socketcan --channel can0`，Windows PEAK 使用 `--interface pcan --channel PCAN_USBBUS1`（需厂商驱动）。

```sh
python -m pip install pyserial
python dm_debug.py --channel COM5 monitor --once
python dm_debug.py --channel COM5 monitor
python dm_debug.py --channel COM5 jog 6 2
python dm_debug.py --channel COM5 move6 1 -1 1 0 0 2
```

monitor 只读，显示位置、速度、估算扭矩、状态码和两路温度，0 表示失能、1 表示使能。运动命令要求所选轴均处于 MIT 模式且失能无故障，以电机当前角度为起点，各轴增量不超过 ±5°。六轴按同一插值进度逐帧发送，受总线延迟影响。每次结束或异常时，程序尝试失能本次接管的电机并读取状态确认。输出的目标和实测角度用于判断跟踪情况。

首次点动请支撑肩肘和末端，确认运动方向与机械限位，并在驱动器预设通信超时保护。低增益点动没有重力补偿，失能后电机会释放；物理断线时软件无法保证失能帧送达。总线角度与 MuJoCo 零点需另行标定。协议依据见[达妙官方 U2CAN Python 例程](https://github.com/dmBots/motor-sdk/tree/main/Python例程/u2can)。

无硬件回归测试：`python -m unittest -v test_dm_debug`。测试覆盖模拟六轴点动及控制帧发送失败后的失能清理。

## 装配体 4 的 CAN / MuJoCo 运动学孪生

`arm4_twin.py` 在同一个进程中读取六轴反馈、计算法兰位姿并更新 MuJoCo 三维视图。默认硬件模式只读。输出 XYZ 单位为 mm，RPY 单位为度，坐标系见上文。无硬件可直接运行：

```sh
python arm4_twin.py --demo
python arm4_twin.py --demo --headless --frames 10
```

实机先运行六轴 monitor 读数，将机械臂置于 STEP 保存姿态，记录六个电机角度作为 `--zero-deg`，逐轴核对编码器方向后设置 `--signs`。换算关系为 `模型角 = 方向 × (实测角 − 零位读数)`。示例中的六个零和正号须替换为标定值：

```sh
python arm4_twin.py --channel COM5 --zero-deg 0 0 0 0 0 0 --signs 1 1 1 1 1 1
```

在这条命令末尾追加 `--jog 6 2` 可进行 J6 点动并同步显示，或追加 `--move6 1 -1 1 0 0 2` 联动。增量沿用调试工具的**电机原生角度方向**。调试和孪生共用一个 CAN 连接，运行前关闭其他占用转接器的程序。动作结束后程序失能所控制的轴，首次点动须支撑机械臂。退出查看器会终止本次点动。六轴反馈按序采集，任一轴反馈超时或报错即停止更新。

模型显示简化连杆、法兰位置与方向。当前实现用于运动学跟随，不估算碰撞力、负载能力或驱动动态。通过 `python -m unittest -v test_dm_debug test_arm4_twin` 运行离线联调测试。
