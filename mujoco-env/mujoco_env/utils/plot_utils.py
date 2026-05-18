"""
绘图工具 - 极简稳定版
"""

import numpy as np
import multiprocessing as mp
import time
import os
from typing import List

class DataDeque:
    def __init__(self, data_num: int, x_limit: int = 1000):
        self.data_num = data_num
        self.x_limit = x_limit
        self.arg_list = [[] for _ in range(data_num)]
    
    def push(self, data: List):
        for i in range(min(len(data), self.data_num)):
            self.arg_list[i].append(data[i])
            if len(self.arg_list[i]) > self.x_limit:
                self.arg_list[i].pop(0)
    
    def data(self):
        return self.arg_list

def plot_worker(row, col, label, x_limit, data_queue):
    """独立绘图进程的主函数"""
    # 延迟导入，确保在子进程中初始化图形环境
    import matplotlib
    # 强制在 Linux 上尝试多种后端
    backends = ['TkAgg', 'Qt5Agg', 'QtAgg', 'Agg']
    success = False
    for b in backends:
        try:
            matplotlib.use(b)
            import matplotlib.pyplot as plt
            # 测试是否能工作
            plt.figure(figsize=(1,1))
            plt.close()
            success = True
            break
        except:
            continue
    
    if not success:
        with open("plot_fatal.log", "w") as f:
            f.write("Failed to initialize any matplotlib backend")
        return

    import matplotlib.pyplot as plt
    
    try:
        plt.ion() # 开启交互模式
        fig, axes = plt.subplots(row, col, figsize=(10, 6), squeeze=False)
        fig.canvas.manager.set_window_title('Force Sensor Monitor')
        axes = axes.flatten()
        
        line_num = sum(len(l) for l in label)
        local_deque = DataDeque(line_num, x_limit)
        
        lines = []
        for k, ax in enumerate(axes):
            ax_lines = []
            if k < len(label):
                for l_name in label[k]:
                    line, = ax.plot([], [], label=l_name)
                    ax_lines.append(line)
                ax.legend(loc='upper left')
                ax.grid(True)
            lines.append(ax_lines)
        
        print("DEBUG: Oscilloscope window opened.")
        
        while True:
            # 检查是否有新数据
            new_data = False
            while not data_queue.empty():
                try:
                    point = data_queue.get_nowait()
                    if point is None: # 退出信号
                        plt.close(fig)
                        return
                    local_deque.push(point)
                    new_data = True
                except:
                    break
            
            if new_data:
                data_snapshot = local_deque.data()
                data_idx = 0
                for k, ax in enumerate(axes):
                    if k >= len(label): break
                    
                    y_min, y_max = float('inf'), float('-inf')
                    for line in lines[k]:
                        y_vals = data_snapshot[data_idx]
                        x_vals = np.arange(len(y_vals))
                        line.set_data(x_vals, y_vals)
                        if y_vals:
                            y_min = min(y_min, min(y_vals))
                            y_max = max(y_max, max(y_vals))
                        data_idx += 1
                    
                    if y_min != float('inf'):
                        ax.set_xlim(0, x_limit)
                        margin = max((y_max - y_min) * 0.1, 0.1)
                        ax.set_ylim(y_min - margin, y_max + margin)
                
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            time.sleep(0.05) # 20Hz 刷新速率，避免过度占用 CPU
            
            # 检查窗口是否还活着
            if not plt.fignum_exists(fig.number):
                break
                
    except Exception as e:
        import traceback
        with open("plot_crash.log", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())

class DataPlot:
    def __init__(self, row=2, col=1, label=[[]], x_limit=1000):
        self.data_queue = mp.Queue(maxsize=100)
        # 使用默认 context (fork on Linux)，通常对图形后端兼容性稍差但启动更快
        # 如果 fork 不行，再考虑 spawn
        self.process = mp.Process(
            target=plot_worker,
            args=(row, col, label, x_limit, self.data_queue)
        )
        self.process.daemon = False
        self.process.start()

    def generate_data(self, *args):
        if not self.process.is_alive(): return
        data = []
        for arg in args:
            if isinstance(arg, (np.ndarray, list)): data.extend(arg)
            else: data.append(arg)
        try:
            if self.data_queue.full(): self.data_queue.get_nowait()
            self.data_queue.put_nowait(data)
        except: pass

    def join(self):
        if self.process.is_alive():
            self.process.join()

class ForceSensorPlotter:
    def __init__(self, x_limit=1000):
        self.plotter = DataPlot(row=2, col=1, label=[['Fx', 'Fy', 'Fz'], ['Tx', 'Ty', 'Tz']], x_limit=x_limit)
    def update(self, force, torque):
        self.plotter.generate_data(force, torque)
    def close(self):
        try: self.plotter.data_queue.put(None)
        except: pass
        self.plotter.join()

__all__ = ['ForceSensorPlotter']
