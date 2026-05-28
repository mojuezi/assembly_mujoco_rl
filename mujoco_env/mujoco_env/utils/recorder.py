
import queue
import time
import threading
from typing import Optional, List

class DataSaver:
    """通用数据保存器 (CSV)"""

    def __init__(self, filename: str, headers: List[str]):
        """
        初始化数据保存器

        Args:
            filename: 保存文件名
            headers: CSV头部列表
        """
        self.filename = filename
        self.file = None
        self.headers = headers

        try:
            self.file = open(filename, 'w')
            # 写入CSV头部
            self.file.write(",".join(headers) + "\n")
            print(f"✅ 数据将保存到: {filename}")
        except Exception as e:
            print(f"❌ 创建数据文件失败: {e}")
            self.file = None

    def save_data(self, *args):
        """保存一行数据"""
        if self.file is None:
            return

        try:
            # 将所有参数转换为字符串并用逗号连接
            # 处理列表/数组类型的参数
            formatted_args = []
            for arg in args:
                if hasattr(arg, '__iter__') and not isinstance(arg, str):
                    formatted_args.extend([f"{x:.6f}" if isinstance(x, float) else str(x) for x in arg])
                else:
                    formatted_args.append(f"{arg:.6f}" if isinstance(arg, float) else str(arg))
            
            line = ",".join(formatted_args) + "\n"
            self.file.write(line)
            self.file.flush()  # 立即写入磁盘
        except Exception as e:
            print(f"保存数据失败: {e}")

    def close(self):
        """关闭文件"""
        if self.file:
            self.file.close()
            print(f"✅ 数据已保存到: {self.filename}")


class DataLogger:
    """实时数据记录和打印器 (线程安全)"""

    def __init__(self, print_interval=0.1, formatter=None):
        """
        初始化数据记录器

        Args:
            print_interval: 打印间隔时间（秒）
            formatter: 自定义格式化函数，接收数据字典返回字符串。如果为None，则使用默认格式。
        """
        self.data_queue = queue.Queue()
        self.running = False
        self.print_interval = print_interval
        self.last_print_time = 0
        self.formatter = formatter
        self.print_thread = None

    def log_data(self, data_dict):
        """记录数据"""
        if self.running:
            self.data_queue.put(data_dict)

    def start(self):
        """启动打印线程"""
        self.running = True
        self.print_thread = threading.Thread(target=self._print_loop)
        self.print_thread.daemon = True
        self.print_thread.start()

    def _print_loop(self):
        """实时打印数据循环"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_print_time >= self.print_interval:
                    # 获取最新数据
                    data_items = []
                    while not self.data_queue.empty():
                        data_items.append(self.data_queue.get())

                    if data_items:
                        # 只打印最新的数据
                        latest_data = data_items[-1]
                        
                        if self.formatter:
                            print(f"\r{self.formatter(latest_data)}", end='', flush=True)
                        else:
                            # 默认打印
                            print(f"\rData: {latest_data}", end='', flush=True)

                        self.last_print_time = current_time

                time.sleep(0.01)  # 小延迟避免CPU占用过高

            except Exception as e:
                print(f"\n数据打印错误: {e}")
                break

    def stop(self):
        """停止数据记录"""
        self.running = False
        if self.print_thread:
            self.print_thread.join(timeout=1.0)

