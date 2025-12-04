from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtCore import QObject, pyqtSignal
class ThreadPoolManager(QObject):
    """线程池管理器"""
    task_completed = pyqtSignal(object, object)  # 任务完成信号：任务ID，结果
    task_failed = pyqtSignal(object, object)     # 任务失败信号：任务ID，异常
    def __init__(self, max_workers=5, parent=None):
        super().__init__(parent)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_counter = 0
        self.pending_tasks = {}  # 存储未完成的任务
        self.completed_tasks = {}  # 存储已完成的任务

    def submit_task(self, task_func, *args, **kwargs):
        """提交任务到线程池"""
        task_id = self.task_counter
        self.task_counter += 1

        # 包装任务函数，以便捕获结果和异常
        def wrapped_task():
            try:
                result = task_func(*args, **kwargs)
                return task_id, result, None
            except Exception as e:
                return task_id, None, e

        # 提交任务并存储future对象
        future = self.executor.submit(wrapped_task)
        self.pending_tasks[task_id] = {
            'future': future,
            'function': task_func,
            'args': args,
            'kwargs': kwargs
        }

        # 连接future完成信号
        future.add_done_callback(self._handle_task_completion)

        return task_id

    def _handle_task_completion(self, future):
        """处理任务完成"""
        try:
            task_id, result, exception = future.result()

            # 从待处理任务中移除
            task_info = self.pending_tasks.pop(task_id, None)

            if exception is None:
                # 任务成功完成
                self.completed_tasks[task_id] = {
                    'result': result,
                    'status': 'completed'
                }
                self.task_completed.emit(task_id, result)
            else:
                # 任务失败
                self.completed_tasks[task_id] = {
                    'exception': exception,
                    'status': 'failed'
                }
                self.task_failed.emit(task_id, exception)

        except Exception as e:
            # 处理任务执行过程中的意外错误
            print(f"任务处理错误: {e}")

    def get_task_status(self, task_id):
        """获取任务状态"""
        if task_id in self.pending_tasks:
            return 'pending'
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]['status']
        return 'unknown'

    def shutdown(self, wait=True):
        """关闭线程池"""
        self.executor.shutdown(wait=wait)

    def get_active_task_count(self):
        """获取活动任务数量"""
        return len(self.pending_tasks)