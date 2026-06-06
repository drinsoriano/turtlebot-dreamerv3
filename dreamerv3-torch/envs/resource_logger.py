"""
resource_logger.py — Lightweight per-episode resource sampling.
Called only at episode end. Never crashes training (all errors caught).

Primary metrics are process-specific (DreamerV3 process only).
System/device-wide columns are retained as labeled context.
"""
from __future__ import annotations
import csv
import datetime
import os
import time

_psutil = None
_nvml_handle = None
_nvml_ok = False
_gpu_name = ''


def _try_import_psutil():
    global _psutil
    try:
        import psutil
        _psutil = psutil
    except ImportError:
        pass


def _try_init_nvml():
    global _nvml_handle, _nvml_ok, _gpu_name
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _gpu_name = pynvml.nvmlDeviceGetName(_nvml_handle)
        if isinstance(_gpu_name, bytes):
            _gpu_name = _gpu_name.decode()
        _nvml_ok = True
    except Exception:
        pass


_try_import_psutil()
_try_init_nvml()


class ResourceLogger:
    def __init__(self, run_name: str, stage: int, odometry_mode: str,
                 device: str, lidar: int):
        self._run_name      = run_name
        self._stage         = stage
        self._odometry_mode = odometry_mode
        self._device        = device   # training compute device: 'cpu' or 'cuda'
        self._lidar_beams   = lidar
        # perception columns — hardcoded until future modes exist
        self._perception_mode    = 'lidar'
        self._imu_enabled        = False
        self._depth_enabled      = False
        self._depth_camera_count = 0
        self._depth_resolution   = ''
        self._sensor_config_id   = f'lidar{lidar}_{odometry_mode}'

        self._session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._start_time = time.time()
        self._proc = None
        if _psutil is not None:
            try:
                self._proc = _psutil.Process(os.getpid())
                self._proc.cpu_percent(interval=None)  # prime the counter
            except Exception:
                pass

        os.makedirs('./csv_logs', exist_ok=True)
        path = f'./csv_logs/resource_{run_name}.csv'
        is_new = not os.path.exists(path)
        if not is_new:
            print(f'[resource_logger] Appending to existing CSV: csv_logs/resource_{run_name}.csv'
                  f'  session_id={self._session_id}'
                  f'  (previous sessions remain; filter by session_id to isolate runs)')
        self._file = open(path, 'a', newline='')
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow([
                'datetime', 'session_id', 'run_name', 'episode', 'stage',
                'device',
                'perception_mode', 'lidar_beams', 'odometry_mode',
                'imu_enabled', 'depth_enabled', 'depth_camera_count',
                'depth_resolution', 'sensor_config_id',
                'episode_wall_time_sec', 'total_wall_time_sec',
                # process-specific (primary training cost):
                'cpu_percent_process', 'ram_used_mb_process', 'gpu_memory_used_mb_process',
                # system/device-wide (context reference):
                'cpu_percent_system', 'ram_percent_system',
                'gpu_available', 'gpu_name',
                'gpu_util_percent_device', 'gpu_memory_used_mb_device',
                'gpu_memory_total_mb', 'gpu_memory_percent_device',
                'gpu_power_watts', 'gpu_temperature_c',
            ])
            self._file.flush()

    def log_episode(self, episode: int, episode_start: float) -> None:
        try:
            now = time.time()
            ep_wall    = round(now - episode_start, 3)
            total_wall = round(now - self._start_time, 3)

            # process-specific (primary)
            cpu_proc = ram_mb = cpu_sys = ram_pct = ''
            if _psutil is not None and self._proc is not None:
                try:
                    cpu_proc = round(self._proc.cpu_percent(interval=None), 2)
                    cpu_sys  = round(_psutil.cpu_percent(interval=None), 2)
                    mi       = self._proc.memory_info()
                    ram_mb   = round(mi.rss / 1024 / 1024, 2)
                    ram_pct  = round(_psutil.virtual_memory().percent, 2)
                except Exception:
                    pass

            # GPU metrics
            gpu_avail = False
            gpu_nm = gpu_util_dev = gpu_mem_proc = ''
            gpu_mem_dev = gpu_mem_tot = gpu_mem_pct_dev = gpu_pw = gpu_temp = ''
            if _nvml_ok and _nvml_handle is not None:
                try:
                    import pynvml
                    gpu_avail = True
                    gpu_nm    = _gpu_name

                    # device-wide utilization (NVML has no per-process util)
                    util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
                    gpu_util_dev = util.gpu

                    # device-wide memory (context reference)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
                    gpu_mem_dev     = round(mem.used  / 1024 / 1024, 2)
                    gpu_mem_tot     = round(mem.total / 1024 / 1024, 2)
                    gpu_mem_pct_dev = round(mem.used  / mem.total * 100, 2)

                    # process-specific GPU memory (primary GPU metric)
                    try:
                        pid   = os.getpid()
                        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(_nvml_handle)
                        for p in procs:
                            if p.pid == pid:
                                gpu_mem_proc = round(p.usedGpuMemory / 1024 / 1024, 2)
                                break
                    except Exception:
                        pass

                    try:
                        gpu_pw = round(
                            pynvml.nvmlDeviceGetPowerUsage(_nvml_handle) / 1000, 2)
                    except Exception:
                        pass
                    try:
                        gpu_temp = pynvml.nvmlDeviceGetTemperature(
                            _nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        pass
                except Exception:
                    pass

            self._writer.writerow([
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                self._session_id, self._run_name, episode, self._stage,
                self._device,
                self._perception_mode, self._lidar_beams, self._odometry_mode,
                self._imu_enabled, self._depth_enabled,
                self._depth_camera_count, self._depth_resolution,
                self._sensor_config_id,
                ep_wall, total_wall,
                # process-specific primary:
                cpu_proc, ram_mb, gpu_mem_proc,
                # system/device-wide context:
                cpu_sys, ram_pct,
                gpu_avail, gpu_nm,
                gpu_util_dev, gpu_mem_dev, gpu_mem_tot, gpu_mem_pct_dev,
                gpu_pw, gpu_temp,
            ])
            self._file.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass
