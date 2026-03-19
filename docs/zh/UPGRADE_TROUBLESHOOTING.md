# 升级故障排除指南

本指南帮助你解决 WhisperJAV 升级过程中的常见问题。

## 预检失败

### "磁盘空间不足"

**错误信息：** `Disk space: X.X GB available, need 5 GB`

**解决方案：**
1. 释放 WhisperJAV 所在磁盘的空间
2. 删除旧文件：`%TEMP%\*`、浏览器缓存、旧下载文件
3. 运行磁盘清理：`cleanmgr`

### "检测到错误的 Python 环境"

**错误信息：**
```
Running from: C:\some\other\python
Expected:     C:\Users\...\AppData\Local\WhisperJAV
```

**解决方案：**
使用 WhisperJAV 自带的 Python 运行升级脚本：
```cmd
%LOCALAPPDATA%\WhisperJAV\python.exe upgrade_whisperjav.py
```

### "WhisperJAV GUI 正在运行"

**错误信息：** `GUI check failed: WhisperJAV GUI is running`

**解决方案：**
1. 关闭 WhisperJAV GUI 窗口
2. 如果无法正常关闭，打开任务管理器并结束以下进程：
   - `pythonw.exe`（WhisperJAV GUI）
   - `WhisperJAV-GUI.exe`
3. 重新尝试升级

### "无网络连接"

**错误信息：** `Network: no connection to GitHub`

**解决方案：**
1. 检查网络连接
2. 尝试在浏览器中访问 https://github.com
3. 如果处于企业防火墙/代理环境：
   ```cmd
   set HTTPS_PROXY=http://your-proxy:port
   python.exe upgrade_whisperjav.py
   ```

## 升级执行失败

### "软件包升级失败"

**症状：** 升级在 "Upgrading WhisperJAV package..." 阶段停止

**可能原因：**
- 网络超时
- GitHub 暂时不可用
- pip 缓存损坏

**解决方案：**

1. **重试升级：**
   ```cmd
   python.exe upgrade_whisperjav.py
   ```

2. **清除 pip 缓存后重试：**
   ```cmd
   python.exe -m pip cache purge
   python.exe upgrade_whisperjav.py
   ```

3. **使用热补丁模式**（如果只需要最新代码）：
   ```cmd
   python.exe upgrade_whisperjav.py --wheel-only
   ```

### "依赖安装失败"

**症状：** 依赖安装阶段部分软件包安装失败

**解决方案：**

1. **继续操作** - 大多数依赖安装失败不是致命错误
2. **手动安装：**
   ```cmd
   python.exe -m pip install <package-name>
   ```

### "权限被拒绝" / "访问被拒绝"

**症状：** 文件操作失败

**解决方案：**
1. 关闭所有 WhisperJAV 窗口
2. 以管理员权限运行命令提示符
3. 临时关闭杀毒软件的实时防护

### "无法更新启动器"

**症状：** 桌面快捷方式未更新

**这不是致命错误。** WhisperJAV 仍可正常使用。手动修复方法：
1. 右键点击桌面快捷方式
2. 点击"属性"
3. 更新目标路径

## 升级后问题

### 升级后出现 "Import error"

**症状：** WhisperJAV 无法启动，显示导入错误

**解决方案：**

1. **重新安装（含依赖）：**
   ```cmd
   python.exe -m pip install --force-reinstall git+https://github.com/meizhong986/whisperjav.git@main
   ```

2. **检查软件包冲突：**
   ```cmd
   python.exe -m pip check
   ```

### 升级后 "CUDA not available"

**症状：** GPU 加速不再可用

**原因：** CPU 版 PyTorch 覆盖了 CUDA 版本

**解决方案：**
```cmd
REM 先检查你的 CUDA 版本
nvidia-smi

REM 重新安装 CUDA 版 PyTorch（以 CUDA 12.1 为例）
python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### 升级后版本未变化

**症状：** `--version` 仍显示旧版本

**解决方案：**
1. 清除 Python 的导入缓存：
   ```cmd
   python.exe -c "import whisperjav; print(whisperjav.__file__)"
   ```
2. 重新安装：
   ```cmd
   python.exe -m pip install --force-reinstall --no-cache-dir git+https://github.com/meizhong986/whisperjav.git@main
   ```

## 高级选项

### 跳过预检

**请谨慎使用** —— 仅在你清楚自己在做什么时使用：
```cmd
python.exe upgrade_whisperjav.py --skip-preflight
```

### 热补丁模式

仅更新 WhisperJAV 代码，跳过所有依赖：
```cmd
python.exe upgrade_whisperjav.py --wheel-only
```

### 非交互模式

用于脚本化升级：
```cmd
python.exe upgrade_whisperjav.py --yes
```

## 以上方法都无效时

请参阅[手动回滚指南](MANUAL_ROLLBACK.md)了解完整的恢复方案。

## 报告问题

如果你遇到本指南未涵盖的问题：

1. 前往 [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) 提交问题
2. 请附上以下信息：
   - 升级过程中的完整控制台输出
   - Windows 版本（`winver`）
   - Python 版本（`python.exe --version`）
   - 当前 WhisperJAV 版本（如果已知）
