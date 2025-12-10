# AdaptiPolish-Sys
AdaptiPolish-Sys 是一个基于点云与深度学习的缺陷检测自动初磨系统，通过融合机械臂控制、计算机视觉和实时监控技术，实现自动化磨抛作业。系统支持工件轮廓识别、路径规划、打磨参数配置等功能，适用于3D打印件、木材等材料的表面处理的解决方案
一.安装教程
1.一键安装
<img width="539" height="53" alt="image" src="https://github.com/user-attachments/assets/925681db-6fca-4d8c-a537-ddf968fd41f3" />
2.创建虚拟环境：
<img width="558" height="310" alt="image" src="https://github.com/user-attachments/assets/8b9ca0ef-0d65-4bb4-831b-f694a4ed56b0" />
3.安装上位机依赖库Python要求3.12：
<img width="551" height="69" alt="image" src="https://github.com/user-attachments/assets/6860a00c-2042-48c0-861d-8ed5217699a1" />
4.安装相机依赖库Python版本要求3.9：
<img width="551" height="54" alt="image" src="https://github.com/user-attachments/assets/cc02db3e-df0d-4f02-b80c-6babb496bb23" />
5.或者直接使用打包好的虚拟环境 要求:(同架构同系统:Bianbu)
6.特殊平台依赖：
<img width="551" height="102" alt="image" src="https://github.com/user-attachments/assets/03a4a5d1-7578-4f8c-b68b-383c57d29af1" />
二.配置机械臂连接
1.	通过串口或网口连接到MyCobot 280到电脑
三.配置相机服务
1.	通过网口连接到相机
2.	打开CameraService.py 修改 python_path 的值为实际Python3.9的路径
3.	启动CameraService服务
四.点云c++部署
1.搭建配置文件夹架构
例：C:\pointcloud_project\
    data\
        train\       ← 训练数据放到这里
    src\				← 运行文件放到这里
        Dataset.py
        build_dataloader.py(gpu版本)
        build_dataloader2.py（cpu版本）

2.点云必要的环境
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scikit-learn tqdm

3.搭建c++点云文件
	1、创建一个pc_backend.cpp文件（具体看资料）
	2、创建一个CMakeLists.txt文件（具体看资料）			

 命令行输入
 cd ~/Desktop/cse/pointcloud_project/src（自定义路径）
 rm -rf build
 mkdir build
 cd build
 cmake ..
 make -j4
 输入完成后会自动生成一个so文件，然后python中调用即可使用c++点云库
 启动系统
 python main.py
使用说明
AdaptiPolish-Sys 使用说明
一、界面布局
系统主界面分为三个主要区域：
1.	左侧区域：摄像头实时画面及检测控制
2.	中间区域：示教点管理、打磨控制和运动控制
3.	右侧区域：摄像头设置、机器人连接和辅助功能
二、核心功能操作指南
1. 机器人连接
1.	在右侧"机器人连接"区域输入IP地址（默认：192.168.25.181）和端口（默认：9000）
2.	点击"连接机器人"按钮
3.	连接成功后状态指示灯变为绿色
注意：首次使用前需确保机械臂已通电并通过USB连接到电脑
2. 摄像头设置与标定
1.	在右侧"摄像头设置"区域：
o	选择摄像头类型（本地/网络）
o	输入网络摄像头的IP和端口（如使用网络摄像头）
o	加载ONNX模型文件（点击"加载"按钮）
2.	点击"开始标定"按钮
3.	将两个ArUco标记放置在相机视野内（相距220mm）
4.	系统自动完成标定后显示"标定成功"
5.	点击获取点云数据按钮，系统会自动调用3D相机扫描点云，与标准工件进行对比检测缺陷，返回缺陷点坐标信息
6.	3D相机标定矩阵已集成在系统内部，返回缺陷点的坐标会自动转换
3. 示教点管理
1.	将机械臂移动到目标位置
2.	在中间"示教点管理"区域：
o	输入示教点名称
o	点击"保存当前位置"按钮
3.	在示教点列表中：
o	选择点位后点击"移动到选定点"进行定位
o	点击"删除选定点"移除不需要的点位
4. 轮廓识别与打磨
1.	在左侧区域点击"启动检测"按钮
2.	将工件放置在相机视野内
3.	系统自动识别工件轮廓并显示在画面中
4.	在中间"打磨控制"区域：
o	设置坐标偏移量（X/Y/Z方向）
o	配置打磨参数（循环次数、进深量、缩放比例）
o	点击"应用参数"确认设置
5.	点击"启动打磨"开始自动磨抛流程
5. 手动控制
1.	在右侧区域点击"手动控制"按钮
2.	在弹出的对话框中：
o	使能/放松各关节
o	启用/禁用自由移动模式
o	保存当前位置为示教点
6. 电机控制
1.	在右侧区域点击"电机控制"按钮
2.	在弹出的对话框中：
o	点击"连接电机"建立连接
o	使用按钮控制电机启停、正反转
o	通过滑块设置电机转速（0-500）
7. 坐标移动
1.	在中间"执行控制"区域的"XYZ坐标移动"部分
2.	输入目标坐标（X/Y/Z值）
3.	勾选"启用角度修正"（推荐）
4.	点击"移动"按钮执行定位
三、高级功能
路径循环打磨
1.	在"打磨参数"中设置循环次数>1
2.	系统将自动执行渐进式多层打磨  帮我修正一下这个markdown使它格式正确
