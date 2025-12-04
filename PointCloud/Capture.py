from PyCameraSDK.AinstecError import *  # 导入Ainstec相机错误模块
from PyCameraSDK.Common import *  # 导入相机SDK通用模块
from PyCameraSDK.Camera import *  # 导入相机操作主模块
from Util import *  # 导入自定义工具函数
import argparse
import os
import datetime


# 打印相机信息列表
def PrintCamInfoList(camInfoList):
    # 遍历所有发现的相机信息
    for i in range(len(camInfoList)):
        # 输出相机索引、IP地址、错误码和系统版本
        print("索引", i, ":", camInfoList[i].cameraIP, "返回码:",
              camInfoList[i].errorCode, "相机版本:", camInfoList[i].cameraSystemVersion)


# 打开指定IP的相机
def OpenOneCamera(cam, camInfoList, strWantedIP=None):
    # 遍历相机列表尝试连接
    for i in range(len(camInfoList)):
        # 如果指定了IP且不匹配则跳过
        if (strWantedIP != None and camInfoList[i].cameraIP != strWantedIP):
            continue
        # 尝试打开相机
        if (cam.Open(camInfoList[i]) == AC_OK):
            print("\033[1;32m", "成功打开",
                  camInfoList[i].cameraIP, "\033[0m")
            return AC_OK, camInfoList[i]  # 返回成功状态和相机信息


    print("\033[1;31m" "无法打开任何相机。", "\033[0m")
    return AC_E_NO_CAMERA, CameraInfo()  # 返回错误代码和空相机信息


# 设置所有输出选项
def OutputAll(camInfo, isSend=True):
    # 批量设置各种数据输出开关
    camInfo.outputSettings.sendPoint3D = isSend  # 3D点云
    camInfo.outputSettings.sendPointUV = isSend  # UV坐标
    camInfo.outputSettings.sendTriangleIndices = isSend  # 三角面索引
    camInfo.outputSettings.sendDepthmap = isSend  # 深度图
    camInfo.outputSettings.sendNormals = isSend  # 法线向量
    camInfo.outputSettings.sendPointColor = isSend  # 点云颜色
    camInfo.outputSettings.sendTexture = isSend  # 纹理图像
    camInfo.outputSettings.sendRemapTexture = isSend  # 重映射纹理


# 仅输出3D点云数据
def OutputOnlyPoint3D(camInfo):
    OutputAll(camInfo, False)  # 关闭所有输出
    camInfo.outputSettings.sendPoint3D = True  # 单独开启3D点云


# 自定义创建输出目录函数
def create_custom_outdir(base_path=None, custom_name=None):
    if base_path is None:
        base_path = '.'

    if custom_name:
        # 使用自定义文件夹名
        path = os.path.join(base_path, custom_name)
    else:
        # 使用时间戳作为默认
        t = datetime.datetime.now()
        path = os.path.join(base_path, '3DRecon', t.strftime('%Y%m%d'), t.strftime('%H%M%S'))

    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)
        print(path + ' 创建成功')
        return path
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return path


# 保存捕获的各种图像数据
def SaveImages(camInfo, frameData, save_path=None):
    if save_path is None:
        filePath = create_custom_outdir()
    else:
        filePath = create_custom_outdir(base_path=save_path.get('base_path', '.'),
                                        custom_name=save_path.get('custom_name'))

    print(f"数据保存到: {filePath}")

    # 根据数据类型调用不同的保存函数
    if frameData.textureSize:
        save_rgb(frameData.texture, camInfo.camParam, filePath)  # 保存RGB纹理

    if frameData.depthmapSize:
        save_deepmap2tiff(frameData.depthmap, camInfo.camParam, filePath)  # 深度图转TIFF

    if frameData.textureSize and frameData.pointUVSize:
        save_rgb_align_depth(frameData.texture, frameData.pointUV,
                             camInfo.camParam, filePath)  # 保存深度对齐的RGB

    if frameData.point3DSize and frameData.pointUVSize and frameData.triangleIndicesSize:
        save_point2wrl(frameData, filePath)  # 点云保存为WRL格式

    if frameData.point3DSize:
        save_point2pcd(frameData, filePath)  # 点云保存为PCD格式

    if frameData.depthmapSize:
        save_deepmap(frameData.depthmap, camInfo.camParam, filePath)  # 保存原始深度图

    if frameData.point3DSize and frameData.triangleIndicesSize:
        save_point2ply(frameData, filePath)  # 点云保存为PLY格式

    if frameData.point3DSize and frameData.normalsSize:
        save_point2ply_normal_color(frameData, filePath)  # 带法线的彩色点云

    if frameData.remapTextureSize:
        save_ir(frameData.remapTexture, camInfo.camParam, filePath, True)  # 保存红外图像


# 设置相机参数
def SetCameraParameters(cam, camInfo, camera_params):
    """
    设置相机参数
    camera_params: 字典，包含要设置的参数和值
    """
    if not camera_params:
        return AC_OK

    # 参数名称映射（将通用名称映射到相机支持的具体参数名）
    param_mapping = {
        'IR_Exposure': 'Exposure',  # IR曝光
        'IR_Gain': 'Gain',  # IR增益
        'Rgb_ExposureAbsolute': 'Texture_ExposureAbsolute',  # RGB曝光
        'Capture_WorkMode': 'TriggerMode'  # 工作模式
    }

    for param_name, param_value in camera_params.items():
        try:
            # 使用映射后的参数名
            mapped_param_name = param_mapping.get(param_name, param_name)

            # 尝试将参数名转换为ParamType枚举
            if hasattr(ParamType, mapped_param_name):
                param_type = getattr(ParamType, mapped_param_name)
                ret = cam.SetValue(camInfo, param_type, param_value)
                if ret == AC_OK:
                    print(f"成功设置参数 {param_name} -> {mapped_param_name} = {param_value}")
                else:
                    print(f"设置参数 {param_name} -> {mapped_param_name} 失败，错误码: {ret}")
            else:
                # 如果参数名不是枚举，尝试作为字符串参数设置
                ret = cam.SetValue(camInfo, mapped_param_name, param_value)
                if ret == AC_OK:
                    print(f"成功设置参数 {param_name} -> {mapped_param_name} = {param_value}")
                else:
                    print(f"设置参数 {param_name} -> {mapped_param_name} 失败，错误码: {ret}")
        except Exception as e:
            print(f"设置参数 {param_name} 时发生错误: {e}")

    return AC_OK

# 主函数，支持命令行参数
def main():
    parser = argparse.ArgumentParser(description='3D相机数据采集程序')
    parser.add_argument('--camera-ip', type=str, help='指定相机IP地址')
    parser.add_argument('--output-dir', type=str, help='指定输出目录基路径')
    parser.add_argument('--folder-name', type=str, help='指定保存文件夹名称')
    parser.add_argument('--output-mode', choices=['all', 'point3d'],
                        default='all', help='输出数据模式')

    # 相机参数设置
    parser.add_argument('--ir-exposure', type=int, help='红外曝光时间(ms)')
    parser.add_argument('--ir-gain', type=int, help='红外增益')
    parser.add_argument('--rgb-exposure', type=int, help='RGB相机曝光时间')
    parser.add_argument('--work-mode', type=int, help='工作模式')

    args = parser.parse_args()

    # 准备保存路径配置
    save_path_config = {
        'base_path': args.output_dir,
        'custom_name': args.folder_name
    }

    # 准备相机参数配置
    camera_params = {}
    if args.ir_exposure:
        camera_params['IR_Exposure'] = args.ir_exposure
    if args.ir_gain:
        camera_params['IR_Gain'] = args.ir_gain
    if args.rgb_exposure:
        camera_params['Rgb_ExposureAbsolute'] = args.rgb_exposure
    if args.work_mode:
        camera_params['Capture_WorkMode'] = args.work_mode

    ##### 主程序开始 ####
    cam = Camera().CreateCamera()  # 创建相机实例

    # 发现网络中的相机
    ret, camInfoList = cam.DiscoverCameras()
    PrintCamInfoList(camInfoList)  # 打印发现的相机信息

    # 尝试打开相机（可指定IP）
    ret, camInfo = OpenOneCamera(cam, camInfoList, args.camera_ip)

    # 成功打开相机后的操作
    if ret == AC_OK:
        # 设置相机参数
        SetCameraParameters(cam, camInfo, camera_params)

        # 根据输出模式配置
        if args.output_mode == 'point3d':
            OutputOnlyPoint3D(camInfo)  # 仅输出3D点云
        else:
            OutputAll(camInfo)  # 启用所有数据输出

        frameData = FrameData()  # 创建帧数据容器
        ret = cam.Capture(camInfo, frameData)  # 执行捕获操作

        # 捕获成功则保存数据
        if ret == AC_OK:
            SaveImages(camInfo, frameData, save_path_config)
        else:
            print(f"捕获失败，错误码: {ret}")

        cam.Close(camInfo)  # 关闭相机连接
    else:
        print("无法打开相机，程序结束")


##### 程序结束 ####

if __name__ == "__main__":
    main()