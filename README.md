# YOLO-CSharp-Deployment

> YOLO 系列目标检测模型的 C#/.NET 部署与推理项目集合，覆盖 ONNX Runtime、TensorRT、OpenCV DNN、OpenVINO 等主流部署后端。

本仓库汇集了多个 YOLO 系列模型（YOLOv5 / YOLOv8 等）在 C#/.NET 平台下的部署与推理实现，包含 WinForm 应用、ONNX Runtime 推理、TensorRT 加速、OpenCV DNN 推断、二维码识别等场景，可用于学习、二次开发和工程落地参考。

## 项目列表

| 目录 | YOLO 版本 | 技术栈 / 后端 | 说明 |
|------|-----------|---------------|------|
| `Csharp_deploy_Yolov8-master` | YOLOv8 | C# WinForm / OpenVINO / TensorRT | YOLOv8 的 C# WinForm 部署，含 OpenVINO、TensorRT 部署文档 |
| `NetOnnx` | YOLOv8 | C# WinForm / ONNX Runtime | YOLOv8 ONNX 模型的 WinForms 推理示例 |
| `YOLOv5-Seg-OnnxRuntime-main` | YOLOv5-seg | C# / ONNX Runtime | YOLOv5 实例分割的 C# ONNX Runtime 实现 |
| `YOLOv5DetectionWithCSharp-main` | YOLOv5 | C# / ONNX Runtime | YOLOv5 目标检测的 C# 实现（含 Python 训练脚本） |
| `yolov5-net-master` | YOLOv5 | C# / ONNX Runtime | Yolov5Net：YOLOv5 的 .NET 推理库与 WinForm 示例 |
| `yolov5-net-master-master` | YOLOv5 | C# / ONNX Runtime | Yolov5Net 的视频检测 / 录制变体 |
| `onnxruntime_yolov5` | YOLOv5 | ONNX Runtime | YOLOv5 ONNX Runtime 推理 |
| `yolov5_dnn` | YOLOv5 | OpenCV DNN | YOLOv5 OpenCV DNN 推理 |
| `yolov5_qrcode` | YOLOv5 | C# / OpenCV | YOLOv5 二维码识别 |
| `Tensorrt_yolov5_qrcode` | YOLOv5 | TensorRT | YOLOv5 TensorRT 加速 + 二维码识别 |
| `Tensorrt_yolov8_qrcode` | YOLOv8 | TensorRT | YOLOv8 TensorRT 加速 + 二维码识别 |

## 目录结构

```
YOLO-CSharp-Deployment/
├── README.md
├── Csharp_deploy_Yolov8-master/
├── NetOnnx/
├── YOLOv5-Seg-OnnxRuntime-main/
├── YOLOv5DetectionWithCSharp-main/
├── yolov5-net-master/
├── yolov5-net-master-master/
├── onnxruntime_yolov5/
├── yolov5_dnn/
├── yolov5_qrcode/
├── Tensorrt_yolov5_qrcode/
└── Tensorrt_yolov8_qrcode/
```

## 环境要求

- Windows 10 / 11
- .NET 6 或 .NET Framework（按具体项目而定）
- ONNX Runtime（C# 包 `Microsoft.ML.OnnxRuntime`）
- OpenCvSharp / Emgu.CV（按项目而定）
- CUDA + TensorRT（TensorRT 相关项目需要）
- Visual Studio 2022

## 快速开始

进入任一子目录，用 Visual Studio 打开对应 `.sln` 解决方案文件，还原 NuGet 依赖后编译运行。例如：

```bash
cd NetOnnx
# 用 VS 打开 NetOnnx.sln，还原并运行
```

各子项目若自带 README，请优先阅读该目录内的说明。

## 相关项目

- YOLO 训练源码集合：https://github.com/cslht11/YOLO-Project-Collection
- 图像匹配与分割集合：https://github.com/cslht11/Image-Matching-Collection

## 许可证

本仓库内各子项目分别遵循其各自的许可证，详见各子目录中的 LICENSE 文件或说明。

## 联系方式

- GitHub: [@cslht11](https://github.com/cslht11)
- Email: heitieya@163.com