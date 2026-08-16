using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Yolov5Net.Scorer.Extensions;
using Yolov5Net.Scorer.Models.Abstract;

namespace Yolov5Net.Scorer
{

    /// Yolov5 评分器
    public class YoloScorer<T> : IDisposable where T : YoloModel
    {
        private readonly T _model;

        private readonly InferenceSession _inferenceSession;


        /// 输出值在0和1之间。
        private float Sigmoid(float value)
        {
            return 1 / (1 + (float)Math.Exp(-value));
        }


        /// 将xywh边界框格式转换为xyxy格式。
        private float[] Xywh2xyxy(float[] source)
        {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }


        /// 将值限制在[min, max]的范围内并返回。
        public float Clamp(float value, float min, float max)
        {
            return (value < min) ? min : (value > max) ? max : value;
        }

        ///处理图像之前检查并应用EXIF方向标记
        private Image ApplyExifOrientation(Image img)
        {
            const int orientationId = 0x112; // 处理图像之前检查并应用EXIF方向标记

            if (!img.PropertyIdList.Contains(orientationId)) return img;

            var orientation = (int)img.GetPropertyItem(orientationId).Value[0];
            var rotateFlip = RotateFlipType.RotateNoneFlipNone;

            switch (orientation)
            {
                case 1: rotateFlip = RotateFlipType.RotateNoneFlipNone; break;
                case 2: rotateFlip = RotateFlipType.RotateNoneFlipX; break;
                case 3: rotateFlip = RotateFlipType.Rotate180FlipNone; break;
                case 4: rotateFlip = RotateFlipType.Rotate180FlipX; break;
                case 5: rotateFlip = RotateFlipType.Rotate90FlipX; break;
                case 6: rotateFlip = RotateFlipType.Rotate90FlipNone; break;
                case 7: rotateFlip = RotateFlipType.Rotate270FlipX; break;
                case 8: rotateFlip = RotateFlipType.Rotate270FlipNone; break;
            }

            if (rotateFlip != RotateFlipType.RotateNoneFlipNone)
            {
                img.RotateFlip(rotateFlip);
            }
            return img;
        }


        /// 调整图像大小以适应模型输入大小并保持纵横比。
        private Bitmap ResizeImage(Image image)
        {
            image = ApplyExifOrientation(image);
            PixelFormat format = image.PixelFormat;

            var output = new Bitmap(_model.Width, _model.Height, format);
            var (w, h) = (image.Width, image.Height); // 图像宽度和高度 w=4000,h=3000
            var (xRatio, yRatio) = (_model.Width / (float)w, _model.Height / (float)h); // xRatio=0.16 和 y=0.21 
            var ratio = Math.Min(xRatio, yRatio); // 比例 = 调整后尺寸 / 原始尺寸
            var (width, height) = ((int)(w * ratio), (int)(h * ratio)); // 区域的宽度和高度width=640,height=480
            var (x, y) = ((_model.Width / 2) - (width / 2), (_model.Height / 2) - (height / 2)); // 区域的 x=0 和 y=80 坐标
            var roi = new Rectangle(x, y, width, height); // 感兴趣的区域x=0, y=80,Width=640,Height=480

            using (var graphics = Graphics.FromImage(output))
            {
                graphics.Clear(Color.FromArgb(0, 0, 0, 0)); // 清除画布
                graphics.SmoothingMode = SmoothingMode.None; // 关闭平滑
                graphics.InterpolationMode = InterpolationMode.Bilinear; // 双线性插值
                graphics.PixelOffsetMode = PixelOffsetMode.Half; // 半像素偏移
                graphics.DrawImage(image, roi); // 绘制缩放后的图像
            }
            return output;
        }


        /// 从图像中提取像素并生成输入网络的张量。
        private Tensor<float> ExtractPixels(Image image)
        {
            var bitmap = (Bitmap)image;

            var rectangle = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            BitmapData bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, bitmap.PixelFormat);
            int bytesPerPixel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;

            var tensor = new DenseTensor<float>(new[] { 1, 3, _model.Height, _model.Width });

            unsafe // 通过直接操作内存加速转换
            {
                Parallel.For(0, bitmapData.Height, (y) =>
                {
                    byte* row = (byte*)bitmapData.Scan0 + (y * bitmapData.Stride);

                    Parallel.For(0, bitmapData.Width, (x) =>
                    {
                        tensor[0, 0, y, x] = row[x * bytesPerPixel + 2] / 255.0F; // 红色通道
                        tensor[0, 1, y, x] = row[x * bytesPerPixel + 1] / 255.0F; // 绿色通道
                        tensor[0, 2, y, x] = row[x * bytesPerPixel + 0] / 255.0F; // 蓝色通道
                    });
                });

                bitmap.UnlockBits(bitmapData);
            }

            return tensor;
        }


        /// 运行推理会话。
        private DenseTensor<float>[] Inference(Image image)
        {
            Bitmap resized = null;

            if (image.Width != _model.Width || image.Height != _model.Height)
            {
                resized = ResizeImage(image); // 将图像大小调整到640x640
            }

            var inputs = new List<NamedOnnxValue> // 将图像添加为ONNX输入
    {
        NamedOnnxValue.CreateFromTensor("images", ExtractPixels(resized ?? image))
    };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = _inferenceSession.Run(inputs); // 运行推理

            var output = new List<DenseTensor<float>>();

            foreach (var item in _model.Outputs) // 添加用于处理的输出
            {
                output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);
            };

            return output.ToArray();
        }



        /// 解析网络输出（detect）为预测结果。
        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height); // 图像的宽度和高度
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x 和 y 的增益
            var gain = Math.Min(xGain, yGain); // 增益 = 调整后尺寸 / 原始尺寸

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); // 左边和右边的填充

            Parallel.For(0, (int)output.Length / _model.Dimensions, (i) =>
            {
                if (output[0, i, 4] <= _model.Confidence) return; // 跳过低 obj_conf 的结果

                Parallel.For(5, _model.Dimensions, (j) =>
                {
                    output[0, i, j] = output[0, i, j] * output[0, i, 4]; // mul_conf = obj_conf * cls_conf
                });

                Parallel.For(5, _model.Dimensions, (k) =>
                {
                    if (output[0, i, k] <= _model.MulConfidence) return; // 跳过低 mul_conf 的结果

                    float xMin = ((output[0, i, 0] - output[0, i, 2] / 2) - xPad) / gain; // 取消填充 bbox 的左上角坐标到原始图像
                    float yMin = ((output[0, i, 1] - output[0, i, 3] / 2) - yPad) / gain; // 取消填充 bbox 的左上角坐标到原始图像
                    float xMax = ((output[0, i, 0] + output[0, i, 2] / 2) - xPad) / gain; // 取消填充 bbox 的右下角坐标到原始图像
                    float yMax = ((output[0, i, 1] + output[0, i, 3] / 2) - yPad) / gain; // 取消填充 bbox 的右下角坐标到原始图像

                    xMin = Clamp(xMin, 0, w - 0); // 将 bbox 左上角坐标裁剪到图像边界
                    yMin = Clamp(yMin, 0, h - 0); // 将 bbox 左上角坐标裁剪到图像边界
                    xMax = Clamp(xMax, 0, w - 1); // 将 bbox 右下角坐标裁剪到图像边界
                    yMax = Clamp(yMax, 0, h - 1); // 将 bbox 右下角坐标裁剪到图像边界

                    YoloLabel label = _model.Labels[k - 5];

                    var prediction = new YoloPrediction(label, output[0, i, k])
                    {
                        Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                    };

                    result.Add(prediction);
                });
            });

            return result.ToList();
        }


        /// 解析网络输出（Sigmoid）为预测结果。
        private List<YoloPrediction> ParseSigmoid(DenseTensor<float>[] output, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height); // 图像的宽度和高度
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x 和 y 的增益
            var gain = Math.Min(xGain, yGain); // 增益 = 调整后尺寸 / 原始尺寸

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); // 左边和右边的填充

            Parallel.For(0, output.Length, (i) => // 遍历模型输出
            {
                int shapes = _model.Shapes[i]; // 每个输出的形状

                Parallel.For(0, _model.Anchors[0].Length, (a) => // 遍历锚点
                {
                    Parallel.For(0, shapes, (y) => // 遍历形状（行）
                    {
                        Parallel.For(0, shapes, (x) => // 遍历形状（列）
                        {
                            int offset = (shapes * shapes * a + shapes * y + x) * _model.Dimensions;

                            float[] buffer = output[i].Skip(offset).Take(_model.Dimensions).Select(Sigmoid).ToArray();

                            if (buffer[4] <= _model.Confidence) return; // 跳过低置信度的结果

                            List<float> scores = buffer.Skip(5).Select(b => b * buffer[4]).ToList(); // mul_conf = obj_conf * cls_conf

                            float mulConfidence = scores.Max(); // 最大置信度分数

                            if (mulConfidence <= _model.MulConfidence) return; // 跳过低 mul_conf 的结果

                            float rawX = (buffer[0] * 2 - 0.5f + x) * _model.Strides[i]; // 预测的边界框 x（中心）
                            float rawY = (buffer[1] * 2 - 0.5f + y) * _model.Strides[i]; // 预测的边界框 y（中心）

                            float rawW = (float)Math.Pow(buffer[2] * 2, 2) * _model.Anchors[i][a][0]; // 预测的边界框宽度
                            float rawH = (float)Math.Pow(buffer[3] * 2, 2) * _model.Anchors[i][a][1]; // 预测的边界框高度

                            float[] xyxy = Xywh2xyxy(new float[] { rawX, rawY, rawW, rawH });

                            float xMin = Clamp((xyxy[0] - xPad) / gain, 0, w - 0); // 取消填充，裁剪左上角
                            float yMin = Clamp((xyxy[1] - yPad) / gain, 0, h - 0); // 取消填充，裁剪左上角
                            float xMax = Clamp((xyxy[2] - xPad) / gain, 0, w - 1); // 取消填充，裁剪右下角
                            float yMax = Clamp((xyxy[3] - yPad) / gain, 0, h - 1); // 取消填充，裁剪右下角

                            YoloLabel label = _model.Labels[scores.IndexOf(mulConfidence)];
                            var prediction = new YoloPrediction(label, mulConfidence)
                            {
                                Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                            };

                            result.Add(prediction);
                        });
                    });
                });
            });

            return result.ToList();
        }

        /// 将网络输出（sigmoid 或 detect 层）解析为预测。
        private List<YoloPrediction> ParseOutput(DenseTensor<float>[] output, Image image)
        {
            return _model.UseDetect ? ParseDetect(output[0], image) : ParseSigmoid(output, image);
        }

        /// 移除重叠的重复项（非极大值抑制，NMS）
        private List<YoloPrediction> Supress(List<YoloPrediction> items)
        {
            var result = new List<YoloPrediction>(items);

            foreach (var item in items) // 遍历每个预测
            {
                foreach (var current in result.ToList()) // 为每次迭代创建一个副本
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    RectangleF intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Area(); // 相交区域面积
                    float unionArea = rect1.Area() + rect2.Area() - intArea; // 联合区域面积
                    float overlap = intArea / unionArea; // 重叠比率

                    if (overlap >= _model.Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }
            return result;
        }

        /// 运行对象检测。

        public List<YoloPrediction> Predict(Image image)
        {
            var predictions = ParseOutput(Inference(image), image);

            // If no objects are detected, return an empty list
            if (predictions.Count == 0)
            {
                return new List<YoloPrediction>();
            }

            // Otherwise, perform non-maximum suppression and return the predictions
            return Supress(predictions);
        }


        /// 创建 YoloScorer 的新实例。
        public YoloScorer()
        {
            _model = Activator.CreateInstance<T>();
        }

        /// 使用权重路径和选项创建 YoloScorer 的新实例。使用CPU时候
        public YoloScorer(string weights, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(File.ReadAllBytes(weights), opts ?? new SessionOptions());
        }

        /// 使用GPU时候
        //public YoloScorer(string weights, SessionOptions opts = null) : this()
        //{
        //    var options = new SessionOptions();
        //    options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
        //    options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
        //    options.IntraOpNumThreads = 1;
        //    options.LogSeverityLevel = 0;
        //    options.EnableProfiling = false;
        //    options.AppendExecutionProvider_CUDA(0);
        //    _inferenceSession = new InferenceSession(File.ReadAllBytes(weights), options);
        //}


        /// 使用权重流和选项创建 YoloScorer 的新实例。
        public YoloScorer(Stream weights, SessionOptions opts = null) : this()
        {
            using (var reader = new BinaryReader(weights))
            {
                _inferenceSession = new InferenceSession(reader.ReadBytes((int)weights.Length), opts ?? new SessionOptions());
            }
        }

        /// 使用权重字节数组和选项创建 YoloScorer 的新实例。
        public YoloScorer(byte[] weights, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(weights, opts ?? new SessionOptions());
        }

        /// 释放 YoloScorer 实例。
        public void Dispose()
        {
            _inferenceSession.Dispose();
        }
    }
}
