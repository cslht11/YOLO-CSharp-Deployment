using OpenCvSharp;
using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models;
using Point = OpenCvSharp.Point;
using System.IO;
using OpenCvSharp.Extensions;

namespace YOLO_WinformDemo
{
    public partial class Form1 : Form
    {
        private YoloScorer<YoloCocoP6Model> scorer;
        private string filePath = "";
        private VideoCapture frame;
        private CancellationTokenSource cancellationTokenSource;

        public Form1()
        {
            InitializeComponent();
            scorer = new YoloScorer<YoloCocoP6Model>(Application.StartupPath + "\\onnx\\best_zsm.onnx");
        }

        private async void btn_selectFile_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFile = new OpenFileDialog();
            openFile.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tif;|Video Files|*.avi;*.mp4;*.mov;*.mkv;|All Files|*.*";
            if (openFile.ShowDialog() == DialogResult.OK)
            {
                filePath = openFile.FileName;

                if (Path.GetExtension(filePath) == ".jpg" || Path.GetExtension(filePath) == ".jpeg" || Path.GetExtension(filePath) == ".png" || Path.GetExtension(filePath) == ".bmp" || Path.GetExtension(filePath) == ".gif" || Path.GetExtension(filePath) == ".tif")
                {
                    using var image = Image.FromFile(openFile.FileName);
                    picbox_Display.BackgroundImage = await AddInfoToImageAsync(image);
                }
                else
                {
                    if (cancellationTokenSource != null)
                    {
                        cancellationTokenSource.Cancel(); // 取消之前的检测任务
                    }

                    cancellationTokenSource = new CancellationTokenSource();
                    frame = new VideoCapture(filePath);

                    // 启动视频检测任务
                    await Task.Run(() => DetectVideoAsync(cancellationTokenSource.Token), cancellationTokenSource.Token);
                }
            }
        }


        private bool isCameraOpen = false; // 用于跟踪摄像头状态

        private void UpdateButtonLabel()
        {
            if (isCameraOpen)
            {
                open_camera.Text = "关闭摄像头";
            }
            else
            {
                open_camera.Text = "开启摄像头";
            }
        }
        private async void open_camera_Click(object sender, EventArgs e)
        {
            if (isCameraOpen)
            {
                // 如果摄像头已经打开，则关闭摄像头
                CloseCamera();
            }
            else
            {
                // 如果摄像头未打开，则打开摄像头
                OpenCamera();
            }
        }

        private void OpenCamera()
        {
            if (cancellationTokenSource != null)
            {
                cancellationTokenSource.Cancel(); // 取消之前的检测任务
            }

            cancellationTokenSource = new CancellationTokenSource();

            // 打开摄像头
            frame = new VideoCapture(0); // 0 表示默认摄像头，如果有多个摄像头，可以尝试不同的索引

            // 启动摄像头检测任务
            Task.Run(() => DetectVideoAsync(cancellationTokenSource.Token), cancellationTokenSource.Token);

            isCameraOpen = true;
            UpdateButtonLabel();//更新按钮文本
        }

        private void CloseCamera()
        {
            if (cancellationTokenSource != null)
            {
                cancellationTokenSource.Cancel(); // 取消检测任务
            }

            // 释放摄像头资源
            if (frame != null)
            {
                frame.Dispose();
            }

            isCameraOpen = false;
            UpdateButtonLabel();//更新按钮文本
        }



        private async Task DetectVideoAsync(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                Mat mat = new Mat();
                frame.Read(mat);
                if (mat.Empty())
                {
                    break;
                }
                var image = BitmapConverter.ToBitmap(mat);
                picbox_Display.BackgroundImage = await AddInfoToImageAsync(image);
            }
        }

        private async Task<Image> AddInfoToImageAsync(Image inputImage)
        {
            DateTime start = DateTime.Now;
            List<YoloPrediction> predictions = scorer.Predict(inputImage);

            if (predictions.Count == 0)
            {
                return (Image)inputImage.Clone();
            }
            else
            {
                Mat inputMat = BitmapConverter.ToMat((Bitmap)inputImage);
                foreach (YoloPrediction prediction in predictions)
                {
                    Point p1 = new Point(prediction.Rectangle.X, prediction.Rectangle.Y);
                    Point p2 = new Point(prediction.Rectangle.X + prediction.Rectangle.Width, prediction.Rectangle.Y + prediction.Rectangle.Height);
                    Point p3 = new Point(prediction.Rectangle.X - 130, prediction.Rectangle.Y - 30);
                    Scalar scalar0 = new Scalar(0, 0, 255);
                    Scalar scalar1 = new Scalar(0, 255, 0);
                    Cv2.Rectangle(inputMat, p1, p2, scalar0, 4);

                    // 在图像上绘制文本
                    Cv2.PutText(inputMat, prediction.Label.Name + " " + Math.Round(prediction.Score, 2), p3, HersheyFonts.HersheyDuplex, 3, scalar1, 3);
                }

                DateTime end = DateTime.Now;
                var fps = 1000 / (end - start).TotalMilliseconds;
                string timeText = "Time:" + (end - start).TotalMilliseconds.ToString("F2");

                // 在图像上绘制耗时和FPS信息
                Cv2.PutText(inputMat, timeText, new Point(10, 80), HersheyFonts.HersheyDuplex, 3, Scalar.Yellow, 3);
                Cv2.PutText(inputMat, $"FPS: {fps:F2}", new Point(10, 220), HersheyFonts.HersheyDuplex, 3, Scalar.Yellow, 3);
                return BitmapConverter.ToBitmap(inputMat);
            }
        }

    }
}
