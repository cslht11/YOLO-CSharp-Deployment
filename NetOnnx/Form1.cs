using System.Drawing.Drawing2D;
using Yolov8Net;
namespace NetOnnx
{
    public partial class Form1 : Form
    {
        private IPredictor yolov8=null;
        string[] mylabel = {"dog","songsu" };//模型分类标签
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            //加载模型
            try
            {
                yolov8 = YoloV8Predictor.Create("best.onnx", mylabel);
                if(yolov8 != null )
                {
                    richTextBox1.Text = ("模型加载成功\r\n");
                }
            }
            catch(System.Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            //dialog.Multiselect = true;//该值确定是否可以选择多个文件
            dialog.Title = "请选择文件";
            dialog.Filter = "图像文件(*.jpg;*.jpeg)|*.jpg;*.jpeg";
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                string file = dialog.FileName;
                textBox1.Text = file;

                showimg(dialog);
            }
        }

        private void showimg(OpenFileDialog dialog)
        {
            string imgpath = dialog.FileName;

            Image image = System.Drawing.Image.FromFile(imgpath);
            Form form = new Form();
            form.Text = dialog.SafeFileName + " " + image.Width + "x" + image.Height;

            PictureBox pictureBox = new PictureBox();
            pictureBox.Parent = form;
            //图像预测
            var predictions = yolov8.Predict(image);
            // Draw your boxes
            //using var graphics = Graphics.FromImage(image);
            foreach (var pred in predictions)
            {
                var originalImageHeight = image.Height;
                var originalImageWidth = image.Width;

                var x = Math.Max(pred.Rectangle.X, 0);
                var y = Math.Max(pred.Rectangle.Y, 0);
                var width = Math.Min(originalImageWidth - x, pred.Rectangle.Width);
                var height = Math.Min(originalImageHeight - y, pred.Rectangle.Height);

                ////////////////////////////////////////////////////////////////////////////////////////////
                // *** Note that the output is already scaled to the original image height and width. ***
                ////////////////////////////////////////////////////////////////////////////////////////////

                // Bounding Box Text
                string text = $"{pred.Label.Name} [{pred.Score}]";

                using (Graphics graphics = Graphics.FromImage(image))
                {
                    graphics.CompositingQuality = CompositingQuality.HighQuality;
                    graphics.SmoothingMode = SmoothingMode.HighQuality;
                    graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    // Define Text Options
                    Font drawFont = new Font("consolas", 11, FontStyle.Regular);
                    SizeF size = graphics.MeasureString(text, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                    // Define BoundingBox options
                    Pen pen = new Pen(Color.Yellow, 2.0f);
                    SolidBrush colorBrush = new SolidBrush(Color.Yellow);

                    // Draw text on image 
                    graphics.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    graphics.DrawString(text, drawFont, fontBrush, atPoint);

                    // Draw bounding box on image
                    graphics.DrawRectangle(pen, x, y, width, height);
                }
            }

            pictureBox.Width = image.Width;
            pictureBox.Height = image.Height;
            pictureBox.Image = (Bitmap)image.Clone();

            form.AutoSize = true;
            form.Show();
            image.Dispose();

        }
    }
}