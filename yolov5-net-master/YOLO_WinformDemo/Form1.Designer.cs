namespace YOLO_WinformDemo
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            btn_selectFile = new Button();
            picbox_Display = new PictureBox();
            label1 = new Label();
            label2 = new Label();
            open_camera = new Button();
            ((System.ComponentModel.ISupportInitialize)picbox_Display).BeginInit();
            SuspendLayout();
            // 
            // btn_selectFile
            // 
            btn_selectFile.Location = new Point(15, 14);
            btn_selectFile.Margin = new Padding(4);
            btn_selectFile.Name = "btn_selectFile";
            btn_selectFile.Size = new Size(120, 40);
            btn_selectFile.TabIndex = 0;
            btn_selectFile.Text = "选择文件";
            btn_selectFile.UseVisualStyleBackColor = true;
            btn_selectFile.Click += btn_selectFile_Click;
            // 
            // picbox_Display
            // 
            picbox_Display.BackgroundImageLayout = ImageLayout.Zoom;
            picbox_Display.Dock = DockStyle.Bottom;
            picbox_Display.Location = new Point(0, 77);
            picbox_Display.Margin = new Padding(4);
            picbox_Display.Name = "picbox_Display";
            picbox_Display.Size = new Size(1029, 544);
            picbox_Display.TabIndex = 1;
            picbox_Display.TabStop = false;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(375, 25);
            label1.Margin = new Padding(4, 0, 4, 0);
            label1.Name = "label1";
            label1.Size = new Size(54, 20);
            label1.TabIndex = 2;
            label1.Text = "耗时：";
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(495, 26);
            label2.Name = "label2";
            label2.Size = new Size(39, 20);
            label2.TabIndex = 3;
            label2.Text = "FPS:";
            // 
            // open_camera
            // 
            open_camera.Location = new Point(175, 14);
            open_camera.Name = "open_camera";
            open_camera.Size = new Size(120, 40);
            open_camera.TabIndex = 4;
            open_camera.Text = "打开相机";
            open_camera.UseVisualStyleBackColor = true;
            open_camera.Click += open_camera_Click;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(9F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(1029, 621);
            Controls.Add(open_camera);
            Controls.Add(label2);
            Controls.Add(label1);
            Controls.Add(picbox_Display);
            Controls.Add(btn_selectFile);
            Margin = new Padding(4);
            Name = "Form1";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)picbox_Display).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button btn_selectFile;
        private PictureBox picbox_Display;
        private Label label1;
        private Label label2;
        private Button open_camera;
    }
}