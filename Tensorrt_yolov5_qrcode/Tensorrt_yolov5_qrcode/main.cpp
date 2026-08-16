#include "yolov5_trt.h"

std::string label_txt_file = "D:/A_Python/yolov5-6.1_qrcode/classes.txt";
std::vector<std::string> readClassNames();
std::vector<std::string> readClassNames() 
{
	std::vector<std::string> classNames;

	std::ifstream fp(label_txt_file);
	if (!fp.is_open()) 
	{
		printf("could not open file...\n");
		exit(-1);
		}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
	}

int main(int argc, char** argv) {
	std::vector<std::string> labels = readClassNames();
	std::string enginefile = "D:/A_Python/yolov5-6.1_qrcode/runs/train/exp/weights/best.engine";
	cv::VideoCapture cap("D:/A_Python/yolov5-6.1_qrcode/imgs/video1.mp4");
	cv::Mat frame;
	auto detector = std::make_shared<YOLOv5TRTDetector>();
	detector->initConfig(enginefile, 0.25, 0.25);
	std::vector<DetectResult> results;

	while (true) {
		bool ret = cap.read(frame);
		if (frame.empty()) {
			break;
		}
		
		detector->detect(frame, results);
	
		for (DetectResult dr : results) {
			std::cout << "11111111111111111111111111111111111" << std::endl;
			cv::Rect box = dr.box;
			cv::putText(frame, labels[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
		}
		cv::imshow("YOLOv5 + Tensorrt8.6", frame);
		char c = cv::waitKey(1);
		if (c == 27) { //ESCÍË³ö
			break;
		}
		results.clear();
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}