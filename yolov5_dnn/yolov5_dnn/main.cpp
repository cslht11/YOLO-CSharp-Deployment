#include "yolov5_dnn.h"
#include <iostream>
#include <fstream>

std::string label_map = "D:/A_Python/yolov5-6.1/classes.txt";
int main(int argc, char** argv) {
	/*std::string names = "10:bike";
	int pos = names.find_first_of(":");
	std::cout << names.substr(0, pos) << " -->> " << names.substr(pos+1) << std::endl;*/
	std::vector<std::string> classNames;
	std::ifstream fp(label_map);
	std::string name;
	while (!fp.eof()) {
		getline(fp, name);
		if (name.length()) {
			classNames.push_back(name);
		}
	}
	fp.close();

	std::shared_ptr<YOLOv5DNNDetector> detector(new YOLOv5DNNDetector());
	detector->initConfig("D:/A_Python/yolov5-6.1/runs/train/exp/weights/best.onnx", 640, 640, 0.25f);
	std::vector<DetectResult> results;
	cv::Mat frame = cv::imread("D:/A_Python/yolov5-6.1/imgs/Image0.jpg");
	detector->detect(frame, results);
	for (DetectResult dr : results) {
		cv::Rect box = dr.box;
		cv::putText(frame, classNames[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
	}
	cv::imshow("YOLOv5-6.1 + OpenCV DNN - by gloomyfish", frame);
	cv::waitKey(0);
	cv::destroyAllWindows();

	cv::VideoCapture capture("D:/A_Python/yolov5-6.1/imgs/video.mp4");
	//cv::Mat frame;
	while (true) {
		bool ret = capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame, results);
		for (DetectResult dr : results) {
			cv::Rect box = dr.box;
			cv::putText(frame, classNames[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
		}
		cv::imshow("YOLOv5-6.1 + OpenCV DNN - by gloomyfish", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}

		results.clear();
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}