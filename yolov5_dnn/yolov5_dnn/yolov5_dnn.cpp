#include "yolov5_dnn.h"

void YOLOv5DNNDetector::initConfig(std::string onnxpath, int iw, int ih, float threshold) {
	this->input_w = iw;
	this->input_h = ih;
	this->threshold_score = threshold;
	this->net = cv::dnn::readNetFromONNX(onnxpath);
	//this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	//this->net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);

	this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

void YOLOv5DNNDetector::detect(cv::Mat& frame, std::vector<DetectResult>& results) {
	int64 start = cv::getTickCount();
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));

	float x_factor = image.cols / 640.0f;
	float y_factor = image.rows / 640.0f;

	cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	cv::Mat preds = this->net.forward();

	//  1x25200x85
	// std::cout << "rows: "<< preds.size[1]<< " data: " << preds.size[2] << std::endl;
	cv::Mat det_output(preds.size[1], preds.size[2], CV_32F, preds.ptr<float>());
	float confidence_threshold = 0.5;
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	for (int i = 0; i < det_output.rows; i++) {
		float confidence = det_output.at<float>(i, 4);
		if (confidence < 0.45) {
			continue;
		}
		cv::Mat classes_scores = det_output.row(i).colRange(5, preds.size[2]);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		//std::cout << score << std::endl;  //得分等信息全为0，读取不到东西
		
		if (score > this->threshold_score)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			// printf("cx:%.2f, cy:%.2f, ow:%.2f, oh:%.2f, x_factor:%.2f, y_factor:%.2f \n", cx, cy, ow, oh, x_factor, y_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		DetectResult dr;
		int index = indexes[i];
		int idx = classIds[index];
		dr.box = boxes[index];
		dr.classId = idx;
		dr.score = confidences[index];
		cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
		cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
			cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
		results.push_back(dr);
	}

	std::ostringstream ss;
	std::vector<double> layersTimings;
	//计算FPS
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

}