#pragma once

# include<fstream>
#include<iostream>
#include<sstream>
#include<opencv2/opencv.hpp>
#include"NvInfer.h"

using namespace nvinfer1;
using namespace cv;

struct DetectResult {
	int classId;
	float score;
	cv::Rect box;
};

class YOLOv5TRTDetector {
public:
	void initConfig(std::string enginefile, float conf_threshold, float score_threshold);
	void detect(cv::Mat& frame, std::vector<DetectResult>& result);
	~YOLOv5TRTDetector();
private:
	float conf_threshold = 0.25;
	float score_threshold = 0.25;
	int input_w = 640;
	int input_h = 640;
	int output_h;
	int output_w;
	IRuntime* runtime{ nullptr };
	ICudaEngine* engine{ nullptr };
	IExecutionContext* context{ nullptr };
	void* buffers[2] = { NULL, NULL };
	std::vector<float>prob;
	cudaStream_t stream;
};