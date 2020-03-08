#include "opencv.hpp" 
#include <opencv2/core/core.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp> 
#include "stdafx.h"
#include <windows.h>
#include <windowsx.h>
#include <vector>
#include <stdlib.h> 
#include <iostream> 
#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>
#include <math.h>
//#include <tensorrt_provider_factory.h>
//#include <mkldnn_provider_factory.h>

using namespace cv;
using namespace std;
std::array<float, 1 * 1 * 28 * 28> results_{};
std::array<float, 800 * 1880 * 3> input_image_{};
std::array<float, 3> mean_vec = { 102.9801, 115.9465, 122.7717 };
Ort::Value input_tensor_{ nullptr };
std::array<int64_t, 3> input_shape_{ 3, 800, 1880 };
Ort::Value output_tensor_{ nullptr };
//std::array<int64_t, 4> output_shape_{78,1,28, 28 };
int main(int argc, char* argv[]) {
	//*************************************************************************
	// initialize  enviroment...one enviroment per process
	// enviroment maintains thread pools and other state info

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	// initialize session options if needed

	Ort::SessionOptions session_options;

	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	//*************************************************************************

#ifdef _WIN32
	const wchar_t* model_path = L"D://onnxruntime-projects-master//test_models//mask_rcnn_R_50_FPN_1x.onnx";
#else
	const char* model_path = "D://squeezenet.onnx";
#endif
	printf("Using Onnxruntime C++ API\n");

	Ort::Session session(env, model_path, session_options);
  
	Ort::AllocatorWithDefaultOptions allocator;
	Mat img = imread("C:\\Users\\lcm\\Downloads\\maskrcnn-benchmark-onnx_stage\\maskrcnn-benchmark-onnx_stage\\demo\\frcnn_demo.jpg");
	Mat dst;
	float ratio = 800.0 / min(img.cols, img.rows);
	const int re_width = int(ratio * img.cols);  
	const int re_height = int(ratio * img.rows);
	cout << re_width << '\n' << re_height << endl;
	cv::resize(img, dst, cv::Size(re_width, re_height));
	cv::Mat bgrimg, bgrimg_f;
	cvtColor(dst, bgrimg, cv::COLOR_RGB2BGR);
	bgrimg.convertTo(bgrimg_f, CV_32FC3);
	for (int i = 0; i < bgrimg_f.rows; i++) {
		for (int j = 0; j < bgrimg_f.cols; j++) {
			bgrimg_f.at<Vec3f>(i, j)[0] -= mean_vec[0];
			bgrimg_f.at<Vec3f>(i, j)[1] -= mean_vec[1];
			bgrimg_f.at<Vec3f>(i, j)[2] -= mean_vec[2];
		}
	}
	const int padd_h = int(ceil(bgrimg_f.rows / 32) * 32);
	const int padd_w = int(ceil(bgrimg_f.cols / 32) * 32);
	cv::Mat paddimg;
	cv::resize(bgrimg_f, paddimg, cv::Size(padd_w, padd_h));
	// 3D matrix transpose HWC -> CHW
	float* output = input_image_.data();
	fill(input_image_.begin(), input_image_.end(), 0.f);
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < padd_h; i++) {
			for (int j = 0; j < padd_w; j++) {
				output[c * padd_w * padd_h + i * padd_w + j] = (paddimg.ptr<float>(i)[j * 3 + c]);
			}
		}
	}
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	//Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
	input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	//output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
	const char* input_names[] = { "image" };
	const char* output_names[] = {"6887" }; //only output the masks
	//OrtValue* input_tensor_1 = input_tensor_;
	//OrtValue* output_tensor_1 = output_tensor_;
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, 
		input_node_names.data(),
		&input_tensor_,
		1,
		output_names,
		1);
    return 0;
    }
