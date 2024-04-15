#ifndef ROTATE_INCLUDE_ROTATED_H_
#define ROTATE_INCLUDE_ROTATED_H_
#include <memory>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <dirent.h>
#include "cuda_runtime_api.h"
#include "NvInfer.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace nvinfer1;

#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"



class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class Rotatation {
public:
    void loadEngine(const std::string& path);
    cv::Mat preprocessImg(cv::Mat& img, const int& input_w, const int& input_h, int& padw, int& padh);
    std::vector<std::string> listJpgFiles(const std::string& directory) ;
    void totalInference(const std::string& directory);

    void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color);
    void rotateInference(cv::Mat & img, json & j);

    void initDetection();
    yolo::Image cvimg(const cv::Mat &image) ;
    void single_inference();

    void run();
  
private:
    Logger logger_;
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;

    float CONF_THRESHOLD = 0.4;
    float NMS_THRESHOLD = 0.25;

    const int BATCH_SIZE = 1;
    const int CHANNELS = 3;
    const int INPUT_H = 1024;
    const int INPUT_W = 1024;

    const int OUTPUT0_BOXES = 21504;
    const int OUTPUT0_ELEMENT = 7;
    const int CLASSES = 2;

    const char* images_ = "images";
    const char* output0_ = "output0";

    std::vector<std::string> class_names_rot{
        "sleeper_normal","sleeper_abnormal" 
    };


    std::string obb_path_ = "/home/ubuntu/wjd/uav/model/sleeper_obb.engine";

    std::string detection_path = "/home/ubuntu/wjd/uav/model/rail_fastener.engine";
    std::shared_ptr<yolo::Infer> yolo_;

    std::string dji_img_path = "/home/ubuntu/wjd/dajiang";

    std::vector<std::string> class_names_det{
        "fastener_normal","fastener_abnormal","fastener_stone","fastener_missing",
        "rail_big"
    };


};


#endif //ROTATE_INCLUDE_ROTATED_H_