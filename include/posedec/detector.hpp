// standard-tools
#include <iostream>
#include <string>

// image-tools
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp> // to load the model

// parser
#include <cxxopts.hpp>

namespace utils{}
namespace utils::parser
{
    void addDetOptions(cxxopts::Options&);
}
namespace utils::stream
{
    std::string gstreamer_pipeline
    (
        int capture_width, 
        int capture_height, 
        int display_width, 
        int display_height, 
        int framerate, 
        int flip_method
    );
}

class Detector
{
    Detector(std::string, std::string, int, int);
    cv::Mat detect(cv::Mat);
};