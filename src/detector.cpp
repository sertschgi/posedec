// standard-tools
#include <iostream>
#include <string>

// header
#include "posedec/detector.hpp"

// image-tools
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp> // to load the model

// parser
#include <cxxopts.hpp>



void utils::parser::addDetOptions(cxxopts::Options& cxxoption)
{
    cxxoption.add_options()
        ("s,stream", "Show the camera-stream in a window")
        ("c,checkpoint", "Path of the model checkpoint (.tflite)", cxxopts::value<std::string>()->default_value("detect.tflite"))
        ("l,labelmap", "Path of the labelmap (.pbtxt)", cxxopts::value<std::string>()->default_value("labelmap.pbtxt"))
        ("t,threshold", "Threshold of when to detect", cxxopts::value<float>()->default_value("0.7"))
        
        ("r,resolution", "Resolution of the camera (HeightxWidth)", cxxopts::value<std::string>()->default_value("1280x720"))
        ("f,framerate", "Framerate of the camera", cxxopts::value<int>()->default_value("30"))
        ("o,orientation", "The orientation of the frame (0,1,2,3)", cxxopts::value<int>()->default_value("0"));
}

// int utils::parser::parseResolution(std::string resolution)
// {
    
// }

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
    ) 
    {
        return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
            std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
            "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
            std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
    }  

}


Detector::Detector
(
    std::string checkpoint_path,
    std::string labelmap_path,
    int width,
    int height
)
{
    cv::dnn::Net net {cv::dnn::readNetFromTfLite(checkpoint_path)};
    const int WIDTH {width};
    const int HEIGHT {height};
    cv::Mat detection{};
}

cv::Mat Detector::detect(cv::Mat frame)
{
    std::string inputLayerName {this->net.getLayerNames()[0]};
    std::string outputLayerName {this->net.getLayerNames()[1]};
    cv::Mat blob
    {
        cv::dnn::blobFromImage (
            frame, 
            1.0, 
            cv::Size(this->WIDTH), 
            cv::Size(this->HEIGHT), 
            cv::Scalar(127.5, 127.5, 127.5), 
            false
        )
    };
    this->net.setInput(blob);
    cv::Mat detection {this->net.forward()};
    return detection&
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("detector");
    utils::parser::addDetOptions(options);
    cxxopts::ParseResult result{};

    const bool CAMERA_WINDOW {result.count("stream") > 0};
    const std::string CHECKPOINT_PATH {result["checkpoint"].as<std::string>()};
    const std::string LABELMAP_PATH {result["labelmap"].as<std::string>()};
    const float CONFIDENCE_THRESHOLD {result["threshold"].as<float>()};
    const char* CAMERA_RESOLUTION {result["resolution"].as<const char*>()};
    const size_t X_POS {std::strchr(CAMERA_RESOLUTION, 'x') - CAMERA_RESOLUTION;}
    const int CAMERA_WIDTH {std::atoi(CAMERA_RESOLUTION)};
    const int CAMERA_HEIGHT {std::atoi(CAMERA_RESOLUTION + X_POS + 1)};
    const int CAMERA_FRAMERATE {result["framerate"].as<int>()};
    const int CAMERA_ORIENTATION {result["orientation"].as<int>()};

    const std::string GST_PIPELINE 
    {
        utils::stream.gstreamer_pipeline (
            CAMERA_WIDTH, 
            CAMERA_HEIGHT, 
            CAMERA_WIDTH, 
            CAMERA_HEIGHT, 
            CAMERA_FRAMERATE, 
            CAMERA_ORIENTATION
        )
    };

    cv::VideoCapture cap(GST_PIPELINE, cv::CAP_GSTREAMER);

    Detector det(CHECKPOINT_PATH, LABELMAP_PATH, CAMERA_WIDTH, CAMERA_HEIGHT);

    if (!cap.isOpened()) 
    {
	    std::cout<<"[Error] Failed to open camera."<< std::endl;
	    return (-1);
    }
        
    cv::namedWindow("Detector", cv::WINDOW_AUTOSIZE);

    cv::Mat img {};
    cv::Mat result {};

    std::cout << "[Info] Hit ESC to exit" << std::endl;

    while (true)
    {
    	if (!cap.read(img)) 
        {
            std::cout<<"[Error] Capture read error" << std::endl;
            break;
	    }

        result = det.detect(img);
	
	    cv::imshow("Detector", img);
	    
        int keycode = cv::waitKey(10) & 0xff; 
    
        if (keycode == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}