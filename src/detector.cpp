// standard-tools
#include <iostream>
#include <string>
#include <cstring>

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
        ("h,help", "Show help")
        ("s,stream", "Show the camera-stream in a window")
        ("c,checkpoint", "Path of the model checkpoint (.tflite)", cxxopts::value<std::string>()->default_value("detect.tflite"))
        ("l,labelmap", "Path of the labelmap (.pbtxt)", cxxopts::value<std::string>()->default_value("labelmap.pbtxt"))
        ("t,threshold", "Threshold of when to detect", cxxopts::value<float>()->default_value("0.7"))
        ("r,resolution", "Resolution of the camera (HeightxWidth)", cxxopts::value<std::string>()->default_value("1280x720"))
        ("f,framerate", "Framerate of the camera", cxxopts::value<int>()->default_value("30"))
        ("o,orientation", "The orientation of the frame (0,1,2,3)", cxxopts::value<int>()->default_value("0"));
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
    this->net = cv::dnn::readNetFromTensorflow(checkpoint_path);
    this->WIDTH = width;
    this->HEIGHT = height;
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
            cv::Size(this->WIDTH, this->HEIGHT), 
            cv::Scalar(127.5, 127.5, 127.5),
            false,
            false
        )
    };
    this->net.setInput(blob);
    cv::Mat detection {this->net.forward()};
    return detection;
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("detector");
    utils::parser::addDetOptions(options);
    cxxopts::ParseResult result {};

    try {
        result = options.parse(argc, argv);
    }
    catch (const cxxopts::exceptions::exception e) {
        std::cerr << "Error parsing command-line arguments: " << e.what() << std::endl
            << "usage: posedec --option" << std::endl
            << "For help use -h or --help" << std::endl;
        exit(1);
    }

    if (result.count("help") > 0)
    {
        std::cout << "Help: " << std::endl << options.help() << std::endl;
        return 0;
    }

    const bool CAMERA_WINDOW = result.count("stream") > 0;
    const std::string CHECKPOINT_PATH = result["checkpoint"].as<std::string>();
    const std::string LABELMAP_PATH = result["labelmap"].as<std::string>();
    const float CONFIDENCE_THRESHOLD = result["threshold"].as<float>();
    const std::string CAMERA_RESOLUTION_STR = result["resolution"].as<std::string>();
    const size_t X_POS = CAMERA_RESOLUTION_STR.find('x');
    const int CAMERA_WIDTH = std::stoi(CAMERA_RESOLUTION_STR.substr(0, X_POS));
    const int CAMERA_HEIGHT = std::stoi(CAMERA_RESOLUTION_STR.substr(X_POS + 1));
    const int CAMERA_FRAMERATE = result["framerate"].as<int>();
    const int CAMERA_ORIENTATION = result["orientation"].as<int>();

    std::cout << "[Info] Options specified: " << std::endl 
        << "Camera window: " << CAMERA_WINDOW << std::endl 
        << "Checkpoint path: " << CHECKPOINT_PATH << std::endl 
        << "Labelmap path: " << LABELMAP_PATH << std::endl 
        << "Confidence threshold: " << CONFIDENCE_THRESHOLD << std::endl 
        << "Camera width: " << CAMERA_WIDTH << std::endl 
        << "Camera height: " << CAMERA_HEIGHT << std::endl 
        << "Camera framerate: " << CAMERA_FRAMERATE << std::endl 
        << "Camera orientation: " << CAMERA_ORIENTATION << std::endl;

    const std::string GST_PIPELINE 
    {
        utils::stream::gstreamer_pipeline (
            CAMERA_WIDTH, 
            CAMERA_HEIGHT, 
            CAMERA_WIDTH, 
            CAMERA_HEIGHT, 
            CAMERA_FRAMERATE, 
            CAMERA_ORIENTATION
        )
    };

    cv::VideoCapture cap(GST_PIPELINE, cv::CAP_GSTREAMER);

    Detector det {CHECKPOINT_PATH, LABELMAP_PATH, CAMERA_WIDTH, CAMERA_HEIGHT};

    if (!cap.isOpened()) 
    {
        std::cerr << "[Error] Failed to open camera."<< std::endl;
        return (-1);
    }
        
    cv::namedWindow("Detection Result", cv::WINDOW_AUTOSIZE);

    cv::Mat img {};
    //cv::Mat result {};

    std::cout << "[Info] Hit ESC to exit" << std::endl;

    while (true)
    {
        if (!cap.read(img)) 
        {
            std::cout << "[Error] Capture read error" << std::endl;
            break;
        }

        auto result = det.detect(img);

        std::cout << "[Info] Detection:" << result << std::endl;
    
        cv::imshow("Detection Result", img);

        int keycode = cv::waitKey(10) & 0xff; 
    
        if (keycode == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
