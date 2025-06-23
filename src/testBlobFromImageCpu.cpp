//
// Created by Administrator on 2025/6/23.
//
#include "include/testBlobFromImageCpu.h"

int testBlobFromImage(){
    #ifdef _WIN32
        // Windows 环境：设置控制台输出编码为 UTF-8
        SetConsoleOutputCP(CP_UTF8);
        // 可选：设置输入编码也为 UTF-8
        SetConsoleCP(CP_UTF8);
    #endif
    cv::setNumThreads(0);  // 使用所有CPU核心

    cv::Mat img = cv::imread("../asserts/bus.jpg");
    if (img.empty()) {
        cerr << "Image not loaded" << endl;
        return -1;
    }

    double scale = 1.0 / 255.0;
    cv::Size size(640, 640);
    cv::Scalar mean(0, 0, 0);

    // 预热一次
    cv::dnn::blobFromImage(img, scale, size, mean, true, true);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {

        cv::Mat blob = cv::dnn::blobFromImage(img, 1 / 255.0, cv::Size(640,640), mean, true, false);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double total_time_ms = duration.count();
    cout << "C++ CPU BlobFrom Image 1000次总耗时: " << total_time_ms << " ms" << endl;
    cout << "C++ CPU BlobFrom Image 单次平均耗时: " << total_time_ms / 10000.0 << " ms" << endl;
    return 0;
}