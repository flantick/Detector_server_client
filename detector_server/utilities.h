#ifndef utilities_H
#define utilities_H

#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <torch/script.h>
#include "torch/torch.h"
#include <iomanip>
#include "box.h"

using namespace std;


cv::Mat coverImg(cv::Mat& img, cv::Size trgSize);

float calculateIoU(const Box& box1, const Box& box2);

vector<Box> nms(const vector<Box>& boxes, float iouThres);

vector<Box> getBoxes(at::Tensor& outputs, float confThres, float iouThres);

void highlightBoxes(cv::Mat& img, vector<Box>& boxes);

cv::Mat detect(
    torch::jit::script::Module& model,
    cv::Mat img,
    torch::Device& device,
    int imgMaxWidth
);
#endif