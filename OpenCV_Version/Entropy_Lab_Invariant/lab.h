#pragma once
#include "header.h"

void LAB_Shadow(vector<Mat>& planes, Mat& dst) {
	normalize(planes[0], planes[0], 0, 100, NORM_MINMAX, CV_8UC1);	
	// L 범위를 0 ~ 100으로 정규화
	double mean_L = mean(planes[0])[0];
	double mean_A = mean(planes[1])[0];
	double mean_B = mean(planes[2])[0];

	double sum_dev = 0;

	for (int j = 0; j < planes[0].rows; j++) {
		for (int i = 0; i < planes[0].cols; i++) {
			sum_dev += pow(planes[0].at<uchar>(j, i) - mean_L, 2);
		}
	}

	double sigma = sqrtf(sum_dev / planes[0].total());
	dst = Mat::zeros(planes[0].size(), CV_8UC1);
	
	for (int j = 0; j < dst.rows; j++) {
		for (int i = 0; i < dst.cols; i++) {
			if (mean_A + mean_B <= 256) {
				if (planes[0].at<uchar>(j, i) <= (mean_L - sigma / 3)) {
					dst.at<uchar>(j, i) = 255;
				}
			}
			else {
				if (planes[0].at<uchar>(j, i) + planes[2].at<uchar>(j, i) < 150) {
					dst.at<uchar>(j, i) = 255;
				}
			}
		}
	}
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(dst, dst, MORPH_OPEN, kernel);
	imshow("dst", dst);
}