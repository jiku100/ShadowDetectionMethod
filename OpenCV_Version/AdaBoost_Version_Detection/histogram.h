#pragma once
#include "source.h"


Mat _calcHist(const Mat& img) {
	Mat hist;
	int channels[] = { 0 };	// 히스토그램 계산 채널
	int dims = 1;			// 출력 영상 차원
	const int histSize[] = { 256 };	// 각 차원의 히스토그램 빈 개수
	float graylevel[] = { 0,256 };	// GrayScale의 최솟값, 최댓값
	const float* ranges[] = { graylevel }; // 각 차원의 히스토그램 범위
	calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges);
	return hist;
}
Mat getHistImage(const Mat& hist) {
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));
	double histMax;
	minMaxLoc(hist, 0, &histMax);
	Mat imgHist(100, 256, CV_8UC3, Scalar(255, 255, 255));
	vector<Point> lines;
	for (int i = 0; i < 256; i++) {
		lines.push_back(Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)));
	}
	polylines(imgHist, lines, false, Scalar(0, 0, 0), 2, LINE_AA);
	return imgHist;
}

double findVally(const Mat& hist) {
	int firstPeak = 0;
	for (int i = 3; i < 250; i++) {
		if (i > 10 && firstPeak == 0) {
			firstPeak = 1;
		}
		double before = (hist.at<float>(i - 2, 0) + hist.at<float>(i - 1, 0) + hist.at<float>(i - 3, 0)) / 3;
		double after = (hist.at<float>(i + 2, 0) + hist.at<float>(i + 1, 0) + hist.at<float>(i + 3, 0)) / 3;
		double present = hist.at<float>(i, 0);
		if (firstPeak == 0) {
			if ((present - before) > 0 && (present - after) > 0)
				firstPeak++;
		}
		else if (firstPeak == 1) {
			if ((present - before) < 0 && (present - after) < 0) {
				return i;
			}
		}
	}
	return 250;
}