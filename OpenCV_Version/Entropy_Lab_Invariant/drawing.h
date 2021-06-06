#pragma once
#include "header.h"

void drawLCS(vector<Point2f>& lcs, Mat& dst) {
	vector<float> x1, x2;

	for (Point2f p : lcs) {
		x1.push_back(p.x);
		x2.push_back(p.y);
	}

	Mat x1_m = Mat(x1);
	Mat x2_m = Mat(x2);

	double minVal, maxVal;
	minMaxLoc(x1_m, &minVal, &maxVal);
	cout << minVal << " " << maxVal << endl;
	int width = cvRound(abs(maxVal - minVal) * 30);
	cout << width << endl;
	minMaxLoc(x2_m, &minVal, &maxVal);
	int height = cvRound(abs(maxVal - minVal) * 30);
	cout << height << endl;

	dst = Mat::zeros(Size(width + 50, height + 50), CV_8UC1);
	Mat x1_norm, x2_norm;
	normalize(x1_m, x1_norm, 0, width, NORM_MINMAX, CV_32FC1);
	normalize(x2_m, x2_norm, 0, height, NORM_MINMAX, CV_32FC1);

	Mat dstROI = dst(Rect(25, 25, width + 1, height + 1));
	for (int i = 0; i < x1_norm.rows; i++) {
		int x = cvRound(x1_norm.at<float>(i, 0));
		int y = dstROI.rows - 1 - cvRound(x2_norm.at<float>(i, 0));
		dstROI.at<uchar>(y, x) = 255;
	}
	imshow("LCS", dst);
}

void drawEntropy(vector<double>& entropy, int& min_angle, Mat& graph){
	vector<Point> pts;
	double minVal, maxVal;
	Point minPos, maxPos;
	Mat entropy_mat = Mat(entropy);
	minMaxLoc(entropy_mat, &minVal, &maxVal, &minPos, &maxPos);

	Mat entropy_norm;
	normalize(entropy_mat, entropy_norm, 50, 250, NORM_MINMAX, CV_32FC1);
	min_angle =  minPos.y;
	cout << "Min Angle: " << min_angle << endl;
	for (int i = 0; i < entropy.size(); i++) {
		pts.push_back(Point(i * 2 + 20, 400 - cvRound(entropy_norm.at<float>(i, 0))));
	}
	
	graph = Mat(Size(400, 400), CV_8UC3, Scalar(255, 255, 255));

	polylines(graph, pts, false, Scalar(0, 0, 255), 2, LINE_AA);
	arrowedLine(graph, Point(min_angle * 2 + 20, 400), Point(min_angle * 2 + 20, 400 - cvRound(entropy_norm.at<float>(min_angle, 0))), Scalar(255, 0, 0), 2, LINE_AA);
	imshow("Entropy", graph);
}

void drawInvariant(Mat& src, Mat& invariant, vector<int>& gray_T) {
	invariant = Mat::zeros(Size(src.cols, src.rows), CV_8UC1);
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			invariant.at<uchar>(j, i) = saturate_cast<uchar>(gray_T[src.cols * j + i]);
		}
	}
	GaussianBlur(invariant, invariant, Size(), 1);
	imshow("invariant", invariant);
}