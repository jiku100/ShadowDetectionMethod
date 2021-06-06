#pragma once
#include "source.h"
#include "contour.h"
#include "histogram.h"

void preprocessing(Mat& src) {
	Mat processed;
	resize(src, processed, Size(300, 400), 0, 0, INTER_LANCZOS4);
	//bilateralFilter(src, processed, -1, 8, 4);
	src = processed.clone();
}

void drawLCS(vector<Point2f>& lcs, Mat& dst) {
	vector<float> x1, x2;

	for (Point2f p : lcs) {
		x1.push_back(p.x);
		x2.push_back(p.y);
	}

	Mat x1_m = Mat(x1);
	Mat x2_m = Mat(x2);

	double minVal_x1, maxVal_x1, minVal_x2, maxVal_x2;
	minMaxLoc(x1_m, &minVal_x1, &maxVal_x1);
	maxVal_x1 += abs(minVal_x1);
	x1_m += Scalar(abs(minVal_x1));

	minMaxLoc(x2_m, &minVal_x2, &maxVal_x2);
	maxVal_x2 += abs(minVal_x2);
	x2_m += Scalar(abs(minVal_x2));

	int width = cvRound(maxVal_x1 * 30);
	int height = cvRound(maxVal_x2 * 30);
	dst = Mat::zeros(Size(width + 50, height + 50), CV_8UC1);

	Mat dstROI = dst(Rect(25, 25, width + 1, height + 1));
	for (int i = 0; i < x1_m.rows; i++) {
		int x = cvRound(x1_m.at<float>(i, 0) * 30);
		int y = dstROI.rows - 1 - cvRound(x2_m.at<float>(i, 0) * 30);
		dstROI.at<uchar>(y, x) = 255;
	}
	imshow("LCS", dst);
}

void drawEntropy(vector<double>& entropy, Vec2i& min_angle, Mat& graph) {
	
	vector<Point> pts;
	Mat entropy_mat = Mat(entropy);
	Mat entropy_norm;
	normalize(entropy_mat, entropy_norm, 50, 250, NORM_MINMAX, CV_32FC1);
	for (int i = 0; i < entropy.size(); i++) {
		pts.push_back(Point(i * 2 + 20, 400 - cvRound(entropy_norm.at<float>(i, 0))));
	}

	graph = Mat(Size(400, 400), CV_8UC3, Scalar(255, 255, 255));

	polylines(graph, pts, false, Scalar(0, 0, 255), 2, LINE_AA);
	arrowedLine(graph, Point(min_angle[0] * 2 + 20, 400), Point(min_angle[0] * 2 + 20, 400 - cvRound(entropy_norm.at<float>(min_angle[0], 0))), Scalar(255, 0, 0), 2, LINE_AA);
	imshow("Entropy", graph);
}

void RGB2LCS(Mat& src, vector<Point2f>& lcs) {
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			Vec3b& pixel = src.at<Vec3b>(j, i);
			double b = pixel[0];
			double g = pixel[1];
			double r = pixel[2];
			if (b < 0.1) {
				b = 0.1;
			}
			if (g < 0.1) {
				g = 0.1;
			}
			if (r < 0.1) {
				r = 0.1;
			}
			double RGB_mean = powf(r * g * b, 1 / 3.);
			double x1 = logf(r) - logf(g); // x1 = log(R/G)
			double x2 = logf(b) - logf(g);	// x2 = log(B/G)
			lcs.push_back(Point2f(x1, x2));
		}
	}
}

void calcT(vector<Point2f>& lcs, int angle, vector<double>& T) {
	double rad = angle * CV_PI / 180.;
	for (int i = 0; i < lcs.size(); i++) {
		T.push_back(lcs[i].x * cosf(rad) + lcs[i].y * sinf(rad));
	}
}

void getMiddleT(vector<double>& T, vector<double>& T_Middle) {

	int lower = T.size() * 0.05;
	int upper = T.size() * 0.95;
	for (int i = lower; i < upper; i++) {
		T_Middle.push_back(T[i]);
	}
}

double calcBin(vector<double>& T) {
	double sum = 0.;
	for (double p : T) {
		sum += p;
	}
	double mean = sum / T.size();
	double sum_dev = 0;

	for (double v : T) {
		sum_dev += powf(v - mean, 2);
	}

	double sigma = sqrtf(sum_dev / T.size());
	return 3.5 * sigma * powf(T.size(), -1. / 3.);
}

void Prob_Dist(vector<double>& T, double& bin, vector<double>& prob) {
	int tot_count = (int)(T.back() - T.front()) / bin;
	int count = 1;
	int c = 0;
	vector<int> freq;
	for (int i = 0; i < T.size(); i++) {
		if (count != tot_count) {
			if (T[i] <= T[0] + bin * count) {
				c++;
			}
			else {
				freq.push_back(c);
				c = 1;
				count++;
			}
		}
		else {
			freq.push_back(T.size() - i);
			break;
		}
	}
	for (double cnt : freq) {
		prob.push_back(cnt / T.size());
	}
}
	
double calcEntropy(vector<double>& prob) {
	double h = 0;
	for (double p : prob) {
		h += -p * (logf(p));
	}
	return h;
}

Vec2i findMinAngle(vector<double>& entropy) {
	int minIdx = 0;
	double minVal = entropy[minIdx];
	for (int i = 0; i < entropy.size(); i++) {
		if (minVal > entropy[i]) {
			minVal = entropy[i];
			minIdx = i;
		}
	}
	return Vec2b(minIdx, minVal);
}

int calcMinimunAngle(vector<Point2f>& lcs) {
	Vec2i min;
	vector<double> entropy;
	for (int angle = 1; angle < 181; angle++) {
		vector<double> T;
		vector<double> T_middle;
		double bin = 0;
		vector<double> prob;

		calcT(lcs, angle, T);
		std::sort(T.begin(), T.end());
		getMiddleT(T, T_middle);
		bin = calcBin(T_middle);
		Prob_Dist(T_middle, bin, prob);
		entropy.push_back(calcEntropy(prob));
	}
	min = findMinAngle(entropy);
	Mat EntropyImg;
	drawEntropy(entropy, min, EntropyImg);
	return min[0];
}

void T2Grayimg(Mat& src, vector<double>& T, Mat& invariant) {
	double min_T = T[0], max_T = T[0];
	for (int i = 1; i < T.size(); i++) {
		max_T = max(max_T, T[i]);
		min_T = min(min_T, T[i]);
	}

	invariant = Mat::zeros(src.size(), CV_8UC1);
	for (int i = 0; i < T.size(); i++) {
		int x = i % src.cols;
		int y = i / src.cols;
		invariant.at<uchar>(y,x) = (cvRound(255 * (T[i] - min_T) / (max_T - min_T)));
	}
}

void grayInvariant(Mat& src, vector<Point2f>& lcs, int angle, Mat& invariant) {
	vector<double> T;
	calcT(lcs, angle, T);
	T2Grayimg(src, T, invariant);
}