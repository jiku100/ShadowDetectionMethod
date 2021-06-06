#pragma once
#include "header.h"
#include <numeric>
#include <math.h>

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
			double x1 = logf(r) - logf(g); // x1 = log(R/G)
			double x2 = logf(b) - logf(r); 	// x2 = log(B/G)
			lcs.push_back(Point2f(x1, x2));
			//if (pixel[0] != 0 && pixel[1] != 0 && pixel[2] != 0) {
			//	double x1 = log(pixel[2]) - log(pixel[1]); // x1 = log(R/G)
			//	double x2 = log(pixel[0]) - log(pixel[1]); 	// x2 = log(B/G)
			//	lcs.push_back(Point2f(x1, x2));
			//}
			//else {
			//	lcs.push_back(Point2f(0, 0));
			//}
		}
	}
}


void cal_T(vector<Point2f>& lcs, int angle, vector<double>& T) {
	double rad = angle * CV_PI / 180.;
	for (int i = 0; i < lcs.size(); i++) {
		T.push_back(lcs[i].x * cosf(rad) + lcs[i].y * sinf(rad));
	}
}

void cal_T_90(vector<double>& T, vector<double>& T_90) {
	int lower = T.size() * 0.05;
	int upper = T.size() * 0.95;
	for (int i = lower; i < upper; i++) {
		T_90.push_back(T[i]);
	}
}

void cal_bin(vector<double>& T_90, double& bin) {
	double sum = 0.;
	for (double p : T_90) {
		sum += p;
	}
	double mean = sum / T_90.size();
	double sum_dev = 0;

	for (double v : T_90) {
		sum_dev += powf(v - mean, 2);
	}

	double sigma = sqrtf(sum_dev / T_90.size());
	bin = 3.5 * sigma / powf(T_90.size(), 1. / 3.);
}

void Prob_Dist(vector<double>& T_90, double& bin, vector<double>& prob) {
	int tot_count = (int)(T_90.back() - T_90.front()) / bin;
	int count = 1;
	int c = 0;
	vector<int> freq;
	for (int i = 0; i < T_90.size(); i++) {
		if (count != tot_count) {
			if (T_90[i] <= T_90[0] + bin * count) {
				c++;
			}
			else {
				freq.push_back(c);
				c = 1;
				count++;
			}
		}
		else {
			freq.push_back(T_90.size() - i);
			break;
		}
	}
	for (double cnt : freq) {
		prob.push_back(cnt / T_90.size());
	}
}

double cal_entropy(vector<double>& prob) {
	double h = 0;
	for (double p : prob) {
		h += -p * (logf(p));
	}
	return h;
}

void getEntropy(vector<Point2f>& lcs, vector<double>& entropy) {

	for (int angle = 1; angle < 181; angle++) {
		vector<double> T;
		vector<double> T_90;	// Áß°£ 90% T
		double bin;
		vector<double> prob;

		cal_T(lcs, angle, T);
		std::sort(T.begin(), T.end());
		cal_T_90(T, T_90);
		cal_bin(T_90, bin);
		Prob_Dist(T_90, bin, prob);
		entropy.push_back(cal_entropy(prob));
	}
}