#pragma once
#include "header.h"

void get_T(vector<Point2f>& lcs, int angle, vector<double>& T) {
	double rad = angle * CV_PI / 180;
	for (int i = 0; i < lcs.size(); i++) {
		T.push_back(lcs[i].x * cosf(rad) + lcs[i].y * sinf(rad));
	}
}

void intrin_image(vector<double>& T, vector<int>& gray_T) {
	double min_T = T[0], max_T = T[0];
	for (int i = 1; i < T.size(); i++) {
		max_T = max(max_T, T[i]);
		min_T = min(min_T, T[i]);
	}


	for (int i = 0; i < T.size(); i++) {
		gray_T.push_back(cvRound(255 * (T[i] - min_T) / (max_T - min_T)));
	}
}

void get_invariant(vector<Point2f>& lcs, int angle, vector<int>& gray_T) {
	vector<double> T;
	get_T(lcs, angle, T);
	intrin_image(T, gray_T);
}