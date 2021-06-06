#pragma once
#include "source.h"

void _findContours(const Mat& src, const vector<Point>& seeds, vector<Point>& contours) {
	for (Point p : seeds) {
		int count = 0;
		for (int i = 0; i < 9; i++) {
			Point pt = p + kernels[i];
			if (src.at<Vec3b>(pt) == Vec3b(0, 0, 0)) {
				count++;
			}
		}
		if (count > 3) {
			contours.push_back(p);
		}
		count = 0;
	}
}