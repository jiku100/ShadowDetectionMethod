#pragma once
#include "source.h"
#include "histogram.h"
#include "contours.h"


void postprocessing(Mat& src, Mat& dst, Mat& original) {
	Mat src_gray;
	Mat src_filtered;
	bilateralFilter(src, src_filtered, -1, 15, 10);
	cvtColor(src_filtered, src_gray, COLOR_BGR2GRAY);
	threshold(src_gray, src_gray, 0, 255, THRESH_BINARY);
	imshow("threshold", src_gray);
	Mat m = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(src_gray, dst, MORPH_OPEN, Mat(), Point(-1, -1), 2);
	morphologyEx(src_gray, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
	imshow("morph", dst);

	vector<vector<Point>> contours;
	findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	drawContours(dst, contours, -1, Scalar(255), -1);
	/*Mat result;
	resize(dst, result, original.size(), 0, 0, INTER_CUBIC);
	dst = result.clone();*/
}

void preprocessing(Mat& src, Mat& dst, Mat& stness) {
	Mat processed;
	Mat gray;
	Mat blurred;
	resize(src, dst, Size(300, 400), 0, 0, INTER_LANCZOS4);
	
	bilateralFilter(dst, processed, -1, 20, 10);
	pyrMeanShiftFiltering(processed, processed, 15, 25, 1);
	cvtColor(processed, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blurred, Size(), 5);
	subtract(gray, blurred, stness);
	dst = processed.clone();
}

double CbCr_distance(Vec3b& seed, Vec3b& pixel) {
	double Cb_seed = 128 + (-0.169 * seed[2]) + (-0.331 * seed[1]) + 0.5 * seed[0];
	double Cb_pixel = 128 + (-0.169 * pixel[2]) + (-0.331 * pixel[1]) + 0.5 * pixel[0];
	double Cr_seed = 128 + 0.5 * seed[2] + (-0.419 * seed[1]) + (-0.081 * seed[0]);
	double Cr_pixel = 128 + 0.5 * pixel[2] + (-0.419 * pixel[1]) + (-0.081 * pixel[0]);

	double dotProdudct = Cb_seed * Cb_pixel + Cr_seed * Cr_pixel;
	double seed_length = sqrt(Cb_seed * Cb_seed + Cr_seed * Cr_seed);
	double pixel_length = sqrt(Cb_pixel * Cb_pixel + Cr_pixel * Cr_pixel);
	double cos_theta = dotProdudct / (seed_length * pixel_length);
	return abs(cos_theta);
}

Vec3b get_median(Mat& src, vector<Point>& seeds) {
	double r = 0;
	double g = 0;
	double b = 0;

	for (Point p : seeds) {
		Vec3b& color = src.at<Vec3b>(p);
		b += color[0];
		g += color[1];
		r += color[2];
	}
	int median_b = b / seeds.size();
	int median_g = g / seeds.size();
	int median_r = r / seeds.size();
	return Vec3b(median_b, median_g, median_r);
}

double get_sdv(const Mat& img) {
	double _sum = 0;
	double count = 0;
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			_sum += img.at<uchar>(j, i);
			count++;
		}
	}
	_sum /= count;
	double variance = 0;
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			variance += pow(img.at<uchar>(j, i) - _sum, 2);
		}
	}
	return sqrt(variance / count);
}

double Y_distance(double Y, Vec3b& pixel) {
	double Y_ML = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];
	return abs(Y - Y_ML);
}

double Y_distance(Vec3b& seed, Vec3b& pixel) {
	double Y = 0.299 * seed[2] + 0.587 * seed[1] + 0.114 * seed[0];
	double Y_ML = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];
	return abs(Y - Y_ML);
}

double get_seed_Y(Mat& src, vector<Point>& seeds) {
	double Y = 0;
	for (Point p : seeds) {
		Vec3b& color = src.at<Vec3b>(p);
		Y += 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0];
	}
	Y /= seeds.size();
	return Y;
}

double get_Y_threshold(Mat& src, double target) {
	vector<double> y_distances;
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			y_distances.push_back(Y_distance(target, src.at<Vec3b>(j, i)));
		}
	}
	Mat y_dis = Mat(y_distances);
	y_dis.convertTo(y_dis, CV_8UC1);
	Mat y_hist = _calcHist(y_dis);
	GaussianBlur(y_hist, y_hist, Size(), 5);
	double y_threshold = findVally(y_hist);
	return y_threshold;
}

double smoothness(Mat& stness, Point& anchor) {
	double sum = 0;
	for (int i = 0; i < 25; i++) {
		Point pt = anchor + kernels_5[i];
		sum += stness.at<uchar>(pt);
	}
	sum /= 25;
	return sum;
}

double localMax(Mat& gray, Point& anchor) {
	double max = gray.at<uchar>(anchor);
	for (int i = 0; i < 25; i++) {
		Point pt = anchor + kernels_5[i];
		double Y = gray.at<uchar>(pt);
		if (max < Y) {
			max = Y;
		}
	}
	return max;
}

double gradSum(Mat& mag, Point& anchor) {
	double sum = 0;
	for (int i = 0; i < 25; i++) {
		Point pt = anchor + kernels_5[i];
		sum += mag.at<uchar>(pt);
	}
	sum /= 25;
	return sum;
}

int check_gradSum(double gradSum) {
	if (gradSum < 5)
		return 1;
	else
		return 0;
}

int check_localMax(double localMax, double mean) {
	if (abs(localMax - mean) < 10) {
		return 1;
	}
	else {
		return 0;
	}
}

int check_cbcr(double distance) {
	static double cos_1 = 1 - abs(cos(CV_PI / 180));
	if (distance < cos_1)
		return 1;
	else
		return 0;
}

int check_y(double distance, double target) {
	
	if (distance < target)
		return 1;
	else
		return 0;
}

int check_smoothness(double distance) {
	if (distance < 1)
		return 1;
	else
		return 0;
}

void seed_growing(Mat& src, Mat& dst, vector<Point>& seeds, Mat& stness, Mat& gray, Mat& mag, double y_threshold) {
	vector<Point> contours;
	vector<double> cbcr_distances;
	vector<double> y_distances;
	vector<Point> checkPoint;
	vector<Point> growPoint;

	double Y = get_seed_Y(src, seeds);
	Vec3b median = get_median(src, seeds);
	int check = 0;

	_findContours(dst, seeds, contours);
	for (Point anchor : contours) {
		for (int i = 0; i < 25; i++) {
			check = 0;
			Point p = anchor + kernels_5[i];
			if (dst.at<Vec3b>(p) == Vec3b(0, 0, 0)) {
				double cbcr_distance = 1 - CbCr_distance(median, src.at<Vec3b>(p));
				double y_distance = Y_distance(Y, src.at<Vec3b>(p));
				double smoothness_distance = smoothness(stness, p);
				double localMax_distance = localMax(gray, p);
				double gradSum_distance = gradSum(mag, p);
				if (check_cbcr(cbcr_distance)) {
					check++;
				}
				if (check_y(y_distance, y_threshold)) {
					check++;
				}
				if (check_smoothness(smoothness_distance)) {
					check++;
				}
				if (check_localMax(localMax_distance, Y)) {
					check++;
				}
				if (check_gradSum(gradSum_distance)) {
					check++;
				}
				if (check >= 3) {
					growPoint.push_back(p);
				}
			}
		}
	}

	for (Point p : growPoint) {
		dst.at<Vec3b>(p) = src.at<Vec3b>(p);
		if (!overlap(seeds, p)) {
			seeds.push_back(p);
		}
	}
}