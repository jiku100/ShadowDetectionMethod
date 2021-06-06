#pragma once
#include "source.h"
#include "histogram.h"
#include "contour.h"

Vec3b get_median(Mat& src, vector<Point>& seeds) {
	double r = 0, g = 0, b = 0;

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

int overlap(vector<Point>& seeds, Point p) {
	for (Point pt : seeds) {
		if (pt == p) {
			return 1;
		}
	}
	return 0;
}

double RGB_distance(Vec3b& seed, Vec3b& pixel) {
	double dotProduct = (seed[0] * pixel[0] + seed[1] * pixel[1] + seed[2] * pixel[2]);
	double seed_length = sqrt(seed[0] * seed[0] + seed[1] * seed[1] + seed[2] * seed[2]);
	double pixel_length = sqrt(pixel[0] * pixel[0] + pixel[1] * pixel[1] + pixel[2] * pixel[2]);
	double cos_theta = dotProduct / (seed_length * pixel_length);
	return abs(cos_theta);
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

double YCbCr_distance(Vec3b& seed, Vec3b& pixel) {
	double Y = 0.299 * seed[2] + 0.587 * seed[1] + 0.114 * seed[0];
	double Y_ML = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];
	double Cb_seed = 128 + (-0.169 * seed[2]) + (-0.331 * seed[1]) + 0.5 * seed[0];
	double Cb_pixel = 128 + (-0.169 * pixel[2]) + (-0.331 * pixel[1]) + 0.5 * pixel[0];
	double Cr_seed = 128 + 0.5 * seed[2] + (-0.419 * seed[1]) + (-0.081 * seed[0]);
	double Cr_pixel = 128 + 0.5 * pixel[2] + (-0.419 * pixel[1]) + (-0.081 * pixel[0]);

	double dotProdudct = Y * Y_ML + Cb_seed * Cb_pixel + Cr_seed * Cr_pixel;
	double seed_length = sqrt(Y * Y + Cb_seed * Cb_seed + Cr_seed * Cr_seed);
	double pixel_length = sqrt(Y_ML * Y_ML + Cb_pixel * Cb_pixel + Cr_pixel * Cr_pixel);
	double cos_theta = dotProdudct / (seed_length * pixel_length);
	return abs(cos_theta);
}

double LCS_distance(Vec3b& seed, Vec3b& pixel) {
	Vec3b seed_color = Vec3b(seed[0], seed[1], seed[2]);
	Vec3b pixel_color = Vec3b(pixel[0], pixel[1], pixel[2]);

	if (seed_color[0] < 0.1) {
		seed_color[0] = 0.1;
	}
	if (seed_color[1] < 0.1) {
		seed_color[1] = 0.1;
	}
	if (seed_color[2] < 0.1) {
		seed_color[2] = 0.1;
	}
	if (pixel_color[0] < 0.1) {
		pixel_color[0] = 0.1;
	}
	if (pixel_color[1] < 0.1) {
		pixel_color[1] = 0.1;
	}
	if (pixel_color[2] < 0.1) {
		pixel_color[2] = 0.1;
	}
	Vec2f seed_lcs = (logf(seed_color[2]) - logf(seed_color[1]), logf(seed_color[0]) - logf(seed_color[1]));
	Vec2f pixel_lcs = (logf(pixel_color[2]) - logf(pixel_color[1]), logf(pixel_color[0]) - logf(pixel_color[1]));

	double dotProdudct = seed_lcs[0] * pixel_lcs[0] + seed_lcs[1] * pixel_lcs[1];
	double seed_length = sqrt(seed_lcs[0] * seed_lcs[0] + seed_lcs[1] * seed_lcs[1]);
	double pixel_length = sqrt(pixel_lcs[0] * pixel_lcs[0] + pixel_lcs[1] * pixel_lcs[1]);
	double cos_theta = dotProdudct / (seed_length * pixel_length);
	return abs(cos_theta);
}

void RGB_distance_histogram(Mat& src, Vec3b& seed) {
	vector<double> distances;
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			distances.push_back(1 - RGB_distance(seed, src.at<Vec3b>(j, i)));
		}
	}
	Mat dis = Mat(distances);
	normalize(dis, dis, 0, 255, NORM_MINMAX, CV_8UC1);
	dis.convertTo(dis, CV_8UC1);
	Mat hist = _calcHist(dis);
	GaussianBlur(hist, hist, Size(), 5);
	Mat histImg = getHistImage(hist);

	imshow("RGB_Distance", histImg);
}

void CbCr_distance_histogram(Mat& src, Vec3b& seed) {
	vector<double> distances;
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			distances.push_back(1 - CbCr_distance(seed, src.at<Vec3b>(j, i)));
		}
	}
	Mat dis = Mat(distances);
	normalize(dis, dis, 0, 255, NORM_MINMAX, CV_8UC1);
	dis.convertTo(dis, CV_8UC1);
	Mat hist = _calcHist(dis);
	GaussianBlur(hist, hist, Size(), 5);
	Mat histImg = getHistImage(hist);

	imshow("CbCr_Distance", histImg);
}

void YCbCr_distance_histogram(Mat& src, Vec3b& seed) {
	vector<double> distances;
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			distances.push_back(1 - YCbCr_distance(seed, src.at<Vec3b>(j, i)));
		}
	}
	Mat dis = Mat(distances);
	normalize(dis, dis, 0, 255, NORM_MINMAX, CV_8UC1);
	dis.convertTo(dis, CV_8UC1);
	Mat hist = _calcHist(dis);
	GaussianBlur(hist, hist, Size(), 5);
	Mat histImg = getHistImage(hist);

	imshow("YCbCr_Distance", histImg);
}

void Y_distance_histogram(Mat& src, Vec3b& seed) {
	vector<double> distances;
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			distances.push_back(YCbCr_distance(seed, src.at<Vec3b>(j, i)));
		}
	}
	Mat dis = Mat(distances);
	normalize(dis, dis, 0, 255, NORM_MINMAX, CV_8UC1);
	dis.convertTo(dis, CV_8UC1);
	Mat hist = _calcHist(dis);
	GaussianBlur(hist, hist, Size(), 5);
	Mat histImg = getHistImage(hist);

	imshow("Y_Distance", histImg);
}

void LCS_distance_histogram(Mat& src, Vec3b& seed) {
	vector<double> distances;
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			distances.push_back(1 - LCS_distance(seed, src.at<Vec3b>(j, i)));
		}
	}
	Mat dis = Mat(distances);
	normalize(dis, dis, 0, 255, NORM_MINMAX, CV_8UC1);
	dis.convertTo(dis, CV_8UC1);
	Mat hist = _calcHist(dis);
	GaussianBlur(hist, hist, Size(), 5);
	Mat histImg = getHistImage(hist);

	imshow("LCS_Distance", histImg);
}

void YCbCr_Mask(Mat& src, Vec3b& seed) {
	vector<double> distances;
	vector<Point> check;
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			distances.push_back(1 - YCbCr_distance(seed, src.at<Vec3b>(j, i)));
			check.push_back(Point(i, j));
		}
	}
	Mat dis = Mat(distances).reshape(1, src.rows);
	normalize(dis, dis, 0, 255, NORM_MINMAX, CV_8UC1);
	dis.convertTo(dis, CV_8UC1);
	Mat hist = _calcHist(dis);
	GaussianBlur(hist, hist, Size(), 5);
	double threshold = findVally(hist);
	Mat histImg = getHistImage(hist);
	line(histImg, Point(threshold, 100), Point(threshold, 0), Scalar(255, 0, 0), 1);
	imshow("YCbCr_Distance", histImg);

	Mat dst = Mat::zeros(src.size(), CV_8UC1);
	for (int j = 0; j < dis.rows; j++) {
		for (int i = 0; i < dis.cols; i++) {
			if (dis.at<uchar>(j,i) < threshold) {
				dst.at<uchar>(j, i) = 255;
			}
		}
	}
	imshow("mask", dst);
}

void Mask(Mat& src, Vec3b& seed) {
	vector<double> CbCr_distances;
	vector<double> Y_distances;
	vector<Point> check;
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			CbCr_distances.push_back(1 - CbCr_distance(seed, src.at<Vec3b>(j, i)));
			Y_distances.push_back(Y_distance(seed, src.at<Vec3b>(j, i)));
			check.push_back(Point(i, j));
		}
	}

	Mat CbCr_dis = Mat(CbCr_distances).reshape(1, src.rows);
	normalize(CbCr_dis, CbCr_dis, 0, 255, NORM_MINMAX, CV_8UC1);
	CbCr_dis.convertTo(CbCr_dis, CV_8UC1);
	Mat CbCr_hist = _calcHist(CbCr_dis);
	GaussianBlur(CbCr_hist, CbCr_hist, Size(), 5);
	double CbCr_threshold = findVally(CbCr_hist);
	Mat CbCr_histImg = getHistImage(CbCr_hist);
	line(CbCr_histImg, Point(CbCr_threshold, 100), Point(CbCr_threshold, 0), Scalar(255, 0, 0), 1);
	imshow("CbCr_Distance", CbCr_histImg);

	Mat Y_dis = Mat(Y_distances).reshape(1, src.rows);
	normalize(Y_dis, Y_dis, 0, 255, NORM_MINMAX, CV_8UC1);
	Y_dis.convertTo(Y_dis, CV_8UC1);
	Mat Y_hist = _calcHist(Y_dis);
	GaussianBlur(Y_hist, Y_hist, Size(), 5);
	double Y_threshold = findVally(Y_hist);
	Mat Y_histImg = getHistImage(Y_hist);
	line(Y_histImg, Point(Y_threshold, 100), Point(Y_threshold, 0), Scalar(255, 0, 0), 1);
	imshow("Y_Distance", Y_histImg);

	Mat dst = Mat::zeros(src.size(), CV_8UC1);
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			if (CbCr_dis.at<uchar>(j, i) < CbCr_threshold) {
				if (Y_dis.at<uchar>(j, i) < Y_threshold) {
					dst.at<uchar>(j, i) = 255;
				}
			}
		}
	}
	imshow("mask", dst);
}

