#pragma once
#include "source.h"
#include "histogram.h"
#include "contour.h"

void postprocessing(Mat& src, Mat& dst) {
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
}

void preprocessing(Mat& src) {
	Mat processed;
	resize(src, src, Size(300, 400), 0, 0, INTER_LANCZOS4);
	bilateralFilter(src, processed, -1, 20, 10);
	pyrMeanShiftFiltering(processed, processed, 15, 25, 1);
	src = processed.clone();
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

void normalize_vector(vector<double>& distance, vector<int>& distances_norm) {
	double max = distance[0];
	double min = distance[0];
	for (int i = 0; i < distance.size(); i++) {
		if (distance[i] > max)
			max = distance[i];
		if (distance[i] < min)
			min = distance[i];
	}
	for (int i = 0; i < distance.size(); i++) {
		distances_norm.push_back(255 * (distance[i] - min) / (max - min));
	}
}

Vec3b get_median(Mat& src, vector<Point>& seeds) {
	vector<int> r;
	vector<int> g;
	vector<int> b;

	for (Point p : seeds) {
		Vec3b& color = src.at<Vec3b>(p);
		b.push_back(color[0]);
		g.push_back(color[1]);
		r.push_back(color[2]);
	}
	int median_b = b[b.size() / 2];
	int median_g = g[g.size() / 2];
	int median_r = r[r.size() / 2];
	return Vec3b(median_b, median_g, median_r);
}

int get_threshold(const Mat& hist) {
	float total = 0;
	float count = 0;
	for (int i = 0; i < hist.total(); i++) {
		total += hist.at<float>(i, 0);
	}
	for (int i = 0; i < hist.total(); i++) {
		count += hist.at<float>(i, 0);
		if (count > total / 5)
			return i;
	}
	return hist.total();
}

int overlap(vector<Point>& seeds, Point p) {
	for (Point pt : seeds) {
		if (pt == p) {
			return 1;
		}
	}
	return 0;
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

void seed_growing(Mat& src, Mat& dst, vector<Point>& seeds) {
	vector<Point> contours;
	vector<double> cbcr_distances;
	vector<double> y_distances;
	vector<Point> checkPoint;
	vector<Point> growPoint;

	Vec3b median = get_median(src, seeds);

	_findContours(dst, seeds, contours);
	for (Point anchor : contours) {
		for (int i = 0; i < 25; i++) {
			Point p = anchor + pointArray[i];
			if (dst.at<Vec3b>(p) == Vec3b(0, 0, 0)) {
				double cbcr_distance = 1 - CbCr_distance(median, src.at<Vec3b>(p));
				cbcr_distances.push_back(cbcr_distance);
				checkPoint.push_back(p);
			}
		}
	}


	double cos_1 = 1 - abs(cos(CV_PI / 180));
	for (int i = 0; i < cbcr_distances.size(); i++) {
		if (cbcr_distances[i] < cos_1) {
			growPoint.push_back(checkPoint[i]);
		}
	}

	for (Point p : growPoint) {
		dst.at<Vec3b>(p) = src.at<Vec3b>(p);
		if (!overlap(seeds, p)) {
			seeds.push_back(p);
		}
	}
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

void get_A_MS(Mat& src, Mat& dst, Mat& seed, vector<Point>& seeds) {
	Mat A = src.clone();
	Mat MS = seed.clone();
	Mat ML;
	subtract(A, MS, ML);
	Mat result = MS.clone();

	double sdv = 0;
	int count = 0;
	while (1) {
		double Y = get_seed_Y(src, seeds);

		Vec3b median = get_median(src, seeds);

		vector<Point> contours;
		vector<double> cbcr_distances;
		vector<double> y_distances;
		vector<Point> checkPoint;
		vector<Point> growPoint;
		_findContours(MS, seeds, contours);
		for (Point anchor : contours) {
			for (int i = 0; i < 25; i++) {
				Point p = anchor + pointArray[i];
				if (MS.at<Vec3b>(p) == Vec3b(0, 0, 0)) {
					double cbcr_distance = 1 - CbCr_distance(median, src.at<Vec3b>(p));
					cbcr_distances.push_back(cbcr_distance);
					double y_distance = Y_distance(Y, src.at<Vec3b>(p));
					y_distances.push_back(y_distance);
					checkPoint.push_back(p);
				}
			}
		}

		Mat y_dis = Mat(y_distances);
		y_dis.convertTo(y_dis, CV_8UC1);
		Mat y_hist = _calcHist(y_dis);
		GaussianBlur(y_hist, y_hist, Size(), 5);
		double y_threshold = findVally(y_hist);
		Mat histImg = getHistImage(y_hist);
		line(histImg, Point(y_threshold, 100), Point(y_threshold, 0), Scalar(255, 0, 0), 1);

		imshow("y", histImg);
		double cos_1 = 1 - abs(cos(CV_PI / 180));
		for (int i = 0; i < cbcr_distances.size(); i++) {
			if (cbcr_distances[i] < cos_1) {
				if (y_distances[i] < y_threshold) {
					growPoint.push_back(checkPoint[i]);
				}
			}
		}

		for (Point p : growPoint) {
			MS.at<Vec3b>(p) = src.at<Vec3b>(p);
			if (!overlap(seeds, p)) {
				seeds.push_back(p);
			}
		}
		subtract(A, MS, ML);
		Mat MS_Gray, ML_Gray;
		cvtColor(MS, MS_Gray, COLOR_BGR2GRAY);
		cvtColor(ML, ML_Gray, COLOR_BGR2GRAY);
		double std_dev_MS, std_dev_ML;
		std_dev_MS = get_sdv(MS_Gray);
		std_dev_ML = get_sdv(ML_Gray);
		if (sdv == std_dev_MS) {
			count++;
		}
		else {
			sdv = std_dev_MS;
			dst = result.clone();
			count = 0;
		}
		if (count == 5) {
			imshow("A", A);
			imshow("MS", MS);
			imshow("ML", ML);
			break;
		}
		else {
			result.release();
			result = MS.clone();
		}
	}
}