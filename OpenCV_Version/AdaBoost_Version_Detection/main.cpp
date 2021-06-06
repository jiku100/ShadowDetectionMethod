#include "shadowDetection.h"
Mat src;
Mat processed;
Mat seed;
Mat result;
Mat gray;
Mat stness;
Mat mag;
vector<Point> seeds;
vector<Point> contours;

void get_3_Point(int event, int x, int y, int flags, void*) {
	static int seed_count = 0;
	if (seed_count == 0)
		seed = Mat::zeros(processed.size(), CV_8UC3);
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		if (seed_count < 3) {
			Point anchor = Point(x, y);
			for (int i = 0; i < 25; i++) {
				Point pt = anchor + kernels_5[i];
				seed.at<Vec3b>(pt) = processed.at<Vec3b>(pt);
				seeds.push_back(pt);
			}
			seed_count++;
			if (seed_count == 3) {
				double Y = get_seed_Y(processed, seeds);
				double y_theshold = get_Y_threshold(processed, Y);
				imshow("seed", seed);
				Mat result;
				double sdv = 0;
				int count = 0;
				while (1) {
					seed_growing(processed, seed, seeds, stness, gray, mag, y_theshold);
					Mat MS_Gray;
					cvtColor(seed, MS_Gray, COLOR_BGR2GRAY);
					double std_dev_MS;
					std_dev_MS = get_sdv(MS_Gray);
					if (sdv == std_dev_MS) {
						count++;
					}
					else {
						sdv = std_dev_MS;
						result = seed.clone();
						count = 0;
					}
					if (count == 5) {
						break;
					}
				}
				imshow("result", result);
				postprocessing(result, result, src);
				imshow("mask", result);
			}
		}
		break;
	default:
		break;
	}
}


int main(void) {
	src = imread("./src/shadow3.png");
	imshow("Input Image", src);
	preprocessing(src, processed, stness);
	namedWindow("src");
	imshow("src", processed);
	imshow("stness", stness);
	cvtColor(processed, gray, COLOR_BGR2GRAY);
	Mat dx, dy;
	Sobel(gray, dx, CV_32FC1, 1, 0);
	Sobel(gray, dy, CV_32FC1, 0, 1);

	Mat fmag;
	magnitude(dx, dy, fmag);
	fmag.convertTo(mag, CV_8UC1);
	imshow("mag", mag);
	setMouseCallback("src", get_3_Point);
	waitKey();
	destroyAllWindows();
}