#define CBCR 1
#define ENTROPY 0
#define MEANSHIFT 0
#define ECCV10 0
#define CVPR 0

#if CBCR == 1
#include "shadow_distance.h"
Mat src;
Mat seed;
Mat result;
vector<Point> seeds;
vector<Point> contours;

void get_3_Point(int event, int x, int y, int flags, void*) {
	static int seed_count = 0;
	if(seed_count == 0)
		seed = Mat::zeros(src.size(), CV_8UC3);
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		if (seed_count < 3) {
			Point anchor = Point(x, y);
			for (int i = 0; i < 25; i++) {
				Point pt = anchor + pointArray[i];
				seed.at<Vec3b>(pt) = src.at<Vec3b>(pt);
				seeds.push_back(pt);
			}
			seed_count++;
			if (seed_count == 3) {
				imshow("seed", seed);
				for (int i = 0; i < 15; i++) {
					seed_growing(src, seed, seeds);
				}
				imshow("growing", seed);
				get_A_MS(src, result, seed, seeds);
				imshow("result", result);
				postprocessing(result, result);
				imshow("mask", result);
			}
		}
		break;
	default:
		break;
	}
}

int main(void) {
	src = imread("./src/shadow7.png");
	preprocessing(src);
	namedWindow("src");
	imshow("src", src);
	setMouseCallback("src", get_3_Point);
	
	waitKey();
	destroyAllWindows();
}
#endif

#if ENTROPY == 1
#include "entropy.h"
Mat src;
vector<Point2f> lcs;

int main(void) {
	src = imread("./src/shadow6.png");
	preprocessing(src);
	imshow("src", src);
	RGB2LCS(src, lcs);
	Mat LCS_img;
	drawLCS(lcs, LCS_img);
	int angle = calcMinimunAngle(lcs);
	Mat invariant;
	grayInvariant(src, lcs, angle, invariant);
	imshow("invariant", invariant);
	/*for (angle = 1; angle < 181; angle++) {
		
		waitKey(50);
	}*/

	
	waitKey(0);
	destroyAllWindows();
}
#endif

#if MEANSHIFT == 1
#include "source.h"
#include "MeanShift.h"

Mat src;
Mat dst;
int sp = 15;
int sr = 25;
int level = 1;

vector<Point> seeds;


void check_color(int event, int x, int y, int flags, void*) {
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		if (!overlap(seeds, Point(x, y))) {
			seeds.push_back(Point(x, y));
			cout << "Seeds Size: " << seeds.size() << endl << endl;
		}
		break;
	case EVENT_RBUTTONDOWN:
	{
		if (!seeds.empty()) {
			Vec3b median = get_median(dst, seeds);
			Vec3b& test = dst.at<Vec3b>(y, x);
			cout << "Seed Color: " << median << endl;
			cout << "Test Color: " << test << endl;
			double RGB = RGB_distance(median, test);
			cout << "RGB Distance: " << 1 - RGB << ", Angle: " << acos(RGB) << endl;
			RGB_distance_histogram(dst, median);
			double CbCr = CbCr_distance(median, test);
			cout << "CbCr Distance: " << 1 - CbCr << ", Angle: " << acos(CbCr) << endl;
			CbCr_distance_histogram(dst, median);
			double YCbCr = YCbCr_distance(median, test);
			cout << "YCbCr Distance: " << 1 - YCbCr << ", Angle: " << acos(YCbCr) << endl;
			YCbCr_distance_histogram(dst, median);
			double Y = Y_distance(median, test);
			cout << "Y Distance: " << Y << endl;
			double LCS = LCS_distance(median, test);
			cout << "LCS Distance: " << 1 - LCS << ", Angle: " << acos(LCS) << endl << endl;
			LCS_distance_histogram(dst, median);
			Y_distance_histogram(dst, median);
			Mask(dst, median);
		}
		break; 
	}
	default:
		break;
	}
}

int main(void) {	
	src = imread("./src/shadow.png");
	resize(src, src, Size(320, 240), 0, 0, INTER_CUBIC);
	bilateralFilter(src, dst, -1, 8, 4);
	namedWindow("pyrMean");
	pyrMeanShiftFiltering(dst, dst, sp, sr, level);
	setMouseCallback("pyrMean", check_color);
	imshow("src", src);
	imshow("pyrMean", dst);
	waitKey();
	destroyAllWindows();
}
#endif

#if ECCV10 == 1
#include "source.h"
#include "histogram.h"

int sp = 10;
int sr = 25;
int level = 1;

int main(void) {
	Mat src = imread("./src/shadow7.png");
	imshow("src", src);
	Mat filtered;
	bilateralFilter(src, filtered, -1, 10, 10);
	/*Mat gray;
	cvtColor(filtered, gray, COLOR_BGR2GRAY);
	Mat dx, dy;
	Sobel(gray, dx, CV_32FC1, 1, 0);
	Sobel(gray, dy, CV_32FC1, 0, 1);

	Mat edge, fmag, marker;
	magnitude(dx, dy, fmag);
	fmag.convertTo(edge, CV_8UC1);

	imshow("magnitude", edge);

	cvtColor(edge, edge, COLOR_GRAY2BGR);
	*/
	Mat mean;
	pyrMeanShiftFiltering(filtered, mean, sp, sr, level);
	imshow("mean", mean);
	Mat gray;
	cvtColor(mean, gray, COLOR_BGR2GRAY);
	double sigma[] = { 1, sqrt(2), 2, sqrt(8) };
	vector<Mat> edges;
	for (int i = 0; i < 4; i++) {
		Mat blur;
		GaussianBlur(gray, blur, Size(), sigma[i]);
		String name1 = format("%d sigma", i);
		String name2 = format("%d Canny", i);
		imshow(name1, blur);
		Mat edge;
		Canny(blur, edge, 15, 30);
		imshow(name2, edge);
		edges.push_back(edge);
	}

	Mat and1, and2, result;
	bitwise_and(edges[0], edges[1], and1);
	bitwise_and(edges[2], edges[3], and2);
	bitwise_and(and1, and2, result);

	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			if (result.at<uchar>(j, i) != 0) {
				circle(src, Point(i, j), 1, Scalar(255, 0, 0), 2);
			}
		}
	}
	imshow("result", result);
	imshow("src", src);
	waitKey();
	destroyAllWindows();
}
#endif

#if CVPR == 1
#include "source.h"
#include "math.h"

Mat src;
Mat gray;
Point l, r;

double get_Gray(Point l, Point r) {
	Vec3b& l_color = src.at<Vec3b>(l);
	Vec3b& r_color = src.at<Vec3b>(r);

	double l_gray = l_color[2] * 0.2126 + l_color[1] * 0.7152 + l_color[0] * 0.0722;
	double r_gray = r_color[2] * 0.2126 + r_color[1] * 0.7152 + r_color[0] * 0.0722;

	return abs(l_gray - r_gray);
}

double localMax(Point anchor) {
	int max = gray.at<uchar>(anchor);
	for (int i = 0; i < 9; i++) {
		Point p = anchor + kernels[i];
		if (max < gray.at<uchar>(p)) {
			max = gray.at<uchar>(p);
		}
	}
	return max;
}
void smoothness() {
	Mat blur;
	Mat stness;
	GaussianBlur(gray, blur, Size(), 5);
	subtract(gray, blur, stness);
	imshow("smoothness", stness);
}
void skewness() {
	Mat mask = imread("./src/mask.png", IMREAD_GRAYSCALE);
	imshow("mask", mask);
	Scalar mean, std;
	meanStdDev(gray, mean, std, mask);
	vector<uchar> value;
	for (int j = 0; j < mask.rows; j++) {
		for (int i = 0; i < mask.cols; i++) {
			if (mask.at<uchar>(j, i) == 255) {
				value.push_back(gray.at<uchar>(j, i));
			}
		}
	}
	std::sort(value.begin(), value.end());
	uchar middle = value[value.size() / 2];
	cout << "Shadow mean and stddev: " << mean[0] << ", " << std[0] << endl;
	cout << "Shadow skewness: "<< 3 * (mean[0] - middle) / std[0] << endl;

	Mat nonShadow_mask = Mat::ones(mask.size(), CV_8UC1) * 255;
	subtract(nonShadow_mask, mask, nonShadow_mask);
	imshow("nonShadow_mask", nonShadow_mask);
	meanStdDev(gray, mean, std, nonShadow_mask);
	value.clear();
	for (int j = 0; j < nonShadow_mask.rows; j++) {
		for (int i = 0; i < nonShadow_mask.cols; i++) {
			if (mask.at<uchar>(j, i) == 255) {
				value.push_back(gray.at<uchar>(j, i));
			}
		}
	}
	std::sort(value.begin(), value.end());
	middle = value[value.size() / 2];
	cout << "Non Shadow mean and stddev: " << mean[0] << ", " << std[0] << endl;
	cout << "Non Shadow skewness: " << 3 * (mean[0] - middle) / std[0] << endl;
}
void click(int event, int x, int y, int flags, void*) {
	Mat cl;
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		l = Point(x, y);
		break;
	case EVENT_RBUTTONDOWN:
		cl = src.clone();
		r = Point(x, y);
		circle(cl, l, 5, Scalar(0, 0, 255), 2);
		circle(cl, r, 5, Scalar(0, 0, 255), 2);
		imshow("src", cl);
		cout << "Gray distance: " << get_Gray(l, r) << endl;
		cout << "Local Max on L Point: " << localMax(l) << "  Local Max on R Point: " << localMax(r) << endl;
		smoothness();
		skewness();
	default:
		break;
	}
}
int main(void) {
	src = imread("./src/shadow.png");
	cvtColor(src, gray, COLOR_BGR2GRAY);
	pyrMeanShiftFiltering(src, src, 15, 25, 1);
	namedWindow("src");
	setMouseCallback("src", click);
	imshow("src", src);
	waitKey();
	destroyAllWindows();
}
#endif