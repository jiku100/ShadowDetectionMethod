#include "header.h"
#include "drawing.h"
#include "entropy.h"
#include "invariant.h"
#include "lab.h"
#include <stack>

#define ENTROPY 1
#define LAB 0
#define DETECTION 0


Point pointArray[25] =
{
	Point(0,0),
	Point(-2,-2),
	Point(-1,-2),
	Point(0,-2),
	Point(1,-2),
	Point(2,-2),
	Point(-2,-1),
	Point(-1,-1),
	Point(0,-1),
	Point(1,-1),
	Point(2,-1),
	Point(-2,0),
	Point(-1,0),
	Point(1,0),
	Point(2,0),
	Point(-2,1),
	Point(-1,1),
	Point(0,1),
	Point(1,1),
	Point(2,1),
	Point(-2,2),
	Point(-1,2),
	Point(0,2),
	Point(1,2),
	Point(2,2)
};
Point kernels[9] = {
	Point(-1, -1),
	Point(-1,0),
	Point(-1,1),
	Point(0, -1),
	Point(0,0),
	Point(0,1),
	Point(1, -1),
	Point(1,0),
	Point(1,1)
};

Mat src;
Mat shadow_seed;
Mat A;
Mat MS;
Mat ML;
vector<Point> seeds;
vector<Point> growing;
vector<double> distances;
vector<int> distances_norm;
vector<Point> Y_growing;
vector<Point> CbCr_growing;
vector<Point> contours;

Mat calcGrayHist(const Mat& img) {
	CV_Assert(img.type() == CV_8UC1);
	Mat hist;
	int channels[] = { 0 };	// 히스토그램 계산 채널
	int dims = 1;			// 출력 영상 차원
	const int histSize[] = { 256 };	// 각 차원의 히스토그램 빈 개수
	float graylevel[] = { 0,256 };	// GrayScale의 최솟값, 최댓값
	const float* ranges[] = { graylevel }; // 각 차원의 히스토그램 범위
	calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges);
	return hist;
}
Mat getGrayHistImage(const Mat& hist) {
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));
	double histMax;
	minMaxLoc(hist, 0, &histMax);
	Mat imgHist(100, 256, CV_8UC1, Scalar(255));
	for (int i = 0; i < 256; i++) {
		line(imgHist, Point(i, 100), Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0));
	}
	return imgHist;
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

double get_Point_distance(Point pt1, Point pt2) {
	return sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2));
}

Vec3b get_median() {			// 현재 평균값
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

double get_seed_Y() {
	double Y = 0;
	for (Point p : seeds) {
		Vec3b& color = src.at<Vec3b>(p);
		Y += 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0];
	}
	Y /= seeds.size();
	return Y;
}

void seed_growing(const int& threshold) {
	for (int c = 0; c < src.rows * src.cols; c++) {			// 히스토그램 계산 필수
		if (distances_norm[c] < threshold) {
			int y = c / src.cols;
			int x = c % src.cols;
			Point anchor = Point(x, y);
			for (Point seed : seeds) {
				if (get_Point_distance(anchor, seed) < 10) {
					growing.push_back(anchor);
					break;
				}
			}
		}
	}
	for (Point grow : growing) {
		shadow_seed.at<Vec3b>(grow) = src.at<Vec3b>(grow);
		seeds.push_back(grow);
	}
}

void all_pixel_calculate(Vec3b& seeds) {
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			/*double distance = 1 - RGB_distance(seeds, src.at<Vec3b>(j, i));
			distances.push_back(distance);*/
			//distances.push_back(Y_distance(seeds, src.at<Vec3b>(j, i)));
			double distance = 1 - CbCr_distance(seeds, src.at<Vec3b>(j, i));
			distances.push_back(distance);
		}
	}

	normalize_vector(distances, distances_norm);
	Mat dis = Mat(distances_norm);
	dis.convertTo(dis, CV_8UC1);
	Mat hist = calcGrayHist(dis);
	Mat hist_blurred;
	GaussianBlur(hist, hist_blurred, Size(), 5);
	seed_growing(get_threshold(hist_blurred));
}

void region_growing() {
	for (int i = 0; i < 3; i++) {
		double Y = get_seed_Y();
		Vec3b median_value = get_median();
		all_pixel_calculate(median_value);
		distances.clear();
		distances_norm.clear();
	}
}

void compare_Y(double Y) {
	for (int j = 0; j < ML.rows; j++) {
		for (int i = 0; i < ML.cols; i++) {
			Vec3b& color = ML.at<Vec3b>(j, i);
			if (Y_distance(Y, color) < 25) {
				Point anchor = Point(i, j);
				for (Point seed : seeds) {
					if (get_Point_distance(anchor, seed) < 10) {
						Y_growing.push_back(Point(i, j));
						break;
					}
				}
			}
		}
	}
}

double get_standard_deviation(Mat& mt) {
	return 0;
}
void compare_CbCr(Vec3b& seed) {
	for (int j = 0; j < ML.rows; j++) {
		for (int i = 0; i < ML.cols; i++) {
			Vec3b& color = ML.at<Vec3b>(j, i);
			double distance = 1 - CbCr_distance(seed, color);
			if (distance < 1e-4) {
				Point anchor = Point(i, j);
				for (Point seed : seeds) {
					if (get_Point_distance(anchor, seed) < 10) {
						CbCr_growing.push_back(Point(i, j));
						break;
					}
				}
			}
		}
	}
	for (Point p : CbCr_growing) {
		seeds.push_back(p);
	}
}


void get_A_MS() {
	A = src.clone();
	MS = shadow_seed.clone();
	subtract(A, MS, ML);
	double Y = get_seed_Y();
	Mat result = MS.clone();
	while(1){

		/*compare_Y(Y);
		for (Point grow : Y_growing) {
			MS.at<Vec3b>(grow) = A.at<Vec3b>(grow);
		}*/
		Vec3b median_value = get_median();
		compare_CbCr(median_value);
		for (Point grow : CbCr_growing) {
			MS.at<Vec3b>(grow) = A.at<Vec3b>(grow);
		}
		CbCr_growing.clear();
		subtract(A, MS, ML);
		
		Scalar mean_MS, std_dev_MS, mean_ML, std_dev_ML;
		meanStdDev(MS, mean_MS, std_dev_MS);
		meanStdDev(ML, mean_ML, std_dev_ML);
		double mean_sdv_MS = (std_dev_MS[0] + std_dev_MS[1] + std_dev_MS[2]) / 3;
		double mean_sdv_ML = (std_dev_ML[0] + std_dev_ML[1] + std_dev_ML[2]) / 3;
		cout << mean_sdv_MS << " " << mean_sdv_ML << endl;
		if (mean_sdv_MS > mean_sdv_ML) {
			break;
		}
		else {
			result.release();
			result = MS.clone();
			imshow("A", A);
			imshow("MS", MS);
			imshow("ML", ML);
			waitKey(10);
		}
	}
	imshow("A", A);
	imshow("MS", MS);
	imshow("ML", ML);
}


void get_seed(int event, int x, int y, int flags, void*) {
	static int seed_count = 0;
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		if (seed_count < 3) {
			Point anchor = Point(x, y);
			for (int i = 0; i < 25; i++) {
				Point pt = anchor + pointArray[i];
				shadow_seed.at<Vec3b>(pt) = src.at<Vec3b>(pt);
				seeds.push_back(pt);
			}
			seed_count++;
			if (seed_count == 3) {
				region_growing();
				imshow("shadow_seed", shadow_seed);
				get_A_MS();
			}
		}
		break;
	default:
		break;
	}

}

int main(void) {
#if ENTROPY == 1
	Mat src = imread("./src/shadow2.png");
	Size src_size = src.size();
	resize(src, src, Size(300, 400), 0, 0, INTER_LANCZOS4);
	imshow("src", src);
	vector<Point2f> lcs;
	vector<double> entropy;
	int angle;

	RGB2LCS(src, lcs);
	Mat LCS_M;
	drawLCS(lcs, LCS_M);
	getEntropy(lcs, entropy);
	/*Mat dst;
	drawEntropy(entropy, angle, dst);*/
	/*vector<int> gray_T;
	get_invariant(lcs, angle, gray_T);
	Mat invariant;
	drawInvariant(src, invariant, gray_T);*/
	//resize(invariant, invariant, src_size, 0, 0, INTER_CUBIC);
	//Mat edge;
	//Canny(invariant, edge, 5, 15);
	////morphologyEx(edge, edge, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
	//imshow("edge", edge);

	//Mat src_Gray;
	//resize(src, src, src_size, 0, 0, INTER_CUBIC);
	//cvtColor(src, src_Gray, COLOR_BGR2GRAY);
	//GaussianBlur(src_Gray, src_Gray, Size(), 1);
	//Mat src_edge;
	//Canny(src_Gray, src_edge, 20, 60);
	////morphologyEx(src_edge, src_edge, MORPH_CLOSE, Mat(), Point(-1,-1), 2);
	//imshow("src_edge", src_edge);

	//Mat shadow;
	//bitwise_xor(edge, src_edge, shadow);
	//imshow("shadow", shadow);
	waitKey();
	destroyAllWindows();

#endif
#if LAB == 1
	Mat src = imread("./src/shadow4.jpg");
	resize(src, src, Size(640, 480), 0, 0, INTER_CUBIC);
	imshow("src", src);
	Mat lab;
	cvtColor(src, lab, COLOR_BGR2Lab);		
	// -> L 0 ~ 255, A ->  1 ~ 255, B -> 1 ~ 255 
	vector<Mat> lab_planes;
	split(lab, lab_planes);
	Mat dst;
	LAB_Shadow(lab_planes, dst);

	waitKey();
	destroyAllWindows();
#endif
#if DETECTION == 1
	src = imread("./src/shadow5.png");
	resize(src, src, Size(300, 400), 0, 0, INTER_AREA);
	shadow_seed = Mat::zeros(src.size(), CV_8UC3);
	imshow("original", src);
	Mat src_filtered;
	bilateralFilter(src, src_filtered, -1, 10, 5);
	src = src_filtered;
	namedWindow("src");
	imshow("src", src);
	setMouseCallback("src", get_seed);
	waitKey();
	destroyAllWindows();
#endif
}
