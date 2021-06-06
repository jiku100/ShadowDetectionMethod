#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
using namespace cv;
using namespace std;


Point kernels_5[25] =
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

Point kernels_3[9] = {
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
