#ifndef DENSETRACK_H_
#define DENSETRACK_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <ctype.h>
#include <unistd.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

using namespace cv;

int start_frame = 0;
int end_frame = INT_MAX;
int scale_num = 8;
const float scale_stride = sqrt(2);

// parameters for descriptors
int patch_size = 32;
int nxy_cell = 2;
int nt_cell = 3;
float epsilon = 0.05;
const float min_flow = 0.4;

// parameters for tracking
double quality = 0.001;
int min_distance = 5;
int init_gap = 1;
int track_length = 15;

// parameters for rejecting trajectory
const float min_var = sqrt(3);
const float max_var = 50;
const float max_dis = 20;

typedef struct {
	int x;       // top left corner
	int y;
	int width;
	int height;
}RectInfo;

typedef struct {
    int width;   // resolution of the video
    int height;
    int length;  // number of frames
}SeqInfo;

typedef struct {
    int length;  // length of the trajectory
    int gap;     // initialization gap for feature re-sampling 
}TrackInfo;

typedef struct {
    int nBins;   // number of bins for vector quantization
    bool isHof; 
    int nxCells; // number of cells in x direction
    int nyCells; 
    int ntCells;
    int dim;     // dimension of the descriptor
    int height;  // size of the block for computing the descriptor
    int width;
}DescInfo; 

// integral histogram for the descriptors
typedef struct {
    int height;
    int width;
    int nBins;
    float* desc;
}DescMat;

class Track
{
public:
    std::vector<Point2f> point;
    std::vector<float> hog;
    std::vector<float> hof;
    std::vector<float> mbhX;
    std::vector<float> mbhY;
    int index;

    Track(const Point2f& point_, const TrackInfo& trackInfo, const DescInfo& hogInfo,
          const DescInfo& hofInfo, const DescInfo& mbhInfo)
        : point(trackInfo.length+1), hog(hogInfo.dim*trackInfo.length),
          hof(hofInfo.dim*trackInfo.length), mbhX(mbhInfo.dim*trackInfo.length), mbhY(mbhInfo.dim*trackInfo.length)
    {
        index = 0;
        point[0] = point_;
    }

    void addPoint(const Point2f& point_)
    {
        index++;
        point[index] = point_;
    }
};

#endif /*DENSETRACK_H_*/
