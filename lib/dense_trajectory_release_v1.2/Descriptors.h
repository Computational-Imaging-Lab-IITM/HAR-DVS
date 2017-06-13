#ifndef DESCRIPTORS_H_
#define DESCRIPTORS_H_

#include "DenseTrack.h"
using namespace cv;

// get the rectangle for computing the descriptor
void GetRect(const Point2f& point, RectInfo& rect, const int width, const int height, const DescInfo& descInfo)
{
	int x_min = descInfo.width/2;
	int y_min = descInfo.height/2;
	int x_max = width - descInfo.width;
	int y_max = height - descInfo.height;

	rect.x = std::min<int>(std::max<int>(cvRound(point.x) - x_min, 0), x_max);
	rect.y = std::min<int>(std::max<int>(cvRound(point.y) - y_min, 0), y_max);
	rect.width = descInfo.width;
	rect.height = descInfo.height;
}

// compute integral histograms for the whole image
void BuildDescMat(const Mat& xComp, const Mat& yComp, float* desc, const DescInfo& descInfo)
{
	float maxAngle = 360.f;
	int nDims = descInfo.nBins;
	// one more bin for hof
	int nBins = descInfo.isHof ? descInfo.nBins-1 : descInfo.nBins;
	const float angleBase = float(nBins)/maxAngle;

	int step = (xComp.cols+1)*nDims;
	int index = step + nDims;
	for(int i = 0; i < xComp.rows; i++, index += nDims) {
		const float* xc = xComp.ptr<float>(i);
		const float* yc = yComp.ptr<float>(i);

		// summarization of the current line
		std::vector<float> sum(nDims);
		for(int j = 0; j < xComp.cols; j++) {
			float x = xc[j];
			float y = yc[j];
			float mag0 = sqrt(x*x + y*y);
			float mag1;
			int bin0, bin1;

			// for the zero bin of hof
			if(descInfo.isHof && mag0 <= min_flow) {
				bin0 = nBins; // the zero bin is the last one
				mag0 = 1.0;
				bin1 = 0;
				mag1 = 0;
			}
			else {
				float angle = fastAtan2(y, x);
				if(angle >= maxAngle) angle -= maxAngle;

				// split the mag to two adjacent bins
				float fbin = angle * angleBase;
				bin0 = cvFloor(fbin);
				bin1 = (bin0+1)%nBins;

				mag1 = (fbin - bin0)*mag0;
				mag0 -= mag1;
			}

			sum[bin0] += mag0;
			sum[bin1] += mag1;

			for(int m = 0; m < nDims; m++, index++)
				desc[index] = desc[index-step] + sum[m];
		}
	}
}

// get a descriptor from the integral histogram
void GetDesc(const DescMat* descMat, RectInfo& rect, DescInfo descInfo, std::vector<float>& desc, const int index)
{
	int dim = descInfo.dim;
	int nBins = descInfo.nBins;
	int height = descMat->height;
	int width = descMat->width;

	int xStride = rect.width/descInfo.nxCells;
	int yStride = rect.height/descInfo.nyCells;
	int xStep = xStride*nBins;
	int yStep = yStride*width*nBins;

	// iterate over different cells
	int iDesc = 0;
	std::vector<float> vec(dim);
	for(int xPos = rect.x, x = 0; x < descInfo.nxCells; xPos += xStride, x++)
	for(int yPos = rect.y, y = 0; y < descInfo.nyCells; yPos += yStride, y++) {
		// get the positions in the integral histogram
		const float* top_left = descMat->desc + (yPos*width + xPos)*nBins;
		const float* top_right = top_left + xStep;
		const float* bottom_left = top_left + yStep;
		const float* bottom_right = bottom_left + xStep;

		for(int i = 0; i < nBins; i++) {
			float sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
			vec[iDesc++] = std::max<float>(sum, 0) + epsilon;
		}
	}

	float norm = 0;
	for(int i = 0; i < dim; i++)
		norm += vec[i];
	if(norm > 0) norm = 1./norm;

	int pos = index*dim;
	for(int i = 0; i < dim; i++)
		desc[pos++] = sqrt(vec[i]*norm);
}

// for HOG descriptor
void HogComp(const Mat& img, float* desc, DescInfo& descInfo)
{
	Mat imgX, imgY;
	Sobel(img, imgX, CV_32F, 1, 0, 1);
	Sobel(img, imgY, CV_32F, 0, 1, 1);
	BuildDescMat(imgX, imgY, desc, descInfo);
}

// for HOF descriptor
void HofComp(const Mat& flow, float* desc, DescInfo& descInfo)
{
	Mat flows[2];
	split(flow, flows);
	BuildDescMat(flows[0], flows[1], desc, descInfo);
}

// for MBH descriptor
void MbhComp(const Mat& flow, float* descX, float* descY, DescInfo& descInfo)
{
	Mat flows[2];
	split(flow, flows);

	Mat flowXdX, flowXdY, flowYdX, flowYdY;
	Sobel(flows[0], flowXdX, CV_32F, 1, 0, 1);
	Sobel(flows[0], flowXdY, CV_32F, 0, 1, 1);
	Sobel(flows[1], flowYdX, CV_32F, 1, 0, 1);
	Sobel(flows[1], flowYdY, CV_32F, 0, 1, 1);

	BuildDescMat(flowXdX, flowXdY, descX, descInfo);
	BuildDescMat(flowYdX, flowYdY, descY, descInfo);
}

// check whether a trajectory is valid or not
bool IsValid(std::vector<Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length)
{
	int size = track.size();
	float norm = 1./size;
	for(int i = 0; i < size; i++) {
		mean_x += track[i].x;
		mean_y += track[i].y;
	}
	mean_x *= norm;
	mean_y *= norm;

	for(int i = 0; i < size; i++) {
		float temp_x = track[i].x - mean_x;
		float temp_y = track[i].y - mean_y;
		var_x += temp_x*temp_x;
		var_y += temp_y*temp_y;
	}
	var_x *= norm;
	var_y *= norm;
	var_x = sqrt(var_x);
	var_y = sqrt(var_y);

	// remove static trajectory
	if(var_x < min_var && var_y < min_var)
		return false;
	// remove random trajectory
	if( var_x > max_var || var_y > max_var )
		return false;

	float cur_max = 0;
	for(int i = 0; i < size-1; i++) {
		track[i] = track[i+1] - track[i];
		float temp = sqrt(track[i].x*track[i].x + track[i].y*track[i].y);

		length += temp;
		if(temp > cur_max)
			cur_max = temp;
	}

	if(cur_max > max_dis && cur_max > length*0.7)
		return false;

	track.pop_back();
	norm = 1./length;
	// normalize the trajectory
	for(int i = 0; i < size-1; i++)
		track[i] *= norm;

	return true;
}

// detect new feature points in an image without overlapping to previous points
void DenseSample(const Mat& grey, std::vector<Point2f>& points, const double quality, const int min_distance)
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);

	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters(width*height);
	int x_max = min_distance*width;
	int y_max = min_distance*height;

	for(int i = 0; i < points.size(); i++) {
		Point2f point = points[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);

		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters[y*width+x]++;
	}

	points.clear();
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters[index] > 0)
			continue;

		int x = j*min_distance+offset;
		int y = i*min_distance+offset;

		if(eig.at<float>(y, x) > threshold)
			points.push_back(Point2f(float(x), float(y)));
	}
}

void InitPry(const Mat& frame, std::vector<float>& scales, std::vector<Size>& sizes)
{
	int rows = frame.rows, cols = frame.cols;
	float min_size = std::min<int>(rows, cols);

	int nlayers = 0;
	while(min_size >= patch_size) {
		min_size /= scale_stride;
		nlayers++;
	}

	if(nlayers == 0) nlayers = 1; // at least 1 scale 

	scale_num = std::min<int>(scale_num, nlayers);

	scales.resize(scale_num);
	sizes.resize(scale_num);

	scales[0] = 1.;
	sizes[0] = Size(cols, rows);

	for(int i = 1; i < scale_num; i++) {
		scales[i] = scales[i-1] * scale_stride;
		sizes[i] = Size(cvRound(cols/scales[i]), cvRound(rows/scales[i]));
	}
}

void BuildPry(const std::vector<Size>& sizes, const int type, std::vector<Mat>& grey_pyr)
{
	int nlayers = sizes.size();
	grey_pyr.resize(nlayers);

	for(int i = 0; i < nlayers; i++)
		grey_pyr[i].create(sizes[i], type);
}

void DrawTrack(const std::vector<Point2f>& point, const int index, const float scale, Mat& image)
{
	Point2f point0 = point[0];
	point0 *= scale;

	for (int j = 1; j <= index; j++) {
		Point2f point1 = point[j];
		point1 *= scale;

		line(image, point0, point1, Scalar(0,cvFloor(255.0*(j+1.0)/float(index+1.0)),0), 2, 8, 0);
		point0 = point1;
	}
	circle(image, point0, 2, Scalar(0,0,255), -1, 8, 0);
}

void PrintDesc(std::vector<float>& desc, DescInfo& descInfo, TrackInfo& trackInfo)
{
	int tStride = cvFloor(trackInfo.length/descInfo.ntCells);
	float norm = 1./float(tStride);
	int dim = descInfo.dim;
	int pos = 0;
	for(int i = 0; i < descInfo.ntCells; i++) {
		std::vector<float> vec(dim);
		for(int t = 0; t < tStride; t++)
			for(int j = 0; j < dim; j++)
				vec[j] += desc[pos++];
		for(int j = 0; j < dim; j++)
			printf("%.7f\t", vec[j]*norm);
	}
}

#endif /*DESCRIPTORS_H_*/
