#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cvstd.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
//using namespace System;

class CV_EXPORTS Img : Mat
{
public:

	Img(cv::String path);
	Img();
	virtual ~Img();

	void setMat(Mat mat);
	Mat getMat();



	Mat Image2Gamma(int C, int value_gamma);
	Mat Image2Negative();
	Mat Image2Log(int C);
	Mat contrastStretching(int r1, int s1, int r2, int s2);
	Mat threshold(int k);
	Mat cvHistogramEqu();
	Mat histogramProcessingManual();
	Mat cvEqualizeLocalHist(int nb);
	Mat LocalHistorgramStatistics(int nb, float k0, float k1, float k2, float E);
	Mat avgFilter(int nb);
	Mat medFilter(int nb);
	Mat lapFilter(Mat _kern);
	Mat lapFilterSCaled();
	Mat sumLap(Mat lap);
	Mat fuzzyFilter(int _vd, int _vg, int _vb);
	Mat LowHighPassFilter(int filter_type, int d0);
	Mat LaplacianFreq();

	Mat GenGaussianFilter(Mat complexPad, int D0);
	Mat GenIdealFilter(Mat complexPad, int D0);
	Mat GenButterworthFilter(Mat complexPad, int D0, int order);
	Mat GenHighIdealFilter(Mat complexPad, int D0);
	Mat GenHighButterworthFilter(Mat complexPad, int D0, int order);
	Mat GenHighGaussianFilter(Mat complexPad, int D0);
	Mat GenLaplacianFreq(Mat complexPad);
	Mat FourierTransform();
	Mat HomomorphicFilter();
	Mat toRGB();
	Mat toHSI();
	Mat dif(Mat hsi, Mat rgb, int x);
	Mat toRGBKernels();
	Mat toHSIKernels();
	Mat toRGBHistogram();
	Mat toHSIHistogram();

	void  SpectrumGenerator();
	void  SpectrumGenerator2();


	void genHnrBF(Mat& dst, Mat Dpk_uv, Mat Dmk_uv, int n, double D0k);
	void genDk_uv(Mat src, Mat &Dpk, Mat &Dmk, int uk, int vk);
	Mat centerPad(Mat src);
	Mat DuvGenerator(Mat src);
	Mat HomomorphicGenerator(Mat D_uv, int D0, double gammaL, double gammaH, double c);
	Mat laplacianFilter(Mat _src, Mat _kern);
	void genTranslationH_uv(Size sz, Mat& dst, double T, double a, double b);
	void genCenterV2(Mat&src, Mat&dst);
	void divSpectrums(Mat& srcA, Mat& srcB, Mat& dst);
	void imSpectrumShow(const String winname, Mat& src, int borderType);
	void imComplexShow(const String winname, Mat& src, bool logEnhance, bool realOnly);
	void imComplexWrite(const String filename, Mat& src, bool realOnly);
	void fig526();
	void fig526_2();
	void refreshFilteredImage(int, void*);
	void GenButterworthFilter(Size sz, Mat& dst, int D0, int order);
	//Mat imComplexWrite(const String filename, Mat src, bool realOnly);
	Mat genTranslationH_uv(Size size,  double T, double a, double b);

	int computeContrast(int point, int r1, int s1, int r2, int s2);
	float calcMean(Mat _src);
	double calcVar(Mat _src);
	double calcVar(Mat _src, float _mean);
	float triangularMembership(float z, float a, float b, float c);
	float trapezoidalMembership(float z, float a, float b, float c, float d);
	void Show(cv::String name);

	Mat Histogram();
	Mat Const(int a, int b);

	Mat Exp(float a);
	Mat SP( float Pa, float Pb);
	Mat EXP2( float a);
	Mat Rayleigh(int a, int b);
	Mat gamma( int a, int b);
	Mat Gauss( float m, float sigma);
	int factorial(int n);


private: 
	Mat mMat;
	cv::String mPath;
	
};