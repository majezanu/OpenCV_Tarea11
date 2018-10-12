#include "Img.h"
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



Img::Img(cv::String path)
{
	mPath = path;
	mMat = imread(mPath);
}

Img::Img()
{

}

Img::~Img()
{
}

void Img::setMat(Mat mat)
{
	mMat = mat;
}
Mat Img::getMat()
{
	return mMat;
}

void Img::fig526_2()
{
	Mat src, srcGray, srcGrayF, srcPad, srcPadCen, srcPCC;
	Mat F_uv, Fr_uv, f_xy, fr_xy;
	Mat H_uv, H_uvC, H_BLPF_uv;
	Mat G_uv, Ffil_uv, g_xy;
	Mat dstG, dstFr, dst;

	cvtColor(mMat, srcGray, COLOR_BGR2GRAY);
	srcGray.convertTo(srcGrayF, CV_64F);

	copyMakeBorder(srcGrayF, srcPad, 0, srcGrayF.rows, 0, srcGrayF.cols, BORDER_WRAP, Scalar::all(0));

	/// Making place for complex values by adding a second layer to the centered padded image
	Mat planesSrcPad[] = { Mat_<double>(srcPad), Mat::zeros(srcPad.size(), CV_64F) };
	merge(planesSrcPad, 2, srcPCC);

	/// Computing discrete fourier transform
	dft(srcPCC, G_uv);

	/// Centering spectrum for multiply
	genCenterV2(G_uv, G_uv);

	/// Generating H(u,v)

	genTranslationH_uv(srcPCC.size(), H_uvC, 1, 0.1, 0.1);

	divSpectrums(G_uv, H_uvC, Fr_uv);

	GenButterworthFilter(Fr_uv.size(), H_BLPF_uv, 0, 0);

	mulSpectrums(Fr_uv, H_BLPF_uv, F_uv, 0, false);

	//Fr_uv = G_uv.clone();

	genCenterV2(Fr_uv, Fr_uv);
	genCenterV2(F_uv, F_uv);

	idft(Fr_uv, fr_xy, DFT_SCALE);
	dft(F_uv, f_xy, DFT_INVERSE);
	genCenterV2(F_uv, F_uv);
	genCenterV2(Fr_uv, Fr_uv);

	imComplexShow("Fig 5.26-2(b) Spectrum G_uv", G_uv, true, false);

	dstFr = fr_xy(Rect(0, 0, fr_xy.cols / 2, fr_xy.rows / 2)).clone();
	imComplexShow("Fig 5.26-2(c) Fr_uv recovered", dstFr, false, false);

	imComplexShow("Fig 5.26-2(d) Spectrum Fr_uv", Fr_uv, true, false);

	dst = f_xy(Rect(0, 0, f_xy.cols / 2, f_xy.rows / 2)).clone();
	imComplexShow("Fig 5.26-2(e) F_uv filtered with BLPF", dst, false, true);

	GenButterworthFilter(Fr_uv.size(), H_BLPF_uv, 250, 10);

	mulSpectrums(Fr_uv, H_BLPF_uv, Ffil_uv, 0, false);

	genCenterV2(Ffil_uv, Ffil_uv);

	dft(Ffil_uv, f_xy, DFT_INVERSE | DFT_SCALE);

	dst = f_xy(Rect(0, 0, f_xy.cols / 2, f_xy.rows / 2)).clone();
	imComplexShow("Fig 5.26-2(e) F_uv filtered with BLPF", dst, false, false);
	//imComplexShow("Computing magnitude of real and imag for displaying", WINDOW_KEEPRATIO, dst, false, false);
	genCenterV2(Ffil_uv, Ffil_uv);
	imComplexShow("Fig 5.26-2(f) Spectrum F_uv filtered", Ffil_uv, true, false);

	imComplexShow("Fig 5.26-2(f) Spectrum F_uv filtered", F_uv, true, false);
}

Mat Img::Image2Gamma( int C, int value_gamma)
{
	mMat.convertTo(mMat, CV_32F);
	pow(mMat, value_gamma / 10.0, mMat);
	mMat = C * mMat;
	normalize( mMat, mMat, 0, 255, NORM_MINMAX);
	convertScaleAbs(mMat, mMat);
	return mMat;
}

void Img::GenButterworthFilter(Size sz, Mat& dst, int D0, int order) {
	Mat aux = Mat::zeros(sz, CV_64F);
	int P = aux.rows;
	int Q = aux.cols;
	for (int u = 0; u < P; u++) {
		for (int v = 0; v < Q; v++) {
			double Duv = (double)sqrt(pow((u - P / 2), 2) + pow(v - Q / 2, 2));
			aux.at<double>(u, v) = 1 / (1 + powf(Duv / D0, 2 * order));
		}
	}

	Mat planes2[] = { Mat_<double>(aux), Mat::zeros(aux.size(), CV_64F) };
	Mat aux2;
	merge(planes2, 2, aux2);
	normalize(aux2, aux2, 0, 1, CV_MINMAX);
	dst = aux2;

}

Mat Img::Image2Negative() 
{
	Mat gray_image;
	cvtColor(mMat, gray_image, cv::COLOR_RGB2GRAY);
	gray_image = 255 - gray_image;
	cvtColor(gray_image, gray_image, cv::COLOR_GRAY2BGR);
	return gray_image;
}
Mat Img::Image2Log(int C)
{

	mMat.convertTo(mMat, CV_32F);
	log(mMat + 1, mMat);
	mMat = C * mMat;
	convertScaleAbs(mMat, mMat);
	normalize(mMat, mMat, 0, 255, NORM_MINMAX);
	return mMat;
}

Mat Img::contrastStretching(int r1, int s1, int r2, int s2)
{
	Mat clone = mMat.clone();
	//cvtColor(clone, gray_image, COLOR_BGR2GRAY);
	for (int y = 0; y <mMat.rows; y++) {
		for (int x = 0; x < mMat.cols; x++) {
			for (int c = 0; c < 3; c++) {
				int output = Img::computeContrast(mMat.at<Vec3b>(y, x)[c], r1, s1, r2, s2);
				clone.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(output);
			}
		}
	}
	return clone;
}

Mat Img::threshold( int k)
{
	Mat clone = mMat.clone();
	//cvtColor(clone, gray_image, COLOR_BGR2GRAY);

	for (int y = 0; y < mMat.rows; y++) {
		for (int x = 0; x < mMat.cols; x++) {
			for (int c = 0; c < 3; c++) {
				if (clone.at<Vec3b>(y, x)[c] <= k)
				{
					clone.at<Vec3b>(y, x)[c] =0;
				}
				else {
					clone.at<Vec3b>(y, x)[c] = 255;
				}
			}
		}
	}
	return clone;
}

Mat Img::cvHistogramEqu()
{
	Mat clone = mMat.clone();
	cvtColor(clone, clone, COLOR_BGR2GRAY);
	equalizeHist(clone, clone);
	cvtColor(clone, clone, COLOR_GRAY2BGR);
	return clone;
}

Mat Img::histogramProcessingManual()
{
	Mat origianl_image = mMat.clone();
	Mat src, src_gray, dst;
	std::vector<float> histogram(256, 0.0000);
	src = origianl_image.clone();


	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	src_gray.convertTo(dst, CV_32F);

	for (int i = 0; i < src_gray.rows; i++) {
		for (int j = 0; j < src_gray.cols; j++) {
			histogram.at(src_gray.at<uchar>(i, j)) += 1;
		}
	}

	for (int i = 1; i < 256; i++) {
		histogram.at(i) = histogram.at(i) + histogram.at(i - 1);
		histogram.at(i - 1) /= dst.total();
		if (i == 255)
		{
			histogram.at(i) /= dst.total();
		}
	}



	for (int i = 0; i < src_gray.rows; i++) {
		for (int j = 0; j < src_gray.cols; j++) {

			dst.at<float>(i, j) = histogram.at(src_gray.at<uchar>(i, j)) * dst.at<float>(i, j);

		}
	}
	normalize(dst, dst, 0, 255, NORM_MINMAX);		//la normalizamos a 256 niveles
	//convertScaleAbs(dst, dst);
	dst.convertTo(dst, CV_8UC1);					//Regresamos la imagen a 8-bit
	cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
	return dst;
}

Mat Img::cvEqualizeLocalHist(int nb)
{
	Mat clone = mMat.clone();
	cvtColor(clone, clone, COLOR_BGR2GRAY);
	Mat destiny = Mat::zeros(clone.size(), clone.type());
	Mat border;
	copyMakeBorder(clone, border, nb, nb, nb, nb, BORDER_CONSTANT);

	Mat kernel = Mat::zeros(2 * nb + 1, 2 * nb + 1, clone.type());
	for (int i = nb; i < clone.rows + nb; i++)
	{
		for (int j = nb; j < clone.cols + nb; j++)
		{
			for (int k = nb * -1; k <= nb; k++)
			{
				for (int n = nb * -1; n <= nb; n++)
				{
					kernel.at<uchar>(k + nb, n + nb) = border.at<uchar>(i + k, j + n);
				}
			}
			equalizeHist(kernel, kernel);
			destiny.at<uchar>(i - nb, j - nb) = kernel.at<uchar>(nb, nb);

		}
	}
	cvtColor(destiny, destiny, COLOR_GRAY2BGR);
	return destiny;



}

Mat Img::LocalHistorgramStatistics(int nb, float k0, float k1, float k2, float E) {
	cvtColor(mMat, mMat, COLOR_BGR2GRAY);
	Mat _dst = Mat::zeros(mMat.size(),mMat.type());

	float meanG = calcMean(mMat); 
	double stddG = sqrt(calcVar(mMat, meanG)); 

	Mat src_border;
	copyMakeBorder(mMat, src_border, nb, nb, nb, nb, BORDER_CONSTANT); 

	Mat kernel = Mat::zeros(2 * nb + 1, 2 * nb + 1, mMat.type()); 

	for (int i = nb; i < mMat.rows + nb; i++) {
		for (int j = nb; j < mMat.cols + nb; j++) {

			for (int k = nb * -1; k <= nb; k++) {
				for (int n = nb * -1; n <= nb; n++) {
					kernel.at<uchar>(k + nb, n + nb) = src_border.at<uchar>(i + k, j + n);
				}
			}

			float meanS = calcMean(kernel);
			float stddS = sqrt(calcVar(kernel, meanS));

			if ((meanS <= k0 * meanG) && (k1*stddG <= stddS) && (stddS <= k2 * stddG)) {
				_dst.at<uchar>(i - nb, j - nb) = kernel.at<uchar>(nb, nb)*E;
			}
			else {
				_dst.at<uchar>(i - nb, j - nb) = kernel.at<uchar>(nb, nb);
			}

		}
	}
	cvtColor(_dst, _dst, COLOR_GRAY2BGR);
	return _dst;
}

Mat Img::avgFilter(int nb) 
{
	cvtColor(mMat, mMat, COLOR_BGR2GRAY);
	Mat _dst = Mat::zeros(mMat.size(), mMat.type());	
	int _kern_size = 1 + 2 * nb;	
	Mat _kern = Mat::ones(_kern_size, _kern_size, CV_32F) / (float)(_kern_size * _kern_size); 
	Point _anchor = Point(nb, nb); 
	filter2D(mMat, _dst, -1, _kern, _anchor, 0, BORDER_DEFAULT); 

	cvtColor(_dst, _dst, COLOR_GRAY2BGR);
	return _dst;
}

Mat Img::medFilter(int nb) 
{

	cvtColor(mMat, mMat, COLOR_BGR2GRAY);
	
	Mat _dst = Mat::zeros(mMat.size(), mMat.type());	
	medianBlur(mMat, _dst, 2 * nb + 1);
	
	cvtColor(_dst, _dst, COLOR_GRAY2BGR);
	return _dst;
}

Mat Img::lapFilter(Mat _kern) {
	cvtColor(mMat, mMat, COLOR_BGR2GRAY);
	mMat.convertTo(mMat, CV_32F);
	Mat _dst = Mat::zeros(mMat.size(), mMat.type());	
	Point _anchor = Point(1, 1);
	filter2D(mMat, _dst, CV_32F, _kern, _anchor, 0, BORDER_DEFAULT); 
	
	return _dst;
}

Mat Img::lapFilterSCaled()
{
	double min, max;
	Mat dst;
	minMaxLoc(mMat, &min, &max);
	mMat.copyTo(dst);
	dst += min;
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8U);
	cvtColor(dst, dst, COLOR_GRAY2BGR);
	return dst;

}

Mat Img::sumLap(Mat lap)
{
	Mat dst;
	cvtColor(mMat, mMat, COLOR_BGR2GRAY);
	mMat.convertTo(mMat, CV_32F);
	dst = mMat - lap;
	dst.convertTo(dst, CV_8U);
	cvtColor(dst, dst, COLOR_GRAY2BGR);
	return dst;

}

Mat Img::fuzzyFilter(int _vd, int _vg, int _vb) { 
	cvtColor(mMat, mMat, COLOR_BGR2GRAY);
	Mat _src = mMat;
	Mat _dst = Mat::zeros(_src.size(), _src.type());
	float udark, ugray, ubright;
	for (int i = 0; i < _src.rows; i++) {
		for (int j = 0; j < _src.cols; j++) {
			udark = trapezoidalMembership(_src.at<uchar>(i, j), 0, 80, 0, 47);
			ugray = triangularMembership(_src.at<uchar>(i, j), 127, 47, 47);
			ubright = trapezoidalMembership(_src.at<uchar>(i, j), 174, 255, 47, 0);
			_dst.at<uchar>(i, j) = (udark*_vd + ugray * _vg + ubright * _vb) / (udark + ugray + ubright);
		}
	}
	
	return _dst;
}
//Aquí se le pasa el tipo del filtro donde 1 es el ideal, 2 es el butterwort y 3 es el gausiano.
//4 es el ideal pasa altas, 5 es el butterwort pasa altas, 6 es el gausiano pasa altas(filter_type)
//También se le pasa el radio para el filtro D0
Mat Img::LowHighPassFilter(int filter_type,int D0) 
{
	Mat src_gray, src_grayF, dst;
	Mat src_padded;
	Mat ILPF;
	Mat complexPad;

	cvtColor(mMat, src_gray, COLOR_BGR2GRAY);
	src_gray.convertTo(src_grayF, CV_32F);

	copyMakeBorder(src_grayF, src_padded, 0, src_grayF.rows, 0, src_grayF.cols, BORDER_CONSTANT, Scalar::all(0));
	for (int i = 0; i < src_padded.rows; i++) {
		for (int j = 0; j < src_padded.cols; j++) {
			src_padded.at<float>(i, j) = (float)(src_padded.at<float>(i, j)*pow(-1, (i + j)));
		}
	}
	Mat planes[] = { Mat_<float>(src_padded), Mat::zeros(src_padded.size(), CV_32F) };

	merge(planes, 2, complexPad);
	dft(complexPad, complexPad, DFT_COMPLEX_OUTPUT);
	switch (filter_type)
	{
	case 1:
		ILPF = GenIdealFilter(complexPad, D0);
		break;
	case 2:
		ILPF = GenButterworthFilter(complexPad, D0, 2);
		break;
	case 3:
		ILPF = GenGaussianFilter(complexPad, D0);
		break;
	case 4:
		ILPF = GenHighIdealFilter(complexPad, D0);
		break;
	case 5:
		ILPF = GenHighButterworthFilter(complexPad, D0, 2);
		break;
	case 6:
		ILPF = GenHighGaussianFilter(complexPad, D0);
		break;
	default:
		break;
	}
	
	Mat complexMasked;
	complexMasked = complexPad.mul(ILPF);

	dft(complexMasked, dst, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);


	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			dst.at<float>(i, j) = (float)(dst.at<float>(i, j)*pow(-1, (i + j)));
		}
	}
	dst.convertTo(dst, CV_8U);
	//convertScaleAbs(dst, dst);
	normalize(dst, dst, 0, 255, CV_MINMAX);


	// 6) Extracting M x N image
	Mat q0(dst, Rect(0, 0, src_grayF.cols, src_grayF.rows));
	return q0;
}
Mat Img::LaplacianFreq()
{
	Mat src_gray, src_grayF, dst;
	Mat src_padded;
	Mat LapH;
	Mat complexPad;

	cvtColor(mMat, src_gray, COLOR_BGR2GRAY);
	src_gray.convertTo(src_grayF, CV_32F);

	normalize(src_grayF, src_grayF, 1.0, 0, NORM_MINMAX);
	copyMakeBorder(src_grayF, src_padded, 0, src_grayF.rows, 0, src_grayF.cols, BORDER_CONSTANT, Scalar::all(0));
	
	for (int i = 0; i < src_padded.rows; i++) {
		for (int j = 0; j < src_padded.cols; j++) {
			src_padded.at<float>(i, j) = (float)(src_padded.at<float>(i, j)*pow(-1, (i + j)));
		}
	}

	Mat planes[] = { Mat_<float>(src_padded), Mat::zeros(src_padded.size(), CV_32F) };

	merge(planes, 2, complexPad);
	dft(complexPad, complexPad, DFT_COMPLEX_OUTPUT);

	LapH = GenLaplacianFreq(complexPad);
	Mat complexMasked;
	complexMasked = LapH.mul(complexPad);

	dft(complexMasked, dst, DFT_INVERSE | DFT_REAL_OUTPUT);
	double min = 0;
	double max = 0;
	minMaxLoc(dst, &min, &max);
	//Dividing Laplacian by it's maximum value as described on eq. 4.9-8
	dst = dst / max;
	dst = src_padded - dst;

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			dst.at<float>(i, j) = (float)(dst.at<float>(i, j)*pow(-1, (i + j)));
		}
	}
	//dst.convertTo(dst, CV_8U);
	//convertScaleAbs(dst, dst);
	
	
	Mat q0(dst, Rect(0, 0, src_grayF.cols, src_grayF.rows));
	//
convertScaleAbs(q0, q0);
	normalize(q0, q0, 0, 255, CV_MINMAX);
	q0.convertTo(q0, CV_8U);
	
	return q0;
}
Mat Img::FourierTransform()
{
	cvtColor(mMat, mMat, COLOR_BGR2GRAY);
	
	Mat padded;
	int m = getOptimalDFTSize(mMat.rows);
	int n = getOptimalDFTSize(mMat.cols);
	copyMakeBorder(mMat, padded, 0, m - mMat.rows, 0, n - mMat.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	magI += Scalar::all(1);
	log(magI, magI);
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	Mat q0(magI, Rect(0, 0, cx, cy));
	Mat q1(magI, Rect(cx, 0, cx, cy));
	Mat q2(magI, Rect(0, cy, cx, cy));
	Mat q3(magI, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(magI, magI, 0, 1, CV_MINMAX);
	//magI.convertTo(magI, CV_8U);
	
	return magI;
	
}

Mat Img::HomomorphicFilter()
{
	Mat srcGray, srcGrayF, srcPad, srcPadCen, srcPCC,Z_uv,H_uv, H_uvC,S_uv,dstS, dst, D_uv;

	cvtColor(mMat, srcGray, COLOR_BGR2GRAY);
	srcGray.convertTo(srcGrayF, CV_64F);
	copyMakeBorder(srcGrayF, srcPad, 0, srcGrayF.rows, 0, srcGrayF.cols, BORDER_CONSTANT, Scalar::all(0));
	srcPad += Scalar::all(1);
	log(srcPad, srcPad);

	srcPadCen = centerPad(srcPad);

	Mat planes[] = { Mat_<double>(srcPadCen), Mat::zeros(srcPadCen.size(), CV_64F) };
	merge(planes, 2, srcPCC);

	dft(srcPCC, Z_uv, DFT_COMPLEX_OUTPUT);

	D_uv = DuvGenerator(Z_uv);
	H_uv = HomomorphicGenerator(D_uv, 80, 0.25, 2, 1);
	Mat planes2[] = { Mat_<double>(H_uv), Mat_<double>(H_uv) };
	merge(planes2, 2, H_uvC);
	S_uv = H_uvC.mul(Z_uv);

	dft(S_uv, dstS, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	dstS = centerPad(dstS);
	exp(dstS, dstS);
	dstS -= Scalar::all(1);
	Mat q0(dstS, Rect(0, 0, mMat.cols, mMat.rows));
	normalize(q0, q0, 1, 0, CV_MINMAX);
	
	return q0;

}
void Img::SpectrumGenerator()
{
	Mat srcGray, srcGrayF, srcPad, srcPadCen, srcPCC;
	Mat Z_uv;
	Mat Hnr_uv, Hnr_uvC;
	Mat S_uv;
	Mat dstS, dstButXFou, dst;
	Mat Dpk_uv, Dmk_uv;
	Img toShow;
	cvtColor(mMat, srcGray, COLOR_BGR2GRAY);
	srcGray.convertTo(srcGrayF, CV_64F);

	copyMakeBorder(srcGrayF, srcPad, 0, srcGrayF.rows, 0, srcGrayF.cols, BORDER_CONSTANT, Scalar::all(0));
	
	srcPadCen = centerPad(srcPad);

	Mat planes[] = { Mat_<double>(srcPadCen), Mat::zeros(srcPadCen.size(), CV_64F) };
	merge(planes, 2, srcPCC);

	dft(srcPCC, Z_uv, DFT_COMPLEX_OUTPUT);

	Hnr_uv = Mat::ones(Z_uv.size(), CV_64F);
	Dpk_uv = Mat::zeros(Z_uv.size(), CV_64F);
	Dmk_uv = Mat::zeros(Z_uv.size(), CV_64F);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 77, 60);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 20);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 85, -54);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 20);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 158, 60);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 10);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 165, -54);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 10);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 238, 57);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 5);

	Mat planes2[] = { Mat_<double>(Hnr_uv), Mat_<double>(Hnr_uv) };
	merge(planes2, 2, Hnr_uvC);
	S_uv = Hnr_uvC.mul(Z_uv);

	dft(S_uv, dstS, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	dstS = centerPad(dstS);
	split(Z_uv, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	magI += Scalar::all(1);
	log(magI, magI);
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	normalize(magI, magI, 0, 1, CV_MINMAX);
	toShow.setMat(magI);
	toShow.Show("Spectrum");

	split(S_uv, planes);
	magnitude(planes[0], planes[1], planes[0]);
	magI = planes[0];
	magI += Scalar::all(1);
	log(magI, magI);
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	normalize(magI, magI, 0, 1, CV_MINMAX);
	toShow.setMat(magI);
	toShow.Show("Butterworth notch reject filter multiplied by fourier");

	Mat q0(dstS, Rect(0, 0, mMat.cols, mMat.rows));
	normalize(q0, q0, 0, 1, CV_MINMAX);
	toShow.setMat(q0);
	toShow.Show("Filtered image");

}
void Img::SpectrumGenerator2()
{
	Mat srcGray, srcGrayF, srcPad, srcPadCen, srcPCC;
	Mat Z_uv;
	Mat Hnr_uv, Hnr_uvC;
	Mat S_uv;
	Mat dstS, dstButXFou, dst;
	Mat Dpk_uv, Dmk_uv;
	Img toShow;
	cvtColor(mMat, srcGray, COLOR_BGR2GRAY);
	srcGray.convertTo(srcGrayF, CV_64F);

	copyMakeBorder(srcGrayF, srcPad, 0, srcGrayF.rows, 0, srcGrayF.cols, BORDER_CONSTANT, Scalar::all(0));

	srcPadCen = centerPad(srcPad);

	Mat planes[] = { Mat_<double>(srcPadCen), Mat::zeros(srcPadCen.size(), CV_64F) };
	merge(planes, 2, srcPCC);

	dft(srcPCC, Z_uv, DFT_COMPLEX_OUTPUT);

	Hnr_uv = Mat::ones(Z_uv.size(), CV_64F);
	Dpk_uv = Mat::zeros(Z_uv.size(), CV_64F);
	Dmk_uv = Mat::zeros(Z_uv.size(), CV_64F);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 62, 0);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 20);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 79, 0);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 20);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 104, 0);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 10);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 124, 0);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 10);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 131, 0);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 5);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 409, 0);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 5);

	genDk_uv(Z_uv, Dpk_uv, Dmk_uv, 593, 0);
	genHnrBF(Hnr_uv, Dpk_uv, Dmk_uv, 4, 5);

	Mat planes2[] = { Mat_<double>(Hnr_uv), Mat_<double>(Hnr_uv) };
	merge(planes2, 2, Hnr_uvC);
	S_uv = Hnr_uvC.mul(Z_uv);

	dft(S_uv, dstS, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	dstS = centerPad(dstS);
	split(Z_uv, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	magI += Scalar::all(1);
	log(magI, magI);
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	normalize(magI, magI, 0, 1, CV_MINMAX);
	toShow.setMat(magI);
	toShow.Show("Spectrum");

	split(S_uv, planes);
	magnitude(planes[0], planes[1], planes[0]);
	magI = planes[0];
	magI += Scalar::all(1);
	log(magI, magI);
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	normalize(magI, magI, 0, 1, CV_MINMAX);
	toShow.setMat(magI);
	toShow.Show("Butterworth notch reject filter multiplied by fourier");

	Mat q0(dstS, Rect(0, 0, mMat.cols, mMat.rows));
	normalize(q0, q0, 0, 1, CV_MINMAX);
	toShow.setMat(q0);
	toShow.Show("Filtered image");

}




Mat Img::GenIdealFilter(Mat complexPad, int D0) {
	Mat aux = Mat::zeros(complexPad.size(), CV_32F);
	int P = aux.rows - 1;
	int Q = aux.cols - 1;
	for (int u = 0; u < Q; u++) {
		for (int v = 0; v < P; v++) {
			float Duv = (float)sqrt(pow((u - P / 2), 2) + pow(v - Q / 2, 2));
			aux.at<float>(u, v) = (Duv <= D0 ? 1 : 0);
		}
	}

	Mat planes2[] = { Mat_<float>(aux), Mat_<float>(aux) };
	Mat aux2;
	merge(planes2, 2, aux2);

	return aux2;
}
Mat Img::GenButterworthFilter(Mat complexPad,int D0, int order) {
	Mat aux = Mat::zeros(complexPad.size(), CV_32F);
	int P = aux.rows;
	int Q = aux.cols;
	for (int u = 0; u < Q; u++) {
		for (int v = 0; v < P; v++) {
			float Duv = (float)sqrt(pow((u - (P - 1) / 2), 2) + pow(v - (Q - 1) / 2, 2));
			aux.at<float>(u, v) = 1 / (1 + powf(Duv / D0, 2 * order));
		}
	}

	Mat planes2[] = { Mat_<float>(aux), Mat_<float>(aux) };
	Mat aux2;
	merge(planes2, 2, aux2);

	return aux2;
}
Mat Img::GenGaussianFilter(Mat complexPad,int D0) {
	Mat aux = Mat::zeros(complexPad.size(), CV_32F);
	int P = aux.rows;
	int Q = aux.cols;
	for (int u = 0; u < Q; u++) {
		for (int v = 0; v < P; v++) {
			float Duv = (float)sqrt(pow((u - (P - 1) / 2), 2) + pow(v - (Q - 1) / 2, 2));
			aux.at<float>(u, v) = expf((-1 * powf(Duv, 2)) / (2 * powf(D0, 2)));
		}
	}

	Mat planes2[] = { Mat_<float>(aux), Mat_<float>(aux) };
	Mat aux2;
	merge(planes2, 2, aux2);

	return aux2;
}
Mat Img::GenHighIdealFilter(Mat complexPad,int D0) {
	Mat aux = Mat::zeros(complexPad.size(), CV_32F);
	int P = aux.rows;
	int Q = aux.cols;
	for (int u = 0; u < Q; u++) {
		for (int v = 0; v < P; v++) {
			float Duv = (float)sqrt(pow((u - (P - 1) / 2), 2) + pow(v - (Q - 1) / 2, 2));
			aux.at<float>(u, v) = (Duv <= D0 ? 0 : 1);
		}
	}

	Mat planes2[] = { Mat_<float>(aux), Mat_<float>(aux) };
	Mat aux2;
	merge(planes2, 2, aux2);

	return aux2;
}
Mat Img::GenHighButterworthFilter(Mat complexPad,int D0, int order) {
	Mat aux = Mat::zeros(complexPad.size(), CV_32F);
	int P = aux.rows;
	int Q = aux.cols;
	for (int u = 0; u < Q; u++) {
		for (int v = 0; v < P; v++) {
			float Duv = (float)sqrt(pow((u - (P - 1) / 2), 2) + pow(v - (Q - 1) / 2, 2));
			aux.at<float>(u, v) = 1 / (1 + powf(D0 / Duv, 2 * order));
		}
	}

	Mat planes2[] = { Mat_<float>(aux), Mat_<float>(aux) };
	Mat aux2;
	merge(planes2, 2, aux2);

	return aux2;
}
Mat Img::GenHighGaussianFilter(Mat complexPad, int D0) {
	Mat aux = Mat::zeros(complexPad.size(), CV_32F);
	int P = aux.rows;
	int Q = aux.cols;
	for (int u = 0; u < Q; u++) {
		for (int v = 0; v < P; v++) {
			float Duv = (float)sqrt(pow((u - (P - 1) / 2), 2) + pow(v - (Q - 1) / 2, 2));
			aux.at<float>(u, v) = 1 - expf((-1 * powf(Duv, 2)) / (2 * powf(D0, 2)));
		}
	}

	Mat planes2[] = { Mat_<float>(aux), Mat_<float>(aux) };
	Mat aux2;
	merge(planes2, 2, aux2);

	return aux2;
}
Mat Img::GenLaplacianFreq(Mat complexPad) {
	Mat aux = Mat::zeros(complexPad.size(), CV_32F);
	int P = aux.rows;
	int Q = aux.cols;
	for (int u = 0; u < P; u++) {
		for (int v = 0; v < Q; v++) {
			float Duv = (float)sqrt(pow((u - (P - 1) / 2), 2) + pow(v - (Q - 1) / 2, 2));
			aux.at<float>(u, v) = -4 * powf(CV_PI, 2)*powf(Duv, 2);
		}
	}

	Mat planes2[] = { Mat_<float>(aux), Mat_<float>(aux) };
	Mat aux2;
	merge(planes2, 2, aux2);

	return aux2;
}

Mat Img::laplacianFilter(Mat _src, Mat _kern) {
	Mat _dst = Mat::zeros(_src.size(), _src.type());
	filter2D(_src, _dst, CV_64F, _kern);
	return _dst;
}

Mat Img::centerPad(Mat src) {
	Mat aux = Mat::zeros(src.size(), CV_64F);
	int P = aux.cols;
	int Q = aux.rows;
	for (int i = 0; i < Q; i++) {
		for (int j = 0; j < P; j++) {
			aux.at<double>(i, j) = (double)(src.at<double>(i, j)*powf(-1, (i + j)));
		}
	}
	return aux;
}
Mat Img::DuvGenerator(Mat src) {
	Mat aux = Mat::zeros(src.size(), CV_64F);
	int P = aux.rows;
	int Q = aux.cols;
	for (int u = 0; u < P; u++) {
		for (int v = 0; v < Q; v++) {
			aux.at<double>(u, v) = (double)sqrt(powf(u - (P / 2), 2) + powf(v - (Q / 2), 2));
		}
	}
	return aux;
}

// Function to generate a gauss high pass filter using the generated D_uv
Mat Img::HomomorphicGenerator(Mat D_uv, int D0, double gammaL, double gammaH, double c) {
	Mat aux = Mat::zeros(D_uv.size(), CV_64F);
	int P = D_uv.rows;
	int Q = D_uv.cols;
	for (int u = 0; u < P; u++) {
		for (int v = 0; v < Q; v++) {
			aux.at<double>(u, v) = (gammaH - gammaL)*(1 - exp(-c * (powf(D_uv.at<double>(u, v), 2) / powf(D0, 2)))) + gammaL;
		}
	}
	return aux;
}

Mat Img::toRGB()
{
	Mat original, srcRGB;
	Mat rgb;
	original = mMat;
	Mat rgbPlanes[] = { Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F) };
	cvtColor(original, srcRGB, COLOR_BGR2RGB);
	split(srcRGB, rgbPlanes);
	blur(rgbPlanes[0], rgbPlanes[0], Size(5, 5));
	blur(rgbPlanes[1], rgbPlanes[1], Size(5, 5));
	blur(rgbPlanes[2], rgbPlanes[2], Size(5, 5));
	merge(rgbPlanes, 3, rgb);
	cvtColor(rgb, rgb, COLOR_RGB2BGR);
	return rgb;
}

Mat Img::toRGBHistogram()
{
	Mat original, srcRGB;
	Mat rgbEqu;

	original = mMat;

	Mat rgbPlanes[] = { Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F) };
	cvtColor(original, srcRGB, COLOR_BGR2RGB);
	split(srcRGB, rgbPlanes);
	equalizeHist(rgbPlanes[0], rgbPlanes[0]);
	equalizeHist(rgbPlanes[1], rgbPlanes[1]);
	equalizeHist(rgbPlanes[2], rgbPlanes[2]);
	merge(rgbPlanes, 3, rgbEqu);
	cvtColor(rgbEqu, rgbEqu, COLOR_RGB2BGR);

	return rgbEqu;
}

Mat Img::toHSIHistogram()
{
	Mat original,  srcHSI;
	Mat hsiEqu;

	original = mMat;
	Mat hsiPlanes[] = { Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F) };
	cvtColor(original, srcHSI, COLOR_BGR2HLS);
	split(srcHSI, hsiPlanes);
	equalizeHist(hsiPlanes[1], hsiPlanes[1]);
	merge(hsiPlanes, 3, hsiEqu);
	cvtColor(hsiEqu, hsiEqu, COLOR_HLS2BGR);

	return hsiEqu;

}

Mat Img::toHSI()
{
	Mat original, srcHSI;
	Mat hsi;
	original = mMat;
	Mat hsiPlanes[] = { Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F) };
	cvtColor(original, srcHSI, COLOR_BGR2HLS);
	split(srcHSI, hsiPlanes);
	blur(hsiPlanes[1], hsiPlanes[1], Size(5, 5));
	merge(hsiPlanes, 3, hsi);
	cvtColor(hsi, hsi, COLOR_HLS2BGR);

	return hsi;
}

Mat Img::toHSIKernels()
{
	Mat original, srcHSI;
	Mat hsi, lapA;
	original = mMat;
	Mat kernelA = (Mat_<float>(3, 3) <<
		0, -1, 0,
		-1, 4, -1,
		0, -1, 0);
	double min, max;


	Mat hsiPlanes[] = { Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F) };
	cvtColor(original, srcHSI, COLOR_BGR2HLS);
	split(srcHSI, hsiPlanes);

	hsiPlanes[1].convertTo(hsiPlanes[1], CV_64F);
	lapA = laplacianFilter(hsiPlanes[1], kernelA);
	hsiPlanes[1] = hsiPlanes[1] - lapA;
	hsiPlanes[1].convertTo(hsiPlanes[1], CV_8U);

	merge(hsiPlanes, 3, hsi);
	cvtColor(hsi, hsi, COLOR_HLS2BGR);

	return hsi;
}

void Img::genTranslationH_uv(Size size, Mat& dst, double T, double a, double b) {
	int P = size.height;
	int Q = size.width;
	int u0 = (P / 2); // P/2 = M -> u0 = M/2
	int v0 = (Q / 2); // Q/2 = N -> v0 = N/2

	Mat planes[] = { Mat::zeros(size, CV_64F), Mat::zeros(size, CV_64F) };
	for (int u = -P / 2; u < P / 2; u++) {
		for (int v = -Q / 2; v < Q / 2; v++) {


			double A = (T / (CV_PI*(u*a / 2 + v * b / 2))) * (sin(CV_PI*(u*a / 2 + v * b / 2)));
			if (cvIsNaN(A)) {
				planes[0].at<double>(u + u0, v + v0) = 1;
				planes[1].at<double>(u + u0, v + v0) = 0;
			}
			else if (abs(A) > 0.00000001) {
				double n = cos(-CV_PI * (u * a / 2 + v * b / 2));
				if (abs(n) < 0.00000001)
				{
					planes[0].at<double>(u + u0, v + v0) = 0;
				}
				else {
					planes[0].at<double>(u + u0, v + v0) = A * n;
				}

				n = sin(-CV_PI * (u * a / 2 + v * b / 2));
				if (abs(n) < 0.00000001)
				{
					planes[1].at<double>(u + u0, v + v0) = 0;
				}
				else {
					planes[1].at<double>(u + u0, v + v0) = A * n;
				}
			}
			else {
				planes[0].at<double>(u + u0, v + v0) = 0;
				planes[1].at<double>(u + u0, v + v0) = 0;
			}

		}
	}
	merge(planes, 2, dst);

}
void Img::genCenterV2(Mat&src, Mat&dst) { //Center on Frequency domain
	int cx = src.cols / 2;
	int cy = src.rows / 2;

	Mat q0(src, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(src, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(src, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(src, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void Img::divSpectrums(Mat& srcA, Mat& srcB, Mat& dst) {
	Mat den, num;

	mulSpectrums(srcB, srcB, den, 0, true);
	Mat planes[] = { Mat::zeros(srcB.size(), CV_64F), Mat::zeros(srcB.size(), CV_64F) };
	split(den, planes);
	planes[1] = planes[0];
	merge(planes, 2, den);

	mulSpectrums(srcA, srcB, num, 0, true);

	dst = num / den;
	int P = dst.rows;
	int Q = dst.cols;
	for (int u = 0; u < P; u++) {
		for (int v = 0; v < Q; v++) {
			double r = abs(srcB.at<Vec2d>(u, v)[0] <= 0.0000001);
			double im = abs(srcB.at<Vec2d>(u, v)[1] <= 0.0000001);
			if (abs(srcB.at<Vec2d>(u, v)[0]) <= 0.0000001 && abs(srcB.at<Vec2d>(u, v)[1]) <= 0.0000001) {
				dst.at<Vec2d>(u, v)[0] = srcA.at<Vec2d>(u, v)[0];
				dst.at<Vec2d>(u, v)[1] = srcA.at<Vec2d>(u, v)[1];
			}

		}
	}

}
void Img::imSpectrumShow(const String winname, Mat& src, int borderType) {
	Mat srcPaded, f_xy, F_uv;
	/// Padding Image
	copyMakeBorder(src, srcPaded, 0, src.rows, 0, src.cols, borderType, Scalar::all(0));

	/// Making place for complex values by adding a second layer to the centered padded image
	Mat planes[] = { Mat_<double>(srcPaded), Mat::zeros(srcPaded.size(), CV_64F) };
	merge(planes, 2, f_xy);

	/// Computing discrete fourier transform
	dft(f_xy, F_uv, DFT_COMPLEX_OUTPUT | DFT_SCALE);
	genCenterV2(F_uv, F_uv);
	imComplexShow(winname, F_uv, true, false);
}



void Img::imComplexShow(const String winname, Mat& src, bool logEnhance, bool realOnly) {
	Mat planes[] = { Mat::zeros(src.size(), CV_64F), Mat::zeros(src.size(), CV_64F) };
	split(src, planes);
	if (!realOnly) {
		magnitude(planes[0], planes[1], planes[0]);
	}
	Mat magI = planes[0];
	if (logEnhance) {
		magI += Scalar::all(1);
		log(magI, magI);
	}
	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols&-2, magI.rows&-2));
	normalize(magI, magI, 0, 1, CV_MINMAX);


	imshow(winname, planes[0]);
}

void Img::imComplexWrite(const String filename, Mat& src, bool realOnly) {
	Mat planes[] = { Mat::zeros(src.size(), CV_64F), Mat::zeros(src.size(), CV_64F) };
	split(src, planes);
	if (!realOnly) {
		magnitude(planes[0], planes[1], planes[0]);
	}
	Mat magI = planes[0];
	normalize(magI, magI, 0, 255, CV_MINMAX);
	magI.convertTo(magI, CV_8U);
	imwrite(filename, magI);
}

Mat Img::dif(Mat hsi, Mat rgb, int x)
{
	Mat resta, original;
	original = mMat;
	Mat Planes[] = { Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F) };
	resta = (hsi - rgb);
	cvtColor(resta, resta, COLOR_BGR2RGB);
	split(resta, Planes);
	resta = (Planes[0] + Planes[1] + Planes[2]) / 3;
	resta = resta +x;

	return resta;
}

Mat Img::toRGBKernels()
{
	Mat original, src_F, srcRGB, lapA;
	Mat rgb;

	original = mMat;
	Mat rgbPlanes[] = { Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F), Mat::zeros(original.size(), CV_64F) };
	Mat kernelA = (Mat_<float>(3, 3) <<
		0, -1, 0,
		-1, 4, -1,
		0, -1, 0);
	double min, max;

	cvtColor(original, srcRGB, COLOR_BGR2RGB);
	split(srcRGB, rgbPlanes);
	rgbPlanes[0].convertTo(rgbPlanes[0], CV_64F);
	lapA = laplacianFilter(rgbPlanes[0], kernelA);
	rgbPlanes[0] = rgbPlanes[0] - lapA;
	rgbPlanes[0].convertTo(rgbPlanes[0], CV_8U);


	rgbPlanes[1].convertTo(rgbPlanes[1], CV_64F);
	lapA = laplacianFilter(rgbPlanes[1], kernelA);
	rgbPlanes[1] = rgbPlanes[1] - lapA;
	rgbPlanes[1].convertTo(rgbPlanes[1], CV_8U);

	rgbPlanes[2].convertTo(rgbPlanes[2], CV_64F);
	lapA = laplacianFilter(rgbPlanes[2], kernelA);
	rgbPlanes[2] = rgbPlanes[2] - lapA;
	rgbPlanes[2].convertTo(rgbPlanes[2], CV_8U);

	merge(rgbPlanes, 3, rgb);
	cvtColor(rgb, rgb, COLOR_RGB2BGR);

	return rgb;
}

Mat Img::genTranslationH_uv(Size size, double T, double a, double b) {
	Mat dst;
	int P = size.height;
	int Q = size.width;
	int u0 = (P / 2); // P/2 = M -> u0 = M/2
	int v0 = (Q / 2); // Q/2 = N -> v0 = N/2

	Mat planes[] = { Mat::zeros(size, CV_64F), Mat::zeros(size, CV_64F) };
	for (int u = -P / 2; u < P / 2; u++) {
		for (int v = -Q / 2; v < Q / 2; v++) {


			double A = (T / (CV_PI*(u*a / 2 + v * b / 2))) * (sin(CV_PI*(u*a / 2 + v * b / 2)));
			if (cvIsNaN(A)) {
				planes[0].at<double>(u + u0, v + v0) = 1;
				planes[1].at<double>(u + u0, v + v0) = 0;
			}
			else if (abs(A) > 0.00000001) {
				double n = cos(-CV_PI * (u * a / 2 + v * b / 2));
				if (abs(n) < 0.00000001)
				{
					planes[0].at<double>(u + u0, v + v0) = 0;
				}
				else {
					planes[0].at<double>(u + u0, v + v0) = A * n;
				}

				n = sin(-CV_PI * (u * a / 2 + v * b / 2));
				if (abs(n) < 0.00000001)
				{
					planes[1].at<double>(u + u0, v + v0) = 0;
				}
				else {
					planes[1].at<double>(u + u0, v + v0) = A * n;
				}
			}
			else {
				planes[0].at<double>(u + u0, v + v0) = 0;
				planes[1].at<double>(u + u0, v + v0) = 0;
			}

		}
	}
	merge(planes, 2, dst);
	return dst;

}

void Img::fig526()
{
	Mat src, srcGray, srcGrayF, srcPad, srcPadCen, srcPCC;
	Mat F_uv, Fr_uv, f_xy, fr_xy;
	Mat H_uv, H_uvC;
	Mat G_uv, g_xy;
	Mat dstG, dstFr, dst;

	cvtColor(mMat, srcGray, COLOR_BGR2GRAY);
	srcGray.convertTo(srcGrayF, CV_64F);
	copyMakeBorder(srcGrayF, srcPad, 0, srcGrayF.rows, 0, srcGrayF.cols, BORDER_WRAP, Scalar::all(0));
	
	Mat planesSrcPad[] = { Mat_<double>(srcPad), Mat::zeros(srcPad.size(), CV_64F) };
	merge(planesSrcPad, 2, srcPCC);
	dft(srcPCC, F_uv);

	genCenterV2(F_uv, F_uv);
	genTranslationH_uv(F_uv.size(), H_uvC, 1, 0.1, 0.1);
	mulSpectrums(H_uvC, F_uv, G_uv, 0, false);
	
	divSpectrums(G_uv, H_uvC, Fr_uv);

	genCenterV2(G_uv, G_uv);
	genCenterV2(Fr_uv, Fr_uv);
	idft(G_uv, g_xy, DFT_SCALE);
	idft(Fr_uv, fr_xy, DFT_SCALE);

	dstG = g_xy(Rect(0, 0, g_xy.cols / 2, g_xy.rows / 2)).clone();
	imComplexShow("Distortion", dstG, false, true);

	imComplexWrite("Images/Fig526(Distorsion).bmp", dstG, true);

	imComplexShow("Original Spectrum", F_uv, true, false);

	genCenterV2(G_uv, G_uv);

	imComplexShow(" Distortion Spectrum", G_uv, true, false);

	dstFr = fr_xy(Rect(0, 0, fr_xy.cols / 2, fr_xy.rows / 2)).clone();
	imComplexShow("Recovered image from", dstFr, false, false);


}
void Img::genDk_uv(Mat src, Mat &Dpk, Mat &Dmk, int uk, int vk) {
	int P = src.rows - 1;
	int Q = src.cols - 1;

	for (int u = 0; u <= P; u++) {
		for (int v = 0; v <= Q; v++) {
			Dpk.at<double>(u, v) = sqrt(pow(u - (P / 2) - uk, 2) + pow(v - (Q / 2) - vk, 2));
			Dmk.at<double>(u, v) = sqrt(pow(u - (P / 2) + uk, 2) + pow(v - (Q / 2) + vk, 2));
		}
	}
}

void Img::genHnrBF(Mat& dst, Mat Dpk_uv, Mat Dmk_uv, int n, double D0k) {
	int P = dst.rows;
	int Q = dst.cols;

	for (int u = 0; u < P; u++) {
		for (int v = 0; v < Q; v++) {
			dst.at<double>(u, v) *= (1 / (1 + pow(D0k / Dpk_uv.at<double>(u, v), 2 * n))) * (1 / (1 + pow(D0k / Dmk_uv.at<double>(u, v), 2 * n)));
		}
	}

}

int Img::computeContrast(int point, int r1, int s1, int r2, int s2)
{
	int x = point;
	float result;
	if (0 <= x && x <= r1) {
		result = (x*s1) / (r1 + 1);
	}
	else if (r1 < x && x <= r2) {
		result = (x*(s2 - s1)) / (r2 - r1 + 1);
	}
	else if (r2 < x && x <= 255) {
		result = (x * (256 - s2) / (256 - r2)) + s2;
	}
	return (int)result;
}
float Img::calcMean(Mat _src) { 
	return (float)sum(_src)[0] / _src.total();
}
double Img::calcVar(Mat _src) { 
	float mean = calcMean(_src);
	return calcVar(_src, mean);
}
double Img::calcVar(Mat _src, float _mean) {
	Mat temp1 = _src;
	temp1.convertTo(temp1, CV_32F);
	pow(temp1 - _mean, 2, temp1);
	return (double)sum(temp1)[0] / temp1.total();
}
float Img::triangularMembership(float z, float a, float b, float c) 
{ 
	if (z >= (a - b) && z < a) {
		return (1 - (a - z) / b);
	}
	else if (z >= a && z <= a + c) {
		return (1 - (z - a) / c);
	}
	else {
		return 0;
	}
}
float Img::trapezoidalMembership(float z, float a, float b, float c, float d) { 
	if (z >= (a - c) && z < a) {
		return (1 - (a - z) / c);
	}
	else if (z >= a && z < b) {
		return 1;
	}
	else if (z >= b && z <= (b + d)) {
		return (1 - (z - b) / d);
	}
	else {
		return 0;
	}
}


void Img::Show(cv::String name)
{
	imshow(name, mMat);
}

Mat Img::Histogram() {
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	Mat hist;


	calcHist(&mMat, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8U, Scalar(0));
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	mMat = histImage;
	return mMat;


}

Mat Img::Const(int a, int b) {
	Mat uniform_noise = Mat::zeros(mMat.size(), CV_8U);
	int M = mMat.rows;
	int N = mMat.cols;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			uniform_noise.at<uchar>(i, j) = (b - a)*rand() / RAND_MAX + a;
		}
	}
	mMat = mMat + uniform_noise;

	return mMat;
}

Mat Img::SP(float Pa, float Pb) {
	Mat saltpepper_noise = Mat::zeros(mMat.size(), CV_8U);
	randu(saltpepper_noise, 0, 255);
	Mat black = saltpepper_noise < Pa*(255);
	Mat white = saltpepper_noise > (1 - Pb)*(255);
	//dst = mMa.clone();
	mMat.setTo(255, white);
	mMat.setTo(0, black);

	return mMat;
}

Mat Img::Exp(float a) {
	Mat exp_noise = Mat::zeros(mMat.size(), CV_32F);
	int M = mMat.rows;
	int N = mMat.cols;
	randu(exp_noise, 0, 1);
	log(exp_noise, exp_noise);
	exp_noise *= (1. / a);
	
	mMat.convertTo(mMat, CV_32F);
	normalize(mMat,mMat, 0, 1, CV_MINMAX);
	mMat = mMat + exp_noise;
	normalize(mMat,mMat, 0, 255, CV_MINMAX);
	mMat.convertTo(mMat, CV_8U);

	return mMat;
}

Mat Img::Rayleigh( int a, int b) {
	Mat rayleigh = Mat::zeros(mMat.size(), CV_32F);
	mMat.convertTo(mMat, CV_32F);
	int M = mMat.rows;
	int N = mMat.cols;
	randu(rayleigh, 0, 1);
	rayleigh = rayleigh * 255;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			if (rayleigh.at<float>(i, j) >= a)
			{
				rayleigh.at<float>(i, j) = (2 * (rayleigh.at<float>(i, j) - a) * exp(pow(-1 * (rayleigh.at<float>(i, j) - a), 2) / b)) / b;
			}
			else
			{
				rayleigh.at<float>(i, j) = 0;
			}
		}
	}



	mMat = mMat + rayleigh;
	
	mMat.convertTo(mMat, CV_8U);
	normalize(mMat,mMat, 0, 255, CV_MINMAX);

	return mMat;

}

Mat Img::EXP2( float a) {
	Mat src = mMat;
	Mat rayleigh = Mat::zeros(src.size(), CV_32F);
	src.convertTo(src, CV_32F);
	int M = src.rows;
	int N = src.cols;
	randu(rayleigh, 0, 255);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			if (rayleigh.at<float>(i, j) >= 0)
			{
				rayleigh.at<float>(i, j) = a * exp(-1 * (a*rayleigh.at<float>(i, j)));
			}
			else
			{
				rayleigh.at<float>(i, j) = 0;
			}
		}
	}



	mMat = src + rayleigh;
	mMat.convertTo(mMat, CV_8U);
	normalize(mMat,mMat, 0, 255, CV_MINMAX);

	return mMat;

}

Mat Img::gamma(int a, int b) {
	Mat src = mMat;
	Mat dst;
	Mat gamma = Mat::zeros(src.size(), CV_32F);
	src.convertTo(src, CV_32F);
	int M = src.rows;
	int N = src.cols;
	randu(gamma, 0, 1);
	gamma = gamma * 255;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			if (gamma.at<float>(i, j) >= a)
			{
				gamma.at<float>(i, j) = (pow(a, b) * pow(gamma.at<float>(i, j), b - 1) * exp(-1 * a *gamma.at<float>(i, j))) / factorial(b - 1);
			}
			else
			{
				gamma.at<float>(i, j) = 0;
			}
		}
	}



	dst = src + gamma;

	dst.convertTo(dst, CV_8U);
	normalize(dst, dst, 0, 255, CV_MINMAX);
	mMat = dst;
	return mMat;

}
Mat Img::Gauss(float m, float sigma) {
	Mat src = mMat;
	Mat dst;
	Mat gauss_noise = Mat::zeros(src.size(), CV_8U);
	randn(gauss_noise, m, sigma);
	dst = src + gauss_noise;
	normalize(dst, dst, 0, 255, CV_MINMAX);
	mMat = dst;
	return mMat;
}
int Img::factorial(int n) {
	int factorial = 1;
	for (int b = 1; b <= n; b++) {
		factorial = b * factorial;
	}
	return factorial;
}
