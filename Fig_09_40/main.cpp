//Este proyecto utiliza una clase llamada Img creada por Manuel Jesús Zavala Núñez
//Esta clase engloba a la clase de OpenCV Mat para hacer más sencilla la implementación de sus funciones

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Img.h" 
#include <iostream>
#include <stdlib.h>

using namespace cv;
using namespace std;

//Crear matriz de Opencv

//Crear variable de la clase Img que engloba a los objetos de OpenCv
Img original, modified;
Mat src, dif;

void fig640();
void fig641();
void withHistogram();
int main(int argc, char** argv)
{
	int myChoice = 0;

	std::cout << "********** Main Menu **********" << std::endl;
	std::cout << "(1): Figure 6.40" << std::endl;
	std::cout << "(2): Figure 6.41" << std::endl;
	std::cout << "(3): RGB && HSI with Histograms" << std::endl;
	std::cout << "********** Main Menu **********" << std::endl;
	std::cin >> myChoice;
	switch (myChoice)
	{
	case 1:
		fig640();
		break;
	case 2:
		fig641();
		break;
	case 3:
		withHistogram();
		break;
	default:
		std::cout << "ERROR! You have selected an invalid choice." << std::endl;
		break;
	}
		

	waitKey(0);

	return 0;
}

void fig641()
{
	src = imread("Images/Fig0638(a)(lenna_RGB).tif");
	original.setMat(src);
	original.Show("Original");
	modified.setMat(original.toRGB());

	modified.Show("RGB");
	modified.setMat(original.toHSI());

	modified.Show("HSI");

	modified.setMat(original.dif(original.toHSI(), original.toRGB(),150));
	modified.Show("Diference");
}

void fig640()
{
	src = imread("Images/Fig0638(a)(lenna_RGB).tif");
	original.setMat(src);
	original.Show("Original");
	modified.setMat(original.toRGBKernels());
	modified.Show("RGB Kernels");

	modified.setMat(original.toHSIKernels());

	modified.Show("HSI Kernels");

	modified.setMat(original.dif(original.toHSIKernels(), original.toRGBKernels(), 150));
	modified.Show("Diference");
}

void withHistogram()
{
	src = imread("Images/fire.jpg");
	original.setMat(src);
	original.Show("Original");
	modified.setMat(original.toRGBHistogram());
	modified.Show("RGB with Histogram");

	modified.setMat(original.toHSIHistogram());

	modified.Show("HSI with Histogram");
}



