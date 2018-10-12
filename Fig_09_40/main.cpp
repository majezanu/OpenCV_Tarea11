//Este proyecto utiliza una clase llamada Img creada por Manuel Jesús Zavala Núñez
//Esta clase engloba a la clase de OpenCV Mat para hacer más sencilla la implementación de sus funciones

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Img.h" 
#include <iostream>
#include <stdlib.h>


Mat imagethreeshold;
Mat imagethreeshold2;


Img img;
int main(int argc, char** argv) {
	Mat image;
	image = imread("images/Fig0940(a)(rice_image_with_intensity_gradient).tif", CV_LOAD_IMAGE_GRAYSCALE);
	img.setMat(image);
	img.Show("a) Original");

	img.setMat(img.thresHold());
	img.Show("b)thresholded image");

	img.setMat(image);
	img.setMat(img.MorphologicFcn(2, 30));
	img.Show("c)image opened");

	img.setMat(image);
	img.setMat(img.MorphologicFcn(5, 20));
	img.Show("d) top hat transformation");

	img.setMat(img.thresHold());
	img.Show("e) top hat threshold");

	

	waitKey(0);
	return 0;
}




