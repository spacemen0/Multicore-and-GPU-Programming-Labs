/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu>

#include "support.h"

unsigned char average_kernel(skepu::Region2D<unsigned char> m, size_t elemPerPx)
{
	float scaling = 1.0 / ((m.oj / elemPerPx * 2 + 1) * (m.oi * 2 + 1));
	float res = 0;
	for (int y = -m.oi; y <= m.oi; ++y)
		for (int x = -m.oj; x <= m.oj; x += elemPerPx)
			res += m(y, x);
	return res * scaling;
}

unsigned char average_kernel_1d(skepu::Region1D<unsigned char> m, size_t elemPerPx)
{
	// your code here
	// float scaling_column = 1.0 / (m.oj / elemPerPx * 2 + 1);
	float scaling = 1.0 / (m.oi / elemPerPx * 2 + 1);

	float res = 0;

	for (int x = -m.oi; x <= m.oi; x += elemPerPx)
		res += m(x);

	// for (int x = -m.oj; x <= m.oj; x += elemPerPx)
	// 	res_row += m(y, x);
	return res * scaling;
}

unsigned char gaussian_kernel(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, size_t elemPerPx)
{
	float res = 0;

	for (int x = -m.oi; x <= m.oi; x += elemPerPx)
	{
		res += m(x) * stencil[(x + m.oi) / elemPerPx];
	}

	return res;
}

int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}

	LodePNGColorType colorType = LCT_RGB;
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);

	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFile = outputFileName + ss.str();

	// Read the padded image into a matrix. Create the output matrix without padding.
	// Padded version for 2D MapOverlap, non-padded for 1D MapOverlap
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrixPad = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> inputMatrix = ReadPngFileToMatrix(inputFileName, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	// more containers...?

	// Original version
	{
		auto conv = skepu::MapOverlap(average_kernel);
		conv.setOverlap(radius, radius * imageInfo.elementsPerPixel);

		auto timeTaken = skepu::benchmark::measureExecTime([&]
														   { conv(outputMatrix, inputMatrixPad, imageInfo.elementsPerPixel); });

		WritePngFileMatrix(outputMatrix, outputFile + "-average.png", colorType, imageInfo);
		std::cout << "Time for combined: " << (timeTaken.count() / 10E6) << "\n";
	}

	// Separable version
	// use conv.setOverlapMode(skepu::Overlap::[ColWise RowWise]);
	// and conv.setOverlap(<integer>)
	{
		auto conv = skepu::MapOverlap(average_kernel_1d);
		conv.setOverlapMode(skepu::Overlap::RowWise);
		conv.setOverlap(radius * imageInfo.elementsPerPixel);
		skepu::Matrix<unsigned char> temp(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);

		auto timeTaken = skepu::benchmark::measureExecTime([&]
														   { 
															conv(temp, inputMatrix, imageInfo.elementsPerPixel); 
															conv.setOverlapMode(skepu::Overlap::ColWise);
															conv.setOverlap(radius);
															conv(outputMatrix, temp, 1); });

		WritePngFileMatrix(outputMatrix, outputFile + "-separable.png", colorType, imageInfo);
		std::cout << "Time for separable: " << (timeTaken.count() / 10E6) << "\n";
	}

	// Separable gaussian
	{
		skepu::Vector<float> stencil = sampleGaussian(radius);

		// skeleton instance, etc here (remember to set backend)
		auto conv = skepu::MapOverlap(gaussian_kernel);
		conv.setOverlapMode(skepu::Overlap::RowWise);
		// conv.setOverlap(radius * imageInfo.elementsPerPixel);
		skepu::Matrix<unsigned char> temp(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);

		auto timeTaken = skepu::benchmark::measureExecTime([&]
														   {
															  conv(temp, inputMatrix, stencil, imageInfo.elementsPerPixel);
															  conv.setOverlapMode(skepu::Overlap::ColWise);
															//   conv.setOverlap(radius);
															  conv(outputMatrix, temp, stencil, 1); });

		WritePngFileMatrix(outputMatrix, outputFile + "-gaussian.png", colorType, imageInfo);
		std::cout << "Time for gaussian: " << (timeTaken.count() / 10E6) << "\n";
	}

	return 0;
}
