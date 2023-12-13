/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>
#include <skepu>
#include "support.h"

unsigned char median_kernel(skepu::Region2D<unsigned char> image, size_t elemPerPx)
{
	size_t size = (image.oi + 1) * ((2* image.oj)/elemPerPx)+1;
	unsigned char values [2048];

	for (int y = -image.oi; y <= image.oi; ++y)
	{
		for (int x = -image.oj; x <= image.oj; x += elemPerPx)
		{
			values[(y+image.oi) * image.oj/elemPerPx + (x + image.oj)/elemPerPx] = image(y, x);
		}
	}

	for (size_t i = 0; i < size - 1; ++i)
	{
		size_t minIndex = i;
		for (size_t j = i + 1; j < size; ++j)
		{
			if (values[j] < values[minIndex])
			{
				minIndex = j;
			}
		}
		auto temp = values[i];
		values[i]=values[minIndex];
		values[minIndex] = temp;
	}

	unsigned char median;
	if (size % 2 == 0)
	{
		median = (values[size / 2 - 1] + values[size / 2]) / 2;
	}
	else
	{
		median = values[size / 2];
	}

	return median;
}

int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;
	
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}
	
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";
		
	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	
	// Skeleton instance
	auto calculateMedian = skepu::MapOverlap(median_kernel);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
	auto timeTaken = skepu::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);
	
	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";
	
	return 0;
}


