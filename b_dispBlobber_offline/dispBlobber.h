/*
 * Copyright (C) 2015 iCub Facility - Istituto Italiano di Tecnologia
 * Authors: Tanis Mar, Giulia Pasquale
 * email:  tanis.mar@iit.it, giulia.pasquale@iit.it
 * Permission is granted to copy, distribute, and/or modify this program
 * under the terms of the GNU General Public License, version 2 or any
 * later version published by the Free Software Foundation.
 *
 * A copy of the license can be found at
 * http://www.robotcub.org/icub/license/gpl.txt
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details
 */

#ifndef __NEARBLOBBER_H__
#define __NEARBLOBBER_H__

#include <string>
#include <vector>
#include <numeric>

// OpenCV
#include <opencv/highgui.h>
#include <opencv/cv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/calib3d/calib3d.hpp"

class nearBlobber
{

	int margin;

	int backgroundThresh;
    int frontThresh;

    int minBlobSize;
    int gaussSize;
    
    int imageThreshRatioLow;
    int imageThreshRatioHigh;
    
    cv::Scalar blue, green, red, white;

    cv::Mat aux, fillMask;

    std::vector<cv::Point2f > center2DcoordsBuffer;
    std::vector<cv::Point2f > topleft2DcoordsBuffer;
    std::vector<cv::Point2f > bottomright2DcoordsBuffer;
    int bufferSize;

    cv::Point mean_center;
    cv::Point mean_topleft;
    cv::Point mean_bottomright;

public:

    nearBlobber(int imH, int imW, int _centroidBufferSize,
    		int _margin,
    		int _backgroundThresh, int _frontThresh,
    		int _minBlobSize, int _gaussSize,
    		int _dispThreshRatioLow, int _dispThreshRatioHigh);

    bool setThresh(int low, int high);
    bool setMargin(int mrg);

   void extractBlob(std::vector<cv::Mat> &images, std::vector<int> &roi, std::vector<int> &centroid, cv::Mat &blob, double *t);
       
};

#endif
