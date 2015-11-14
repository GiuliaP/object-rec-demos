/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libelas.
Authors: Andreas Geiger

libelas is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libelas is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <fstream>

// OpenCV
#include <opencv/highgui.h>
#include <opencv/cv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include "elasWrapper.h"

using namespace cv;
using namespace std;

bool loadExtrinsics(Mat& Ro, Mat& To)
{

    Ro = Mat::zeros(3,3,CV_64FC1);
    To = Mat::zeros(3,1,CV_64FC1);

    double pXoarr[] = {0.999644, -0.013273, -0.0231613, -0.0679876, 0.0142311, 0.999029, 0.0417045, -0.0005149, 0.0225852, -0.0420192, 0.998861, 0.00129894, 0, 0, 0, 1};
    std::vector<double> pXo (pXoarr, pXoarr + sizeof(pXoarr) / sizeof(double));

    for (int i=0; i<(pXo.size()-4); i+=4)
    {
        Ro.at<double>(i/4,0)=pXo[i];
        Ro.at<double>(i/4,1)=pXo[i+1];
        Ro.at<double>(i/4,2)=pXo[i+2];
        To.at<double>(i/4,0)=pXo[i+3];
    }

    return true;
}

bool loadIntrinsics(Mat &KL, Mat &KR, Mat &DistL, Mat &DistR)
{

    // right

    double fx = 413.169;
    double fy = 412.434;

    double cx = 326.806;
    double cy = 228.968;

    double k1 = -0.390229;
    double k2 = 0.141388;

    double p1 = 2.37523e-05;
    double p2 = -0.00148838;

    DistR = Mat::zeros(1,8,CV_64FC1);
    DistR.at<double>(0,0)=k1;
    DistR.at<double>(0,1)=k2;
    DistR.at<double>(0,2)=p1;
    DistR.at<double>(0,3)=p2;

    KR = Mat::eye(3,3,CV_64FC1);
    KR.at<double>(0,0)=fx;
    KR.at<double>(0,2)=cx;
    KR.at<double>(1,1)=fy;
    KR.at<double>(1,2)=cy;

    // left

    fx = 409.9;
    fy = 409.023;

    cx = 337.575;
    cy = 250.798;

    k1 = -0.393895;
    k2 = 0.150157;

    p1 = -0.000753467;
    p2 = -0.00102573;

    DistL = Mat::zeros(1,8,CV_64FC1);
    DistL.at<double>(0,0)=k1;
    DistL.at<double>(0,1)=k2;
    DistL.at<double>(0,2)=p1;
    DistL.at<double>(0,3)=p2;

    KL = Mat::eye(3,3,CV_64FC1);
    KL.at<double>(0,0)=fx;
    KL.at<double>(0,2)=cx;
    KL.at<double>(1,1)=fy;
    KL.at<double>(1,2)=cy;

    return true;
}

int64 workBegin()
{
	return getTickCount();
}

double workEnd(int64 work_begin)
{
    int64 d = getTickCount() - work_begin;
    double f = getTickFrequency();
    double work_time = d / f;
    return work_time;
}

string text(double value)
{
    stringstream ss;
    ss << setiosflags(ios::left)
        << setprecision(4) << value;
    return ss.str();
}

int main (int argc, char** argv) {

	  // registry of images

	  /*string root_dir = "/media/giulia/DATA/humanoids2015/dumpings/bbball2/SFM_rect_imgs";

	  string out_root_dir = "/media/giulia/DATA/humanoids2015/dumpings/bbball2/SFM_disp_offline";

	  string registry_file_left = "/media/giulia/DATA/humanoids2015/dumpings/bbball2/SFM_rect_imgs/left.txt";
	  string registry_file_right = "/media/giulia/DATA/humanoids2015/dumpings/bbball2/SFM_rect_imgs/right.txt";*/

	  /*string root_dir = "/media/giulia/DATA/disparity_data_tmp/dumping/SFM/SFM_rect_imgs";

	  string out_root_dir = "/media/giulia/DATA/disparity_data_tmp/dumping/SFM";

	  string registry_file_left = "/media/giulia/DATA/disparity_data_tmp/dumping/SFM/SFM_rect_imgs/left.txt";
	  string registry_file_right = "/media/giulia/DATA/disparity_data_tmp/dumping/SFM/SFM_rect_imgs/right.txt";*/

      string root_dir = "/media/giulia/DATA/ICUBWORLD_ULTIMATE/iCubWorldUltimate_finaltree/mug/mug1/ROT2D/day5";

	  string out_root_dir = "/media/giulia/DATA/demoDay/disp/mug";

	  string registry_file = "/media/giulia/DATA/ICUBWORLD_ULTIMATE/iCubWorldUltimate_finaltree/mug/mug1/ROT2D/day5/img_info_LR.txt";

	  /*string root_dir = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_rect_imgs";

	  string out_root_dir = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_disp_offline";

	  string registry_file_left = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_rect_imgs/left_mug3.txt";
	  string registry_file_right = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_rect_imgs/right_mug3.txt";*/

	  //string registry_file_left = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_rect_imgs/left_octo.txt";
	  //string registry_file_right = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_rect_imgs/right_octo.txt";

	  //string registry_file_left = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_rect_imgs/left_squeezer.txt";
	  //string registry_file_right = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_rect_imgs/right_squeezer.txt";

	  vector<string> registry_left;
	  vector<string> registry_right;
	  vector<string> registry_out;

	  ifstream infile;
	  string line;

	  infile.open (registry_file.c_str());

	  getline(infile,line);
	  cout << line << endl;
	  while(!infile.eof())
	  {
	      // extract left and right image names
	      vector <string> tokens;

	      split(tokens, line, boost::is_any_of(" "));

	      string left_imgname = tokens[5].substr(0, tokens[5].size()-4) + ".jpg";
	      string right_imgname = tokens[0].substr(0, tokens[0].size()-4) + ".jpg";

		  registry_left.push_back("/left/" + left_imgname);
		  registry_right.push_back("/right/" + right_imgname);
		  registry_out.push_back("/" + left_imgname);

	      getline(infile,line);
	      cout << line << endl;
	  }
	  infile.close();

	  int num_images = registry_left.size();

	  // ELAS setup

	   string elas_string = "MIDDLEBURY";

	   double disp_scaling_factor = 1.0;

	   elasWrapper *elaswrap = new elasWrapper(disp_scaling_factor, elas_string);

	   elaswrap->set_postprocess_only_left(true);

	   elaswrap->set_subsampling(false);

	   elaswrap->set_add_corners(true);

	   elaswrap->set_ipol_gap_width(20);


	   cout << endl << "ELAS parameters:" << endl << endl;

	   cout << "disp_scaling_factor: " << disp_scaling_factor << endl;

	   cout << "setting: " << elas_string << endl;

	   cout << "postprocess_only_left: " << elaswrap->get_postprocess_only_left() << endl;

	   cout << "subsampling: " << elaswrap->get_subsampling() << endl;

	   cout << "add_corners: " << elaswrap->get_add_corners() << endl;

	   cout << "ipol_gap_width: " << elaswrap->get_ipol_gap_width() << endl;

	   cout << endl;

	  // input

	  cv::Mat imL, imR;

	  // ELAS

	  cv::Mat disp_elas, map_elas;

	  cv::Mat map11, map12, map21, map22;
	  cv::Mat Kleft, Kright, DistL, DistR, RLrect, RRrect, PLrect, PRrect;
	  cv:Mat R0, T0, Q;

	  loadIntrinsics(Kleft,Kright,DistL,DistR);
	  loadExtrinsics(R0,T0);

	  Mat zeroDist=Mat::zeros(1,8,CV_64FC1);

	  for (int i=0; i<num_images; i++)
	  {

		  imL = cv::imread(root_dir + registry_left[i]);
		  imR = cv::imread(root_dir + registry_right[i]);

		  cout << root_dir + registry_left[i] << endl;

		  int numberOfDisparities = (imL.cols<=320)?96:128;;

		  cv::Mat img1r, img2r;

		  Size img_size = imL.size();

		  cv::stereoRectify(Kleft, zeroDist, Kright, zeroDist, img_size, R0, T0, RLrect, RRrect, PLrect, PRrect, Q, -1);

		  cv::initUndistortRectifyMap(Kleft, zeroDist, RLrect, PLrect, img_size, CV_32FC1, map11, map12);
		  cv::initUndistortRectifyMap(Kright,  zeroDist, RRrect, PRrect, img_size, CV_32FC1, map21, map22);

		  cv::remap(imL, img1r, map11, map12, cv::INTER_LINEAR);
		  cv::remap(imR, img2r, map21, map22, cv::INTER_LINEAR);

		  namedWindow("RectL", CV_WINDOW_AUTOSIZE );
		  imshow("RectL", img1r );

		  namedWindow("RectR", CV_WINDOW_AUTOSIZE );
		  imshow("RectR", img2r );

		  elaswrap->compute_disparity(img1r, img2r, disp_elas, numberOfDisparities);

		  map_elas = disp_elas * (255.0 / numberOfDisparities);
		  map_elas.convertTo(map_elas, CV_8UC1);

		  cv::namedWindow( "ELAS", cv::WINDOW_AUTOSIZE);
		  cv::imshow( "ELAS", map_elas );

		  cv::waitKey(1);

		  cv::cvtColor(map_elas, map_elas, CV_GRAY2BGR);
		  cv::imwrite(out_root_dir + registry_out[i], map_elas);

	  }

  return 0;
}
