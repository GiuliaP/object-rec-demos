// std::system includes

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// OpenCV includes

#include <opencv/highgui.h>
#include <opencv/cv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"

// Boost includes

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/filesystem.hpp>

// Project includes

#include "elasWrapper.h"

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

string text(double value)
{
    stringstream ss;
    ss << setiosflags(ios::left) << setprecision(4) << value;
    return ss.str();
}

int main (int argc, char** argv) {

    // Working directory containing the images and the results
    string root_dir = "/media/giulia/DATA/demoDay";

    // Choose the categories, object instances and train/test sets from which extract the disparity

    vector <string> categories;
    categories.push_back("mug");
    categories.push_back("flower");
    categories.push_back("book");

    vector <string> objnumbers;
    objnumbers.push_back("1");
    objnumbers.push_back("2");
    objnumbers.push_back("3");
    objnumbers.push_back("4");
    objnumbers.push_back("5");

    vector <string> sets;
    sets.push_back("train");
    sets.push_back("test");

    // No need to change beyond this line if the folder tree of the dataset is formatted correctly

    string image_dir = root_dir + "/images";
    string in_dir = image_dir;
    string out_dir = root_dir + "/disp";

    // IO extentions

    string in_ext = ".jpg";
    string out_ext = in_ext;

    // LIBELAS setup parameters

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
    cout << "ipol_gap_width: " << elaswrap->get_ipol_gap_width() << endl << endl;

    // Load camera parameters

    cv::Mat map11, map12, map21, map22;
    cv::Mat Kleft, Kright, DistL, DistR, RLrect, RRrect, PLrect, PRrect;
    cv:Mat R0, T0, Q;

    loadIntrinsics(Kleft,Kright,DistL,DistR);
    loadExtrinsics(R0,T0);

    Mat zeroDist = Mat::zeros(1,8,CV_64FC1);

    // Go!!

    for (int c=0; c<categories.size(); c++)
    {
        string category = categories[c];

        for (int o=0; o<objnumbers.size(); o++)
        {
            string objnumber = objnumbers[o];

            for (int s=0; s<sets.size(); s++)
            {
                string set = sets[s];

                string registry_file = image_dir + "/" + category + "/" + category + objnumber + "/" + set + "/img_info_LR.txt";

                vector<string> registry_left;
                vector<string> registry_right;
                vector<string> registry_out;

                ifstream infile;
                string line;
                infile.open (registry_file.c_str());

                getline(infile,line);
                while(!infile.eof())
                {
                    // Extract left and right image names from the registry

                    vector <string> tokens;
                    split(tokens, line, boost::is_any_of(" "));
                    string left_imgname = tokens[5].substr(0, tokens[5].size()-4);
                    string right_imgname = tokens[0].substr(0, tokens[0].size()-4);

                    registry_left.push_back("left/" + left_imgname);
                    registry_right.push_back("right/" + right_imgname);

                    // We choose to call the output disparity as the left image
                    registry_out.push_back(right_imgname);

                    getline(infile,line);
                }
                infile.close();

                int num_images = registry_left.size();

                cout << "Found " << num_images << " images for " << category + objnumber << ": " << set << endl;

                // Output preparation

                if (boost::filesystem::exists(out_dir + "/" + category + "/" + category + objnumber + "/" + set)==false)
                    boost::filesystem::create_directories(out_dir + "/" + category + "/" + category + objnumber + "/" + set);

                // Input matrices
                cv::Mat imL, imR;

                // Auxiliary and output matrices
                cv::Mat disp_elas, map_elas;

                for (int i=0; i<num_images; i++)
                {

                    // Read image pair

                    imL = cv::imread(in_dir + "/" + category + "/" + category + objnumber + "/" + set + "/" + registry_left[i] + in_ext);
                    imR = cv::imread(in_dir + "/" + category + "/" + category + objnumber + "/" + set + "/" + registry_right[i] + in_ext);

                    int numberOfDisparities = (imL.cols<=320)?96:128;;

                    // Rectify image pair

                    cv::Mat img1r, img2r;
                    Size img_size = imL.size();
                    cv::stereoRectify(Kleft, zeroDist, Kright, zeroDist, img_size, R0, T0, RLrect, RRrect, PLrect, PRrect, Q, -1);
                    cv::initUndistortRectifyMap(Kleft, zeroDist, RLrect, PLrect, img_size, CV_32FC1, map11, map12);
                    cv::initUndistortRectifyMap(Kright,  zeroDist, RRrect, PRrect, img_size, CV_32FC1, map21, map22);
                    cv::remap(imL, img1r, map11, map12, cv::INTER_LINEAR);
                    cv::remap(imR, img2r, map21, map22, cv::INTER_LINEAR);

                    // Visualize rectified images

                    namedWindow("RectL", CV_WINDOW_AUTOSIZE );
                    imshow("RectL", img1r );
                    namedWindow("RectR", CV_WINDOW_AUTOSIZE );
                    imshow("RectR", img2r );

                    // Compute disparity

                    elaswrap->compute_disparity(img1r, img2r, disp_elas, numberOfDisparities);
                    map_elas = disp_elas * (255.0 / numberOfDisparities);
                    map_elas.convertTo(map_elas, CV_8UC1);

                    // Visualize disparity

                    cv::namedWindow( "ELAS", cv::WINDOW_AUTOSIZE);
                    cv::imshow( "ELAS", map_elas );
                    cv::waitKey(1);

                    // Save disparity

                    cv::cvtColor(map_elas, map_elas, CV_GRAY2BGR);
                    cv::imwrite(out_dir + "/" + category + "/" + category + objnumber + "/" + set + "/" + registry_out[i] + out_ext, map_elas);

                }
            }
        }
    }

    return 0;
}
