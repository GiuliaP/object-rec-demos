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

// Boost includes

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/filesystem.hpp>

// Project includes

#include "dispBlobber.h"

using namespace std;

int main(int argc, char **argv)
{

    // Working directory containing the images and the results
    string root_dir = "/data/DATASETS/iCubWorld";

    // Choose the categories, object instances and train/test sets from which extract the disparity

    vector <string> categories;
    //categories.push_back("mug");
    //categories.push_back("flower");
    categories.push_back("book");
    categories.push_back("cellphone");
    /*categories.push_back("mouse");
    categories.push_back("pencilcase");
    categories.push_back("ringbinder");
    categories.push_back("hairbrush");
    categories.push_back("hairclip");
    categories.push_back("perfume");
    categories.push_back("sunglasses");
    categories.push_back("wallet");
    categories.push_back("flower");
    categories.push_back("glass");
    categories.push_back("mug");
    categories.push_back("remote");
    categories.push_back("soapdispenser");
    categories.push_back("bodylotion");
    categories.push_back("ovenglove");
    categories.push_back("sodabottle");
    categories.push_back("sprayer");
    categories.push_back("squeezer");*/

    vector <string> objnumbers;
    objnumbers.push_back("1");
    objnumbers.push_back("2");
    /*objnumbers.push_back("3");
    objnumbers.push_back("4");
    objnumbers.push_back("5");
    objnumbers.push_back("6");
    objnumbers.push_back("7");
    objnumbers.push_back("8");
    objnumbers.push_back("9");
    objnumbers.push_back("10");*/

    vector <string> transformations;
    transformations.push_back("ROT2D");
    transformations.push_back("ROT3D");
    /*transformations.push_back("TRANSL");
    transformations.push_back("MIX");
    transformations.push_back("SCALE");*/

    vector <int> days;
    days.push_back(1);
    days.push_back(2);

    // No need to change beyond this line if the folder tree of the dataset is formatted correctly

    string image_dir = root_dir + "/images";
    string in_dir = root_dir + "/disp";

    string out_dir_txtdata = root_dir + "/txtdata";
    string out_dir_bmask = root_dir + "/binary_mask";
    string out_dir_visualization = root_dir + "/visualization";
    string out_dir_crop = root_dir + "/crop";

    // IO extentions

    string in_ext = ".jpg";
    string out_ext = in_ext;

    // dispBlobber setup parameters: 
    // Gianma look here
    // if segmentation is not good! 

	int imH = 480;
	int imW = 640;

	//string cropMethod = "centroid";
	string cropMethod = "bbox";

    int cropMargin = 40;
    int cropRadius = 127;

    int backgroundThresh = 30;

    int minBlobSize = 1300;
    int gaussSize = 5;

    int dispThreshRatioLow = 10;
    int dispThreshRatioHigh = 20;

    int centroidBufferSize = 3;

    dispBlobber *blob_extractor = new dispBlobber(imH, imW, centroidBufferSize,
    		backgroundThresh,
    		minBlobSize, gaussSize,
    		dispThreshRatioLow, dispThreshRatioHigh);

    // For visualization

    cv::Scalar blue = cv::Scalar(255,0,0);
    cv::Scalar green = cv::Scalar(0,255,0);
    cv::Scalar red = cv::Scalar(0,0,255);
    cv::Scalar white = cv::Scalar(255,255,255);

    // Go!!

    for (int c=0; c<categories.size(); c++)
    {
        string category = categories[c];

        for (int o=0; o<objnumbers.size(); o++)
        {
            string objnumber = objnumbers[o];

            for (int t=0; t<transformations.size(); t++)
            {
                string transf = transformations[t];

                for (int d=0; d<days.size(); d++)
                {
                
                     int daynumber_req = days[d];
                     int daynumber;
                     string day;
                     string path_ss = image_dir + "/" + category + "/" + category + objnumber + "/" + transf;

                     for (boost::filesystem::directory_iterator itr(path_ss); itr!=boost::filesystem::directory_iterator(); ++itr)
                     {
                         string cur_day = itr->path().filename().string(); // filename only
                         cout << cur_day << endl;
                         if (boost::filesystem::is_directory(itr->status())) 
                            daynumber = cur_day[3] - '0';
                            cout << daynumber << endl;
                            cout << daynumber_req << endl;
                            if (daynumber % 2==0 && daynumber_req % 2==0 || daynumber % 2==1 && daynumber_req % 2==1)
                                day = cur_day;
                     }

                     string registry_file = image_dir + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day + "/img_info_LR.txt";

                vector<string> registry_right;
                vector <string> registry_left;

                ifstream infile;
                string line;
                infile.open (registry_file.c_str());

                getline(infile,line);
                while(!infile.eof())
                {

                    vector <string> tokens;
                    split(tokens, line, boost::is_any_of(" "));

                    string left_imgname = tokens[5].substr(0, tokens[5].size()-4);
                    string right_imgname = tokens[0].substr(0, tokens[0].size()-4);

                    registry_left.push_back(left_imgname);
                    registry_right.push_back(right_imgname);

                    getline(infile,line);

                }
                infile.close();

                int numImages = registry_left.size();

                cout << "Found " << numImages << " images for " << category + objnumber + transf + day << endl;

                // Output preparation

                if (boost::filesystem::exists(out_dir_txtdata + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day)==false)
                    boost::filesystem::create_directories(out_dir_txtdata + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day);

                if (boost::filesystem::exists(out_dir_crop + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day)==false)
                    boost::filesystem::create_directories(out_dir_crop + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day);

                if (boost::filesystem::exists(out_dir_bmask + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day)==false)
                    boost::filesystem::create_directories(out_dir_bmask + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day);

                if (boost::filesystem::exists(out_dir_visualization + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day)==false)
                    boost::filesystem::create_directories(out_dir_visualization + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day);

                // Blob extraction

                int countFails = 0;

                ofstream outfile;
                std::string out_filename = out_dir_txtdata + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day + "/centroid_bbox.txt";
                outfile.open (out_filename.c_str());

                for (int i = 0; i < numImages; i++)
                {

                    string disp_path = in_dir + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day + "/" + registry_right[i];
                    cv::Mat disp = cv::imread(disp_path + in_ext);

                    string image_path = image_dir + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day + "/left/" + registry_left[i];
                    cv::Mat image = cv::imread(image_path + in_ext);

                    std::vector<int> bbox;
                    std::vector<int> centroid;
                    cv::Mat bmask;

                    double blobSize = blob_extractor->extractBlob(disp, bbox, centroid, bmask);

                    if (blobSize<0)
                        countFails++;

                    // Visualization and output production

                    // bmask

                    cv::namedWindow("bmask", cv::WINDOW_AUTOSIZE);
                    cv::imshow( "bmask", bmask );
                    cv::imwrite(out_dir_bmask + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day + "/" + registry_right[i] + out_ext, bmask);

                    // crop

                    cv::Point tl;
                    cv::Point br;

                    if (cropMethod=="bbox")
                    {
                        tl = cv::Point(std::max(bbox[0]-cropMargin,0), std::max(bbox[1]-cropMargin,0));
                        br = cv::Point( std::min(bbox[2]+cropMargin,image.cols-1), std::min(bbox[3]+cropMargin,image.rows-1));
                    }
                    else if (cropMethod=="centroid")
                    {
                        tl = cv::Point(std::max(centroid[0]-cropRadius,0), std::max(centroid[1]-cropRadius,0));
                        br = cv::Point( std::min(centroid[0]+cropRadius,image.cols-1), std::min(centroid[1]+cropRadius,image.rows-1));
                    }

                    cv::Rect imBox(tl,br);

                    cv::namedWindow("crop", cv::WINDOW_AUTOSIZE);
                    cv::imshow( "crop", image(imBox) );
                    cv::imwrite(out_dir_crop + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day + "/" + registry_right[i] + out_ext, image(imBox).clone());

                    // txtdata

                    for (int j=0; j<centroid.size(); j++)
                        outfile << centroid[j] << "\t";
                    for (int j=0; j<bbox.size(); j++)
                        outfile << bbox[j] << "\t";
                    outfile << std::endl;

                    // disp

                    cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
                    cv::rectangle(disp, imBox, green, 2);
                    cv::circle(disp, cv::Point(centroid[0], centroid[1]), 4, green, -1);
                    cv::imshow( "disp", disp );

                    // complete visualization

                    cv::namedWindow("view", cv::WINDOW_AUTOSIZE);
                    cv::rectangle(image, imBox, green, 2);
                    cv::circle(image, cv::Point(centroid[0], centroid[1]), 4, green, -1);
                    cv::imshow( "view", image );
                    cv::imwrite(out_dir_visualization + "/" + category + "/" + category + objnumber + "/" + transf + "/" + day + "/" + registry_right[i] + out_ext, image);

                    cv::waitKey(100);

                }

                cout << "perc. missed frames: " << (float)countFails/(float)numImages << endl;

                outfile.close();
            }
        }
    }

}

 return 0;

}
