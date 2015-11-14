// std::system includes

#include <stdio.h>
#include <iostream>
#include <fstream>

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

// Project includes

#include "dispBlobber.h"

using namespace std;

int main(int argc, char **argv)
{

    ////////////////////////////////////////////////////////////////////////////////
    // nearBlobber Initialization
    ////////////////////////////////////////////////////////////////////////////////

	int imH = 480;
	int imW = 640;

    int _margin = 40;

    int _backgroundThresh = 30;
    int _frontThresh = 190;
    //double _cannyThresh = 20;

    int _minBlobSize = 1300;
    int _gaussSize = 5;

    int _dispThreshRatioLow = 10;
    int _dispThreshRatioHigh = 20;

    // nearBlobber class declaration
    nearBlobber *blob_extractor;

    bool _timing = true;

    int centroidBufferSize = 3;

    // nearBlobber class instantiation
    blob_extractor = NULL;
    blob_extractor = new nearBlobber(imH, imW, centroidBufferSize,
    		_margin,
    		_backgroundThresh, _frontThresh,
    		_minBlobSize, _gaussSize,
    		_dispThreshRatioLow, _dispThreshRatioHigh);

    ////////////////////////////////////////////////////////////////////////////////
    // Registry preparation
    ////////////////////////////////////////////////////////////////////////////////

    /*string root_dir = "/media/giulia/DATA/humanoids2015/dumpings/dumping_humanoids_640objects/candybottle/SFM_disp_offline/dispSGBM";

    string out_dir_blobs = "/media/giulia/DATA/humanoids2015/dumpings/dumping_humanoids_640objects/candybottle/nearBlobber_blobs_offline/blobsSGBM_notime";
    string out_dir_opt = "/media/giulia/DATA/humanoids2015/dumpings/dumping_humanoids_640objects/candybottle/nearBlobber_opt_offline/optSGBM_notime";

    string registry_file = "/media/giulia/DATA/humanoids2015/dumpings/dumping_humanoids_640objects/candybottle/SFM_rect_imgs/left.txt";*/

    string image_dir = "/media/giulia/DATA/ICUBWORLD_ULTIMATE/iCubWorldUltimate_finaltree/mug/mug1/ROT2D/day5/left";

    string root_dir = "/media/giulia/DATA/demoDay/disp/mug";

    string out_dir_blobs = "/media/giulia/DATA/demoDay/blob/mug";
    string out_dir_opt = "/media/giulia/DATA/demoDay/binary_mask/mug";
    string out_dir_roi = "/media/giulia/DATA/demoDay/roi/mug";
    string out_dir_segm = "/media/giulia/DATA/demoDay/segm/mug";

    string registry_file = "/media/giulia/DATA/ICUBWORLD_ULTIMATE/iCubWorldUltimate_finaltree/mug/mug1/ROT2D/day5/img_info_LR.txt";

    /*string root_dir = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_disp_offline/ELAS_notime/squeezer";

    string image_dir = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_rect_imgs/SFM_rectleft_rgb_selected/squeezer";

    string out_dir_blobs = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/nearBlobber_blobs_offline/ELAS_notime/squeezer";
    string out_dir_opt = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/nearBlobber_opt_offline/ELAS_notime/squeezer";
    string out_dir_roi = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/nearBlobber_roi_offline/ELAS_notime/squeezer";
    string out_dir_segm = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/nearBlobber_segm_offline/ELAS_notime/squeezer";

    string registry_file = "/media/giulia/DATA/humanoids2015/dumpings/applications/onthefly/SFM_rect_imgs/left_squeezer.txt";*/

    vector<string> registry;

    ifstream infile;
    string line;

    infile.open (registry_file.c_str());

    getline(infile,line);
    cout << line << endl;
    while(!infile.eof())
    {
        vector <string> tokens;

        split(tokens, line, boost::is_any_of(" "));

        string disp_imgname = tokens[5].substr(0, tokens[5].size()-4) + ".jpg";

        registry.push_back(disp_imgname);

    	getline(infile,line);
    	cout << line << endl;
    }
    infile.close();

    int num_images = registry.size();

    ////////////////////////////////////////////////////////////////////////////////
    // Blob extraction
    ////////////////////////////////////////////////////////////////////////////////

    std::vector<cv::Mat> buffer;
    int buffer_size = 1;

    std::vector< vector<int> > blobs;
    std::vector< vector<int> > centroids;

    int count_t = 0;
    double count_time = 0.0;


    for (int i = 0; i < num_images; i++)
    {

    	string image_path = root_dir + "/" + registry[i];
    	cv::Mat img = cv::imread(image_path);

    	buffer.push_back(img);

    	if (buffer.size()>buffer_size)
    	{
    		buffer.erase(buffer.begin());
    	}

    	blobs.push_back(std::vector<int>());
    	centroids.push_back(std::vector<int>());

        cv::Mat opt;

        double t[2];

    	blob_extractor->extractBlob(buffer, blobs[i], centroids[i], opt, t);

    	if (t[0]<0)
    		count_t++;

    	cout << t[0] << " " << count_t << endl;

    	count_time += t[1];

    	cout << t[1] << " " << count_time << endl;

    	cv::imwrite(out_dir_opt + "/" + registry[i], opt);

        cv::Mat originalimage = cv::imread(image_dir + "/" + registry[i]);

        cv::Rect imBox(cv::Point(blobs[i][0],blobs[i][1]),cv::Point(blobs[i][2],blobs[i][3]));

        cv::namedWindow("segm", cv::WINDOW_AUTOSIZE);
        cv::imshow( "segm", originalimage(imBox) );
        cv::Mat temp = originalimage(imBox).clone();
        cv::imwrite(out_dir_segm + "/" + registry[i], temp);

        cv::namedWindow("roi", cv::WINDOW_AUTOSIZE);
        //cv::putText(originalimage, "look:squeezer", cv::Point(blobs[i][0],blobs[i][1]-5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2.0);

        cv::rectangle(originalimage, imBox, cv::Scalar(0,0,255), 2);
        cv::imshow( "roi", originalimage );
        cv::imwrite(out_dir_roi + "/" + registry[i], originalimage);

        cv::waitKey(500);

    }

    cout << "perc. found blobs: " << (float)count_t/(float)num_images << endl;

    cout << "avg. time: " << count_time/num_images << endl;

    ////////////////////////////////////////////////////////////////////////////////
    // Output production
    ////////////////////////////////////////////////////////////////////////////////


    ofstream outfile;
    std::string out_filename = out_dir_blobs + "/" + "blobs.txt";

    outfile.open (out_filename.c_str());

    cout << centroids.size();
    for (int i=0; i<centroids.size(); i++)
    {
    	for (int j=0; j<centroids[i].size(); j++)
    	{
    		outfile << centroids[i][j] << "\t";
    	}
    	outfile << std::endl;
    }

    outfile.close();

}
