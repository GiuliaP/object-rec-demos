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

#include "CaffeFeatExtractor.hpp"

using namespace std;

int main(int argc, char **argv)
{

    // Working directory containing the images and the results
    string root_dir = "/media/giulia/DATA/demoDay";

    // Choose the categories, object instances and train/test sets from which extract the disparity

    vector <string> categories;
    categories.push_back("mug");
    categories.push_back("flower");

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
    string in_dir = root_dir + "/crop";

    // IO extentions

    string in_ext = ".jpg";
    string out_ext = ".txt";

    // Caffe setup parameters

    CaffeFeatExtractor<float> *caffe_extractor;

    string modelName = "caffenet";
    //string modelName = "googlenet";

    // Binary file (.caffemodel) containing the pretrained network's weights
    string pretrained_binary_proto_file = "/home/giulia/REPOS/object-rec-demos/c_caffeCoder_offline/caffe_models/" + modelName + ".caffemodel";

    // Text file (.prototxt) defining the network structure
    string feature_extraction_proto_file = "/home/giulia/REPOS/object-rec-demos/c_caffeCoder_offline/caffe_models/" + modelName + ".prototxt";

    cout << "Using pretrained network: " << pretrained_binary_proto_file << endl;
    cout << "Using network defined in: " << feature_extraction_proto_file << endl;

    // Names of layers to be extracted
    string extract_features_blob_names = "prob";

    // Output dirs
    vector <string> out_dirs;
    split(out_dirs, extract_features_blob_names, boost::is_any_of(","));
    int numFeatures = out_dirs.size();
    for (int d=0; d<numFeatures; d++)
    {
        out_dirs[d] = root_dir + "/" + modelName + "_" + out_dirs[d];
    }

    // GPU or CPU mode
    string compute_mode = "GPU";
    // If compute_mode="GPU", must specify device ID
    int device_id = 0;

    bool timing = true;

    int batchSize = 1;
    int batchSizeCaffe = 1;

    caffe_extractor = new CaffeFeatExtractor<float>(pretrained_binary_proto_file,
        		   feature_extraction_proto_file,
        		   extract_features_blob_names,
        		   compute_mode,
        		   device_id,
        		   timing);

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

                vector<string> registry;

                ifstream infile;
                string line;
                infile.open (registry_file.c_str());

                getline(infile,line);
                while(!infile.eof())
                {

                    vector <string> tokens;
                    split(tokens, line, boost::is_any_of(" "));
                    string crop_imgname = tokens[0].substr(0, tokens[0].size()-4);

                    registry.push_back(crop_imgname);

                    getline(infile,line);

                }
                infile.close();

                int numImages = registry.size();

                cout << "Found " << numImages << " images for " << category + objnumber << ": " << set << endl;

                // Output preparation

                for (int d=0; d<numFeatures; d++)
                {
                    if (boost::filesystem::exists(out_dirs[d] + "/" + category + "/" + category + objnumber + "/" + set)==false)
                        boost::filesystem::create_directories(out_dirs[d] + "/" + category + "/" + category + objnumber + "/" + set);
                }

                // Feature extraction

                if (numImages%batchSize!=0)
                {
                    batchSize = 1;
                    cout << "WARNING main: image number is not multiple of batch size, setting to 1." << endl;
                }

                int numMiniBatches = numImages/batchSize;

                vector< vector<float> > features;
                vector<cv::Mat> images;

                for (int batch_index = 0; batch_index < numMiniBatches; batch_index++)
                {
                    for (int i=0; i<batchSize; i++)
                    {
                        string crop_path = in_dir + "/" + category + "/" + category + objnumber + "/" + set + "/" + registry[batch_index*batchSize + i] + in_ext;
                        cv::Mat crop = cv::imread(crop_path);
                        images.push_back(crop);
                    }

                    caffe_extractor->extractBatch_multipleFeat_1D(images, batchSizeCaffe, features);

                    for (int i=0; i<batchSize; i++)
                    {
                        for (int d=0; d<numFeatures; d++)
                        {
                            string out_filename = out_dirs[d] + "/" + category + "/" + category + objnumber + "/" + set + "/" + registry[batch_index*batchSize + i] + out_ext;
                            ofstream outfile;
                            outfile.open (out_filename.c_str());

                            for (int j=0; j<features[i+d*batchSize].size(); j++)
                                outfile << features[i+d*batchSize][j] << endl;
                            outfile.close();
                        }
                    }

                    features.clear();
                    images.clear();
                }

            }
        }
    }

    return 0;
}
