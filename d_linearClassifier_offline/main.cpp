// std::system includes

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>

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

#include "linearClassifier.h"

using namespace std;

int main(int argc, char * argv[])
{

    // Working directory containing the images and the results
    string root_dir = "/media/giulia/DATA/demoDay";

    // No need to change beyond this line if the folder tree of the dataset is formatted correctly

    string classifier_dir = root_dir + "/classifierDB";
    string vis_dir = root_dir + "/visualization";
    string reg_dir = root_dir + "/images";
    string out_vis_dir = root_dir + "/visualization_classified";

    // IO extentions

    string in_ext = ".jpg";
    string out_ext = ".jpg";

    // Classifier setup parameters

    string featureType = "caffenet_fc6";

    int bufferSize = 1; // 1 5 10
    double voteThreshold = 0; // 50 75
    bool weightedClassification = false; // or true

    bool retrainWhenForget = true;
    bool newGURLSinterface = true;

    cout << "Hello, let's start learning!" << endl << endl;

    linearClassifier predictor(classifier_dir, bufferSize, voteThreshold, weightedClassification, newGURLSinterface);

    string cmd, reply;
    bool success;

    while (true)
    {
        // Read command

        std::getline(std::cin, cmd);

        // Split command into words

        std::istringstream buf(cmd);
        std::istream_iterator<std::string> beg(buf), end;
        std::vector<std::string> tokens(beg, end);

        if (tokens.size()==1 && tokens[0]=="help")
        {
            success = true;
            reply = "Commands available: \n";
            reply = reply + "quit: Quit the program.\n";
            reply = reply + "set bufferSize [>0]: set the number of frames to average on for prediction.\n";
            reply = reply + "set voteThreshold [0-100]: set the minimum percentage of frames in the buffer that must agree in the prediction.\n";
            reply = reply + "set weightedClassification [0/1]: whether to weight the examples or not.\n";
            reply = reply + "set newGURLSinterface [0/1]: whether to use the new wrappers or the usual pipeline.\n";
            reply = reply + "get bufferSize: retrieve the parameter.\n";
            reply = reply + "get voteThreshold: retrieve the parameter.\n";
            reply = reply + "get weightedClassification: retrieve the parameter.\n";
            reply = reply + "get newGURLSinterface: retrieve the parameter.\n";
            reply = reply + "observe <category_objID> <label>: load in the database the features of the specified object.\n";
            reply = reply + "train: train the classifiers.\n";
            reply = reply + "test <category_objID> <label>: predict the features of the specified object and compute the accuracy considering the provided label as the true one.\n";
            reply = reply + "forget all: forget all objects.\n";
            reply = reply + "forget <label>: forget the specified class.\n";
        }
        else if (tokens.size()==1 && tokens[0]=="quit")
        {
            success = predictor.releaseModels();
            reply = "Bye bye!";
            break;
        }
        else if (tokens.size()==3 && tokens[0]=="set")
        {
            if (tokens[1] == "bufferSize")
            {
                success = predictor.set_bufferSize(atoi(tokens[2].c_str()));
                reply = "Parameter set.";
            }
            else if (tokens[1] == "voteThreshold")
            {
                success = predictor.set_voteThreshold(atoi(tokens[2].c_str()));
                reply = "Parameter set.";
            }
            else if (tokens[1] == "weightedClassification")
            {
                success = predictor.set_weightedClassification(atoi(tokens[2].c_str()));
                reply = "Parameter set.";
            }
            else if (tokens[1] == "newGURLSinterface")
            {
                success = predictor.set_newGURLSinterface(atoi(tokens[2].c_str()));
                reply = "Parameter set, forgotten everything. Ready to restart learning.";
            }
            else
            {
                reply = "Parameter not recognized.";
                success = true;
            }
        }
        else if (tokens.size()==2 && tokens[0]=="get")
        {
            if (tokens[1] == "bufferSize")
            {
                cout << predictor.get_bufferSize() << endl;
                reply = "";
                success = true;
            }
            else if (tokens[1] == "voteThreshold")
            {
                cout << predictor.get_voteThreshold() << endl;
                reply = "";
                success = true;
            }
            else if (tokens[1] == "weightedClassification")
            {
                cout << predictor.get_weightedClassification() << endl;
                reply = "";
                success = true;
            }
            else if (tokens[1] == "newGURLSinterface")
            {
                cout << predictor.get_newGURLSinterface() << endl;
                reply = "";
                success = true;
            }
            else
            {
                reply = "Parameter not recognized.";
                success = true;
            }
        }
        else if (tokens.size()==3 && tokens[0]=="observe")
        {
            vector <string> subtokens;
            split(subtokens, tokens[1], boost::is_any_of("_"));
            string category = subtokens[0];
            string objNumber = subtokens[1];
            string path = root_dir + "/" + featureType + "/" + category + "/" + category + objNumber + "/train";
            string label = tokens[2];
            success = predictor.importFeatures(path, label);

            reply = label + ": observed.";
        }
        else if (tokens[0]=="list")
        {
            vector <string> list;
            success = predictor.getClassList(list);

            reply = "";
            cout << "Found " << list.size() << " class(es):" << endl;
            for (int i=0; i<list.size(); i++)
            {
                reply += list[i] + "\n";
            }

        }
        else if (tokens[0]=="train")
        {
            success = predictor.trainClassifiers();
            reply = "Trained.";
        }
        else if (tokens.size()==3 && tokens[0]=="test")
        {

            vector <string> subtokens;
            split(subtokens, tokens[1], boost::is_any_of("_"));
            string category = subtokens[0];
            string objNumber = subtokens[1];
            string path = root_dir + "/" + featureType + "/" + category + "/" + category + objNumber + "/test";

            string truelabel = tokens[2];

            vector <string> predictions;
            vector <vector <double> > scores;
            vector <string> scoresOrder;

            success = predictor.recognize(path, predictions, scores, scoresOrder);

            string registry_file = reg_dir + "/" + category + "/" + category + objNumber + "/test/img_info_LR.txt";

             vector<string> registry;

             ifstream infile;
             string line;
             infile.open (registry_file.c_str());

             getline(infile,line);
             while(!infile.eof())
             {

                 vector <string> tokens;
                 split(tokens, line, boost::is_any_of(" "));
                 string imgname = tokens[0].substr(0, tokens[0].size()-4);

                 registry.push_back(imgname);

                 getline(infile,line);

             }
             infile.close();

             int numImages = registry.size();

             cout << "Found " << numImages << " images for " << category + objNumber << ": test" << endl;

             if (boost::filesystem::exists(out_vis_dir + "/" + category + "/" + category + objNumber + "/test")==false)
                 boost::filesystem::create_directories(out_vis_dir + "/" + category + "/" + category + objNumber + "/test");

             //cout << "numImages" << numImages << " predictions" << predictions.size() << endl;

             for (int i = 0; i < numImages; i++)
              {
                  cv::Mat img = cv::imread(vis_dir + "/" + category + "/" + category + objNumber + "/test/" + registry[i] + in_ext);

                  cv::putText(img, predictions[i], cv::Point(10,img.rows-10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2.0);
                  cv::namedWindow("view", cv::WINDOW_AUTOSIZE);
                  cv::imshow( "view", img );
                  cv::imwrite(out_vis_dir + "/" + category + "/" + category + objNumber + "/test/" + registry[i] + out_ext, img);

                  cv::waitKey(100);
              }

             int countCorrect = 0;
             for (int i=0; i<numImages; i++)
                 if (predictions[i]==truelabel)
                     countCorrect++;

             double accuracy = (double)countCorrect/(double)numImages;
             cout << "Accuracy (% correct): " << accuracy << endl;

             reply = "Recognition done.";
        }
        else if (tokens.size()==2 && tokens[0]=="forget")
        {
            string what = tokens[1];
            if (what=="all")
            {
                success = predictor.forgetAll(retrainWhenForget);
                reply = "Tabula rasa.";
            }
            else
            {
                success = predictor.forgetClass(what, retrainWhenForget);
                reply = what + ": forgotten.";
            }
        }
        else
        {
            reply = "Command not recognized.";
        }

        if (success)
            cout << reply << endl;
        else
            cout << "Error while executing command." << endl;
    }

    if (success)
        cout << reply << endl;
    else
        cout << "Error while executing command." << endl;

    return 0;
}
