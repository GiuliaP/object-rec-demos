// std::system includes

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <vector>

// OpenCV includes

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Project includes

#include "linearClassifier.h"

using namespace std;

int main(int argc, char * argv[])
{

    string root_dir = "/media/giulia/DATA/demoDay/classfierDB";

    int buffer_size = 20;
    double min_vote_perc = 0.75;

    double CSVM = 1.0;
    bool paramsel = true;
    bool weighted = true;

    bool retrainWhenForget = true;

    cout << "Hello, let's start learning!" << endl << endl;

    linearClassifier predictor(root_dir, buffer_size, min_vote_perc, weighted);

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

        if (tokens.size()==1 && tokens[0]=="quit")
        {
            success = predictor.releaseModels();
            reply = "Bye bye!";
            break;
        }
        else if (tokens.size()==3 && tokens[0]=="observe")
        {

            string obj = tokens[1];
            string path = tokens[2];
            success = predictor.importFeatures(path, obj);

            reply = obj + ": observed.";
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
        else if (tokens.size()==2 && tokens[0]=="test")
        {
            string path = tokens[1];
            success = predictor.recognize(path);
            reply = "Recognition done.";
        }
        else if (tokens.size()==2 && tokens[0]=="forget")
        {
            string what = tokens[1];
            if (what=="all")
            {
                success = predictor.forgetAll();
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
