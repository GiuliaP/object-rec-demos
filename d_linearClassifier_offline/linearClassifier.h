#ifdef _WIN32
    #include "win_dirent.h"
#else
    #include "dirent.h"
#endif

#include <sys/types.h>
#include <sys/stat.h>

#include <iostream>
#include <string>

// GURLS includes

#include "gurls++/gurls.h"
#include "gurls++/gmat2d.h"
#include "gurls++/primal.h"

#include <vector>
#include <fstream>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

// Project includes
#include "RLSlinear.h"

using namespace std;
using namespace gurls;
  
class linearClassifier
{

    string currPath;
    string pathObj;

    fstream objFeatures;

    // Dataset variables

    vector<pair<string,vector<string> > > knownObjects;
    vector<vector<vector<double> > > Features;
    vector<int> datasetSizes;

    // Learning variables

    int bufferSize;
    double minVotePercentage;
    int useWeightedClassification;

    // GURLS variables

    vector<RLSlinear> linearClassifiers_RLS;
    vector<vector<double > > bufferScores_RLS;
    vector<vector<int > > countBuffer_RLS;

    double CSVM;
    bool paramsel;

    // Utilities

    bool getdir(string dir, vector<string> &files);
    bool loadFeatures();
    bool readFeatures(string filePath, vector<vector<double> > *featuresMat);
    bool createFullPath(const char * path);
    bool checkKnownObjects();

public:

    linearClassifier(string rootPath, int _bufferSize, double minVotePercentage, bool _weighted);

    bool importFeatures(string inputPath, string objName);
 
    bool trainClassifiers();

    bool recognize(string inputPath);

    bool forgetClass(string className, bool retrain=true);

    bool forgetAll();

    bool releaseModels();

    bool getClassList(vector <string> &list);

};

