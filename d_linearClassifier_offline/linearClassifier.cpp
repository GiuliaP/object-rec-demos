// Project includes

#include "linearClassifier.h"

using namespace gurls;

linearClassifier::linearClassifier(string rootPath, int _bufferSize, double _minVotePercentage, bool _weighted)
{

    currPath = rootPath;

    bufferSize = _bufferSize;
    minVotePercentage = _minVotePercentage;

    useWeightedClassification = _weighted;

    trainClassifiers();

}

bool linearClassifier::trainClassifiers()
{

    if (linearClassifiers_RLS.size()>0)
        for (int i=0; i<linearClassifiers_RLS.size(); i++)
            linearClassifiers_RLS[i].freeModel();

    cout << "Loading samples..." << endl;

    bool success = loadFeatures();
    if (success==false || datasetSizes.size()==0)
    {
        cout << "No samples found!" << endl;
        return true;
    }

    cout << "Samples loaded!" << endl;

    linearClassifiers_RLS.clear();

    int T = knownObjects.size();

    int d = 0;
    if (Features.size()>0 && Features[0].size()>0)
        d = Features[0][0].size();
    else
    {
        cout << "Error! No samples provided." << endl;
        return false;
    }

        int n = 0;
        for (int i=0; i<knownObjects.size(); i++)
            n += Features[i].size();

        gurls::gMat2D<float> Y(n,1);
        gurls::gMat2D<float> X(n,d+1);

        for (int idx_curr_obj=0; idx_curr_obj<knownObjects.size(); idx_curr_obj++)
        {
            string name = knownObjects[idx_curr_obj].first;
            RLSlinear rlsmodel(name);

            float pos_weight = useWeightedClassification?sqrt((float)(((float)n-Features[idx_curr_obj].size())/Features[idx_curr_obj].size())):1.0;
            float neg_weight = -1.0;

            // Fill positive labels for Y

            int curr_row_idx = 0;
            for(int idx_obj=0; idx_obj<knownObjects.size(); idx_obj++)
            {
                for(int idx_feat=0; idx_feat<Features[idx_obj].size(); idx_feat++)
                {
                    float weight=(idx_obj==idx_curr_obj)?pos_weight:neg_weight;

                    for(int i=0; i<d; i++)
                        X(curr_row_idx,i)=abs(weight)*Features[idx_obj][idx_feat][i];
                    X(curr_row_idx,d)=abs(weight);

                    Y(curr_row_idx,0)=weight;

                    curr_row_idx++;
                }
            }

            printf("[RLS] nClass: %d nPositive: %d nNegative: %d\n",int(knownObjects.size()),int(Features[idx_curr_obj].size()),n-int(Features[idx_curr_obj].size()));

            rlsmodel.trainModel(X,Y);
            linearClassifiers_RLS.push_back(rlsmodel);
            string tmpModelPath=currPath+"/"+knownObjects[idx_curr_obj].first+"/rlsmodel";
        }

    return true;

}

bool linearClassifier::loadFeatures()
{

    bool success = checkKnownObjects();

    if (success==false || knownObjects.size()==0)
        return false;

    Features.clear();
    Features.resize(knownObjects.size());
    datasetSizes.clear();

    for (int i=0; i<knownObjects.size(); i++)
    {
        vector<string> obj=knownObjects[i].second;
        int cnt=0;
        for (int k=0; k< obj.size(); k++)
        {
            vector<vector<double> > tmpF;
            bool success = readFeatures(obj[k],&tmpF);
            if (success==false)
                return false;

            cnt=cnt+tmpF.size();
            for (int t =0; t<tmpF.size(); t++)
                Features[i].push_back(tmpF[t]);
        }

        if(cnt>0)
            this->datasetSizes.push_back(cnt);
    }

    return true;

}
bool linearClassifier::readFeatures(string filePath, vector<vector<double> > *featuresMat)
{
    string line;
    ifstream infile;
    infile.open(filePath.c_str());
    if (infile.is_open()==false)
        return false;

    while(!infile.eof())
    {
        vector<double> f;
        getline(infile,line); // Saves the line in STRING.

        char * val= strtok((char*) line.c_str()," ");

        while(val!=NULL)
        {

            double value=atof(val);
            f.push_back(value);
            val=strtok(NULL," ");
        }
        if(f.size()>0)
            featuresMat->push_back(f);
    }

    infile.close();

    return true;
}

bool linearClassifier::importFeatures(string inputPath, string objName)
{

    // Prepare output path

    pathObj = currPath + "/" + objName;

    struct stat info;
    if (stat(pathObj.c_str(), &info)!=0)
    {
        bool success = createFullPath(pathObj.c_str());
        if (success==false)
            return false;
        pathObj = pathObj + "/1.txt";
    }
    else
    {
        char tmpPath[255];
        bool proceed = true;

        struct stat info;
        for (int i=1; proceed; i++)
        {
               sprintf(tmpPath,"%s/%d.txt",pathObj.c_str(),i);
               if (stat(tmpPath, &info)==0)
                   proceed = true;
               else
                   proceed = false;
               sprintf(tmpPath,"%s/%d.txt",pathObj.c_str(),i);
        }

        pathObj = tmpPath;

    }

    objFeatures.open(pathObj.c_str(),fstream::out | fstream::out);
    if (objFeatures.is_open()==false)
        return false;

    // Load features

    vector <string> fileList;
    getdir(inputPath, fileList);

    for (int i=0; i< fileList.size(); i++)
    {
        if (fileList[i].compare(".") && fileList[i].compare(".."))
        {

            string line;
            vector<double> feature;

            string fullPath = inputPath + "/" + fileList[i];
            ifstream infile;
            infile.open(fullPath.c_str());

            if (infile.is_open())
            {
                getline(infile,line);
                while(!infile.eof())
                {
                    istringstream sin(line);
                    double tmp;
                    while (sin>>tmp)
                        feature.push_back(tmp);
                    getline(infile,line);
                }

                infile.close();
            } else
                return false;

            for (int i=0; i<feature.size(); i++)
                objFeatures << feature[i] << " ";
            objFeatures << endl;

        }
    }

    objFeatures.close();

    return true;
}

bool linearClassifier::createFullPath(const char * path)
{

    bool success = true;

    struct stat info;
    if (stat(path, &info)!=0)
    {
        string strPath = string(path);
        size_t found = strPath.find_last_of("/");

        while (strPath[found]=='/')
            found--;

        success = success && createFullPath(strPath.substr(0,found+1).c_str());

        int r = mkdir(strPath.c_str(), 0777);
        if (r==-1)
           return false;
    }

    return success;
}

bool linearClassifier::checkKnownObjects()
{

    knownObjects.clear();
    linearClassifiers_RLS.clear();

    struct stat info;
    if(stat(currPath.c_str(), &info)!=0)
    {
        createFullPath(currPath.c_str());
        return true;
    }

    vector<string> files;
    bool success = getdir(currPath,files);
    if (success==false)
        return false;

    for (int i=0; i< files.size(); i++)
    {
        if(!files[i].compare(".") || !files[i].compare("..") || !files[i].compare("rlsmodel"))
            continue;

        string objPath = currPath + "/" + files[i];
        string rlsPath = objPath + "/" + "rlsmodel";

        if (stat(rlsPath.c_str(), &info)==0)
            RLSlinear rls_model(files[i]);

        vector<string> featuresFile;
        vector<string> tmpFiles;
        success = getdir(objPath,featuresFile);
        if (success==false)
                return false;

        for (int j=0; j< featuresFile.size(); j++)
        {
            if(!featuresFile[j].compare(".") || !featuresFile[j].compare(".."))
                continue;

            string tmp = objPath + "/" + featuresFile[j];
            tmpFiles.push_back(tmp);
        }

        pair<string, vector<string> > obj(files[i],tmpFiles);
        knownObjects.push_back(obj);

    }

    return true;

}

bool linearClassifier::getdir(string dir, vector<string> &files)
{

    DIR *dp;
    struct dirent *dirp;

    if ((dp  = opendir(dir.c_str())) == NULL)
    {
        cout << "Error opening " << dir << endl;
        return false;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }

    closedir(dp);

    return true;
}

bool linearClassifier::recognize(string inputPath)
{

    if (linearClassifiers_RLS.size()==0)
        return false;

    countBuffer_RLS.resize(bufferSize);
    bufferScores_RLS.resize(bufferSize);

    for (int i=0; i<bufferSize; i++)
    {
        bufferScores_RLS[i].resize(linearClassifiers_RLS.size());
        countBuffer_RLS[i].resize(linearClassifiers_RLS.size());
    }

     vector <string> fileList;
     bool success = getdir(inputPath, fileList);
     if (success==false)
         return false;

     int current = 0;

     for (int i=0; i< fileList.size(); i++)
     {
         if (fileList[i].compare(".") && fileList[i].compare(".."))
         {

             // Read feature

             string line;
             vector<double> feature;

             string fullPath = inputPath + "/" + fileList[i];
             ifstream infile;
             infile.open(fullPath.c_str());

             if (infile.is_open())
             {
                 getline(infile,line);
                 while(!infile.eof())
                 {
                     istringstream sin(line);
                     double tmp;
                     while (sin>>tmp)
                         feature.push_back(tmp);
                     getline(infile,line);
                 }

                 infile.close();
             } else
                 return false;

            // Classify!

             string winnerClass_RLS;

             double maxVal_RLS = -1000;
             double minValue_RLS = 1000;
             double idWin_RLS = -1;

             gurls::gMat2D<float> feature_RLS(1,feature.size()+1);

             for(int i=0; i<feature.size(); i++)
                 feature_RLS(0,i) = feature[i];
             feature_RLS(0,feature.size()) = 1.0;

             for(int i=0; i<linearClassifiers_RLS.size(); i++)
             {
                 double value_RLS = linearClassifiers_RLS[i].predictModel(feature_RLS);

                 if(value_RLS>maxVal_RLS)
                 {
                     maxVal_RLS=value_RLS;
                     idWin_RLS=i;
                 }
                 if(value_RLS<minValue_RLS)
                     minValue_RLS=value_RLS;

                 bufferScores_RLS[current%bufferSize][i]=(value_RLS);
                 countBuffer_RLS[current%bufferSize][i]=0;
             }
             countBuffer_RLS[current%bufferSize][idWin_RLS]=1;

             vector<double> avgScores_RLS(linearClassifiers_RLS.size(),0.0);
             vector<double> bufferVotes_RLS(linearClassifiers_RLS.size(),0.0);

             for(int i=0; i<bufferSize; i++)
                 for(int k=0; k<linearClassifiers_RLS.size(); k++)
                 {
                     avgScores_RLS[k]=avgScores_RLS[k]+bufferScores_RLS[i][k];
                     bufferVotes_RLS[k]=bufferVotes_RLS[k]+countBuffer_RLS[i][k];
                 }

             double maxValue_RLS=-100;
             double maxVote_RLS=0;
             int indexClass_RLS=-1;
             int indexMaxVote_RLS=-1;

             for(int i =0; i<linearClassifiers_RLS.size(); i++)
             {
                 avgScores_RLS[i]=avgScores_RLS[i]/bufferSize;
                 if(avgScores_RLS[i]>maxValue_RLS)
                 {
                     maxValue_RLS=avgScores_RLS[i];
                     indexClass_RLS=i;
                 }
                 if(bufferVotes_RLS[i]>maxVote_RLS)
                 {
                     maxVote_RLS=bufferVotes_RLS[i];
                     indexMaxVote_RLS=i;
                 }
             }

             winnerClass_RLS = knownObjects[indexClass_RLS].first;

             if (maxVote_RLS/bufferSize<minVotePercentage)
                 winnerClass_RLS="?";

             for(int i =0; i<linearClassifiers_RLS.size(); i++)
             {
                 cout << knownObjects[i].first.c_str() << ": ";
                 cout << bufferScores_RLS[current%bufferSize][i];
             }
             cout << endl;

         }

         current++;
     }

     return true;

}

bool linearClassifier::forgetClass(string className, bool retrain)
{

    string classPath = currPath + "/" + className;

    struct stat info;
    if(stat(classPath.c_str(), &info)!=0)
        return true;

    vector<string> files;
    bool success = getdir(classPath,files);
    if (success==false)
        return false;

    for (int i=0; i< files.size(); i++)
    {
        if(!files[i].compare(".") || !files[i].compare(".."))
            continue;

        string feature=classPath+"/"+files[i];

        int r=remove(feature.c_str());
        if (r==-1)
            return false;
    }

    int res=rmdir(classPath.c_str());
    if (res==-1)
        return false;

    if (retrain)
    {
        success = trainClassifiers();
        if (success==false)
            return false;
    }

    return true;
}

bool linearClassifier::forgetAll()
{
    bool success = checkKnownObjects();
    if (success==false)
        return false;

    for (int i=0; i<knownObjects.size(); i++)
    {
        success = forgetClass(knownObjects[i].first, false);
        if (success==false)
            return false;
    }

    return true;
}

bool linearClassifier::getClassList(vector <string> &list)
{

    for(int i=0; i<knownObjects.size(); i++)
        list.push_back(knownObjects[i].first.c_str());

    return true;
}

bool linearClassifier::releaseModels()
{
    if (linearClassifiers_RLS.size()>0)
        for (int i=0; i<linearClassifiers_RLS.size(); i++)
            linearClassifiers_RLS[i].freeModel();

    return true;
}
