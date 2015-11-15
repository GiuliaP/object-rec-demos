

#include <iostream>
#include <string>

#include "gurls++/gurls.h"
#include "gurls++/gmat2d.h"

#include <vector>
#include <fstream>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

using namespace std;

class  RLSlinear
{
    private:

        gurls::GURLS RLS;
        string className;
        gurls::GurlsOptionsList* modelLinearRLS;

    public:

        RLSlinear(string className);

        void trainModel(gurls::gMat2D<float> &X, gurls::gMat2D<float> &Y, bool newGURLS);
        float predictModel(gurls::gMat2D<float> &X, bool newGURLS);

        void freeModel();

};
