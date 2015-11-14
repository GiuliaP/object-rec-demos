#include "gurls++/gurls.h"
//#include "gurls++/primal.h"
#include "gurls++/gurls.h"
#include "gurls++/traintest.h"

#include "RLSlinear.h"

using namespace gurls;

RLSlinear::RLSlinear(string className) {

    this->className=className;
    modelLinearRLS=NULL;

}

void RLSlinear::trainModel(gMat2D<float> &X, gMat2D<float> &Y)
{
    if(modelLinearRLS!=NULL)
        delete modelLinearRLS;

    /*
    // "Old interface"

    OptestTaskSequence *seq = new OptTaskSequence();

    OptProcess* process_train = new OptProcess();
    OptProcess* process_predict = new OptProcess();

    *seq << "kernel:linear" << "split:ho" << "paramsel:hodual";
    *process_train << GURLS::computeNsave << GURLS::compute << GURLS::computeNsave;
    *process_predict << GURLS::load << GURLS::ignore << GURLS::load;

    *seq<< "optimizer:rlsdual"<< "pred:dual";
    *process_train<<GURLS::computeNsave<<GURLS::ignore;
    *process_predict<<GURLS::load<<GURLS::computeNsave;

    GurlsOptionsList * processes = new GurlsOptionsList("processes", false);
    processes->addOpt("train",process_train);
    processes->addOpt("test",process_predict);

    modelLinearRLS = new GurlsOptionsList(className, true);

    modelLinearRLS->addOpt("seq", seq);
    modelLinearRLS->addOpt("processes", processes);

    RLS.run(X,Y,*modelLinearRLS,"train");

    */

    // "New" interface

    unsigned long n = X.rows();
    unsigned long d = X.cols();
    unsigned long t = Y.cols(); // should be 1

    modelLinearRLS = train(X.getData(), Y.getData(), n, d, t, "krls", "linear");

}

float RLSlinear::predictModel(gMat2D<float> &X)
{
    
    if(modelLinearRLS==NULL)
    {
        cout << "Error: train model first!" << endl;
        return 0.0;
    }
    
    /*
    // "Old" interface

    gMat2D<float> empty;
    RLS.run(X,empty,*modelLinearRLS,"test");
    */

    // "New" interface

    unsigned long nTest = X.rows();
    unsigned long d = X.cols();

    gMat2D<float> empty(nTest, 1);
    unsigned long t = empty.cols();

    float* perfBuffer = new float[empty.cols()];
    float* predBuffer = new float[empty.cols()*nTest];

    test(*modelLinearRLS, X.getData(), empty.getData(), predBuffer, perfBuffer, nTest, d, t, "auto");

    // Retrieve prediction

    gMat2D<float>& pred = modelLinearRLS->getOptValue<OptMatrix<gMat2D<float> > >("pred");
    
    return pred(0,0);
}

void RLSlinear::freeModel()
{
    if (modelLinearRLS!=NULL)
        delete modelLinearRLS;
}

