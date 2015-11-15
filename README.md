# object-rec-demos
A repository containing a demo of a simple object recognition system

## Installation

##### Dependencies

- SFM, dispBlobber, caffeCoder, linearClassifier: [OpenCV](http://opencv.org/downloads.html), 
- SFM, dispBlobber, caffeCoder, linearClassifier: [Boost](http://www.boost.org/)
- caffeCoder: [Caffe](http://caffe.berkeleyvision.org/)
- LinearClassifier: [GURLS](https://github.com/LCSL/GURLS)

Each module performs a step of the recognition pipeline and can be compiled and run independently. The output of a module is the input of the following one.
All modules are supposed to work on the same folder tree. The input data will be provided.

Documentation about `caffeCoder` setup can be found in the [README_Caffe](https://github.com/robotology/himrep/blob/master/README_Caffe.md).
At the same link instructions about the installation of `OpenCV` and `Boost` can be found. For GURLS installation, please refer to the library documentation.

Once the dependences are installed (following the instructions), to build a module is sufficient to:

cd <module_folder> <br />
mkdir build <br />
cd build <br />
ccmake ../ <br />
make <br />

ensuring that in the configuration step all the dependences are met.