// Description: This file declares a struct for storing per-execution configuration information.

#include <iostream>
#include <string>

// store all of our command-line configuration parameters

struct config_t {

    int clusters;

    int seed;

    int threads;

    int total_points;

    int seed2;

    // simple constructor
    config_t() : clusters(80), seed(0) , threads(-1),total_points(-1){ }

};