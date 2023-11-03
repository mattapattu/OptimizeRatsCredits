#include "optimize.h"
#include "utils.h"
#include <Rcpp.h>


// [[Rcpp::export]]
void callOptimize_pagmo(const Rcpp::S4 ratdata, const Rcpp::S4 modelData, const Rcpp::S4 testModel, const int sim) {

    //Rcpp::Rcout << "Calling optimizeRL_pagmo" <<std::endl;
    
    optimizeRL_pagmo(ratdata, modelData, testModel, sim);


    return;
}
