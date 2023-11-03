#ifndef OPTIMIZE_H
#define OPTIMIZE_H


#include <cstdlib>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problems/schwefel.hpp>
#include "TurnsNew.h"



using namespace Rcpp;
using namespace pagmo;

class OptimizeRL {
public:
  OptimizeRL();
  // Constructor
  OptimizeRL(const Rcpp::S4& ratdata_, const Rcpp::S4& testModel_,  
  const std::string& learningRule_, const int& sim_):
  ratdata(ratdata_),  testModel(testModel_), learningRule(learningRule_),  sim(sim_) {};


  // Destructor
  ~OptimizeRL() {}

  // Fitness function
  vector_double fitness(const vector_double& v) const;

  // Bounds function
  std::pair<vector_double, vector_double> get_bounds() const;


private:
  // Members
  const Rcpp::S4& ratdata;
  const Rcpp::S4& testModel;
  const std::string& learningRule; 
  const int & sim;
};



void optimizeRL_pagmo(const Rcpp::S4& ratdata, const Rcpp::S4& testModel, const std::string& learningRule, const int& sim);

#endif