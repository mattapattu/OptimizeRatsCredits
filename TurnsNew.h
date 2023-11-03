#ifndef TURNSNEW_H
#define TURNSNEW_H

// #include "aca3.hpp"
// #include "sarsa.hpp"
// #include "aca2.hpp"
// #include "aca4.hpp"
// #include "avgRewardQLearning.hpp"
// #include "discountedRwdQlearning.hpp"
#include "utils.h"



using namespace Rcpp;


Rcpp::List simulateTurnsModels(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, arma::vec turnStages);
std::vector<double> getTurnsLikelihood(Rcpp::S4 ratdata, std::vector<double> params, const std::string& learningRule, Rcpp::S4 testModel, int sim);
arma::mat getProbMatrix(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim);

Rcpp::List simulateDiscountedRwdQlearning(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, Rcpp::S4 turnModel, arma::vec turnStages, bool debug);
std::vector<double> getDiscountedRwdQlearningLik(Rcpp::S4 ratdata, std::vector<double> params, Rcpp::S4 testModel, int sim, bool debug);
arma::mat getDiscountedRwdQlearningProbMat(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug);
Rcpp::List getDiscountedRwdQlearningProbMat2(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug);

Rcpp::List simulateQLearn(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, Rcpp::S4 turnModel, arma::vec turnStages, bool debug);
std::vector<double> getQLearningLik(Rcpp::S4 ratdata, std::vector<double> params, Rcpp::S4 testModel, int sim, bool debug);
arma::mat getQLearningProbMat(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug);
arma::mat getQLearningProbMat2(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug);


Rcpp::List simulateAca2TurnsModels(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, Rcpp::S4 turnModel, arma::vec turnStages, bool debug);
std::vector<double> getAca2Likelihood(Rcpp::S4 ratdata, std::vector<double> params, Rcpp::S4 testModel, int sim, bool debug);
arma::mat getAca2ProbMatrix(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug);
arma::mat getAca2ProbMatrix2(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug);

#endif