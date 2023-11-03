#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include <regex>
#include <cmath>
#include <vector>
#include <regex>
#include "tree.h"

int aca_getNextState(int curr_state, int action, int last_turn);
double getAlphaPrime(double alpha, int episodeNb);
Edge softmax_action_sel(Graph graph, std::vector<Edge> edges);
void set_seed(double seed);
std::vector<double> testSample(Rcpp::IntegerVector actions, arma::vec prob, double d, bool setSeed);
Rcpp::IntegerVector cumsum1(Rcpp::IntegerVector x);
std::vector<double> quartiles(std::vector<double> samples);
arma::vec simulateTurnDuration(arma::mat turnTimes, arma::mat allpaths, int turnId, int turnNb, arma::vec turnStages, Rcpp::List nodeGroups, bool debug);
arma::mat simulatePathTime(arma::mat turnTimes, arma::mat allpaths, int pathNb, int path, arma::vec pathStages);

#endif