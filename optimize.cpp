#include "optimize.h"
#include <pagmo/problems/schwefel.hpp>
#include <RInside.h>


using namespace pagmo;

pagmo::vector_double OptimizeRL::fitness(const pagmo::vector_double& v) const
{
    std::string model = Rcpp::as<std::string>(testModel.slot("Name"));
    std::vector<double> params(4);
    params[0]=v[0];
    params[1]=v[1];
    params[2]=0.5;
    params[3]=0;

    std::vector<double> lik = getTurnsLikelihood(ratdata, params, learningRule, testModel, sim);
    
    double result = 0.0;
    result = std::accumulate(lik.begin(), lik.end(), 0.0);
    result = result*(-1);
    //std::cout << "v[0] =" << v[0]  << ", v[1]="<< v[1]<< ", result=" <<result <<std::endl;
    return{result};
}


  std::pair<pagmo::vector_double, pagmo::vector_double> OptimizeRL::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;
    if(learningRule == "aca2")
    {
        bounds.first={0,0};
        bounds.second={1,1};
    }
    if(learningRule == "qlearningAvgRwd")
    {
        bounds.first={0,1e-9};
        bounds.second={1,1e-6};

    }
    else if(learningRule == "qlearningDisRwd")
    {
      bounds.first={0,0};
        bounds.second={1,1};
    }
    return(bounds);
  }


void optimizeRL_pagmo(const Rcpp::S4& ratdata, const Rcpp::S4& testModel, const std::string& learningRule, const int& sim) {

    std::cout << "Initializing problem class" <<std::endl;
    // Create a function to optimize
    OptimizeRL orl(ratdata,testModel,learningRule,sim);
    std::cout << "Initialized problem class" <<std::endl;

    // Create a problem using Pagmo
    problem prob{orl};
    //problem prob{schwefel(30)};
    
    std::cout << "created problem" <<std::endl;
    // 2 - Instantiate a pagmo algorithm (self-adaptive differential
    // evolution, 100 generations).
    pagmo::algorithm algo{sade(10,2,2)};

    std::cout << "creating archipelago" <<std::endl;
    // 3 - Instantiate an archipelago with 5 islands having each 5 individuals.
    archipelago archi{5u, algo, prob, 7u};

    // 4 - Run the evolution in parallel on the 5 separate islands 5 times.
    archi.evolve(5);
    std::cout << "DONE1:"  << '\n';
    //system("pause"); 

    // 5 - Wait for the evolutions to finish.
    archi.wait_check();

    // 6 - Print the fitness of the best solution in each island.
    

    //system("pause"); 

    for (const auto &isl : archi) {
        std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
        std::vector<double> dec_vec = isl.get_population().champion_x();
        for (auto const& i : dec_vec)
             std::cout << i << ", ";
        std::cout << "\n" ;
    }
    // sink();


    return;
}



Rcpp::S4 loadAndParseRData(RInside R, const std::string& rdataFilePath, const std::string& objectName) {

  std::cout <<"Inside loadAndParseRData" <<std::endl;

  // Initialize RInside
  

  // Load the Rdata file
  R.parseEvalQ("load('" + rdataFilePath + "')");

  // Parse the S4 class object
  Rcpp::S4 s4Object = R.parseEval("get('" + objectName + "')");
  //Rcpp::NumericMatrix m = s4Object.slot("allpaths");
  std::cout <<"Returning S4 obj" <<std::endl;
  return s4Object;
}


int main() {
  std::cout <<"Inside main" <<std::endl;
  // Replace with the path to your Rdata file and the S4 object name
  std::string rdataFilePath = "/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/ratdata.Rdata";
  std::string s4ObjectName = "ratdata";
  RInside R;
  // Load and parse the Rdata file
//   std::cout <<"Load and parse the ratdata file" <<std::endl;
//   Rcpp::S4 ratdata = loadAndParseRData(R, rdataFilePath, s4ObjectName);
  
//   rdataFilePath = "/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/Hybrid3.Rdata";
//   s4ObjectName = "Hybrid3";

//   Rcpp::S4 Hybrid3 = loadAndParseRData(R, rdataFilePath, s4ObjectName);

    
    std::string cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/ratdata.Rdata')";
    R.parseEvalQ(cmd);                  
    Rcpp::S4 ratdata = R.parseEval("get('ratdata')");

    cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/Hybrid3.Rdata')";
    R.parseEvalQ(cmd);                  
    Rcpp::S4 Hybrid3 = R.parseEval("get('Hybrid3')"); 
   
   std::cout <<"Calling optimizeRL_pagmo" <<std::endl;  
   
//    std::vector<double> params = {0.606825, 8.52906e-07, 0.5, 0};
//    std::vector<double> lik = getTurnsLikelihood(ratdata, params, "qlearningAvgRwd", Hybrid3, 2);
    
//    double result = 0.0;
//    result = std::accumulate(lik.begin(), lik.end(), 0.0);
//    result = result*(-1);
//     std::cout << "result=" <<result <<std::endl;
   
   optimizeRL_pagmo(ratdata,Hybrid3,"qlearningAvgRwd",2);
  
  return 0;
}




