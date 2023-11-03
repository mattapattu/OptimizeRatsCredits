#include "utils.h"
#include <RcppArmadilloExtensions/sample.h>

using namespace Rcpp;

int aca_getNextState(int curr_state, int action, int last_turn)
{
  //Rcpp::Rcout << "curr_state=" << curr_state << ", action=" << action << ", last_turn=" << last_turn << std::endl;
  int new_state = -1;
  if (action == 4 || action == 5)
  {
    new_state = curr_state;
  }
  else if (action == 6)
  {
    if (last_turn == 4 || last_turn == 7 || last_turn == 12 || last_turn == 15)
    {
      new_state = 1;
    }
    else if (last_turn == 5 || last_turn == 6 || last_turn == 13 || last_turn == 14)
    {
      new_state = 0;
    }
  }
  else if (curr_state == 0)
  {
    new_state = 1;
  }
  else if (curr_state == 1)
  {
    new_state = 0;
  }
  
  //Rcpp::Rcout << "new_state=" << new_state << std::endl;
  
  return (new_state);
}

double getAlphaPrime(double alpha, int episodeNb)
{
  double power = 0;
  double denominator = std::pow(episodeNb, power);
  double alphaPrime = alpha/denominator;
  
  return(alphaPrime);
  
}


Edge softmax_action_sel(Graph graph, std::vector<Edge> edges)
{
  
  std::vector<double> probVec;
  for (auto edge = edges.begin(); edge != edges.end(); edge++)
  {
    probVec.push_back((*edge).probability);
  }
  arma::vec probVec_arma(probVec);
  //Rcpp::Rcout <<"creditVec_arma="<<creditVec_arma<<std::endl;
  
  IntegerVector actions = seq(0, (edges.size() - 1));
  int action_selected = Rcpp::RcppArmadillo::sample(actions, 1, true, probVec_arma)[0];
  //int action_selected = Rcpp::sample(actions, 1, true, probVec_arma)[0];
  //Rcpp::Rcout <<"action_selected="<<action_selected<<std::endl;
  
  return (edges[action_selected]);
}

// [[Rcpp::export]]
void set_seed(double seed) {
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(std::floor(std::fabs(seed)));
}


// [[Rcpp::export]]
 std::vector<double> testSample(Rcpp::IntegerVector actions, arma::vec prob, double d, bool setSeed)
{
    if(setSeed)
    {
      set_seed(d);
    }
    
    //int action_selected = Rcpp::RcppArmadillo::sample(actions, 1, true, prob)[0];
    std::vector<double> actVec;
    for(int i = 0; i < 10; i++)
    {
      int action_selected = Rcpp::RcppArmadillo::sample(actions, 1, true, prob)[0];
      actVec.push_back(action_selected);
    }
  //int action_selected = Rcpp::sample(actions, 1, true, probVec_arma)[0];
  //Rcpp::Rcout <<"action_selected="<<action_selected<<std::endl;
  
  return (actVec);
}


Rcpp::IntegerVector cumsum1(Rcpp::IntegerVector x)
{
  // initialize an accumulator variable
  double acc = 0;
  // initialize the result vector
  Rcpp::IntegerVector res(x.size());
  for (int i = 0; i < x.size(); i++)
  {
    acc += x[i];
    res[i] = acc;
  }
  return res;
}

// [[Rcpp::export]]
std::vector<double> quartiles(std::vector<double> samples)
{
    // return as vector containing {first quartile, median, third quartile}
    std::vector<double> answer;
    int size = samples.size();
    std::sort(samples.begin(), samples.end());
    // First Quartile
    answer.push_back(samples[size/4]);
    // Second Quartile = Median
    if (size % 2 == 0)
        answer.push_back((samples[size / 2 - 1] + samples[size / 2]) / 2);
    else
        answer.push_back(samples[size / 2]);
    // Third Quartile
    answer.push_back(samples[size*3/4]);
    return answer;
}

// [[Rcpp::export]]
arma::vec simulateTurnDuration(arma::mat turnTimes, arma::mat allpaths, int turnId, int turnNb, arma::vec turnStages, Rcpp::List nodeGroups, bool debug)
{
  //Rcpp::Rcout <<"Inside simulateTurnDuration" << std::endl;
  

  int start = -1;
  int end = 0;
  if(turnNb < turnStages(1))
  {
    start = 1;
    end = turnStages(1)-1;
  }
  else if(turnNb >= turnStages(1) && turnNb < turnStages(2))
  {
    start = turnStages(1);
    end = turnStages(2)-1;
  }
  else if(turnNb >= turnStages(2))
  {
    start = turnStages(2);
    end = turnStages(3);
  }

  start = start-1;
  end = end-1;
  //Rcpp::Rcout << "start=" << start << ", end=" << end << std::endl;
  
  arma::mat turnTimes_submat = turnTimes.rows(start,end);
  arma::vec turnId_submat = turnTimes_submat.col(3);

  
  Rcpp::IntegerVector idx;
  for(int i=0;i<nodeGroups.size();i++)
  {
    Rcpp::CharacterVector vecGrp = Rcpp::as<Rcpp::CharacterVector>(nodeGroups[i]);
    Rcpp::CharacterVector turnIds = Rcpp::as<Rcpp::CharacterVector>(Rcpp::wrap(turnId_submat));
    
     Rcpp::CharacterVector table(1);
     table(0) = turnId;
     Rcpp::IntegerVector vec =  Rcpp::match(vecGrp , table ) ;
     //Rcpp::Rcout <<"TurnId=" <<turnId << ", vecGrp=" <<vecGrp << std::endl;
     
     //bool res = Rcpp::any(vec == 1);
     if(std::find(vec.begin(), vec.end(), 1)!=vec.end())
     {
       //Rcpp::Rcout <<"TurnId = " << turnId << ", Matched vecGroup =" <<vecGrp  << std::endl;
       //Rcpp::Rcout << "turnIds=" << turnIds << std::endl;
       //Rcpp::Rcout <<"v=" <<v << std::endl;
       Rcpp::IntegerVector vec1 =  Rcpp::match(turnIds, vecGrp) ;
       //Rcpp::Rcout <<"vec1=" <<vec1 << std::endl;
       Rcpp::IntegerVector v = Rcpp::seq(0, turnId_submat.size()-1);
       idx = v[!Rcpp::is_na(vec1)];
       //Rcpp::Rcout <<"idx=" <<idx << std::endl;
       break;
     }
  }
  
  arma::uvec arma_idx = Rcpp::as<arma::uvec>(idx);
  //Rcpp::Rcout <<"arma_idx.size =" << arma_idx.n_elem << std::endl;
 
  arma::vec turndurations_submat = turnTimes_submat.col(5);
  arma::mat submat_sample = turnTimes_submat.rows(arma_idx);
  //Rcpp::Rcout <<"submat_sample=" <<submat_sample << std::endl;
  arma::vec sample;
  sample = submat_sample.col(5);
  //Rcpp::Rcout <<"sample=" <<sample << std::endl;

  std::vector<double> q = quartiles(arma::conv_to<std::vector<double>>::from(sample));
  arma::uvec final_sample_ids = arma::find(sample >= q[0] && sample <= q[2]);
  //arma::vec fin_sample = sample.elem(final_sample_ids);
  arma::vec pvec(final_sample_ids.n_elem); 
  double probability = (double) 1/(double) final_sample_ids.n_elem;
  pvec.fill(probability);
  arma::uword sampled_id = Rcpp::RcppArmadillo::sample(final_sample_ids, 1, true, pvec)[0];
    
   //logger.Print(msg.str()); 

  arma::rowvec turnRow = submat_sample.row(sampled_id);
  arma::uvec cols; //3 = ActionNb, 5 = actionNb
  cols = {0,5};
   //msg.str("");
   //msg << "turnRow=";
  //logger.PrintArmaRowVec(msg.str(),turnRow); 
  arma::vec turnDurations = turnRow.elem(cols);

    
   //logger.PrintArmaVec(msg.str(),turnDurations); 

  return(turnDurations);
}



arma::mat simulatePathTime(arma::mat turnTimes, arma::mat allpaths, int pathNb, int path, arma::vec pathStages)
{
  std::vector<int> grp1 = {0};
  std::vector<int> grp2 = {1};
  std::vector<int> grp3 = {3,4};
  std::vector<int> grp4 = {2,5};


  int start = -1;
  int end = 0;
  if(pathNb < pathStages(1))
  {
    start = 1;
    end = pathStages(1)-1;
  }
  else if(pathNb >= pathStages(1) && pathNb < pathStages(2))
  {
    start = pathStages(1);
    end = pathStages(2)-1;
  }
  else if(pathNb >= pathStages(2))
  {
    start = pathStages(2);
    end = pathStages(3);
  }

  start = start-1;
  end = end-1;
  Rcpp::Rcout << "start=" << start << ", end=" << end << std::endl;
  arma::mat allpaths_submat = allpaths.rows(start,end);
  arma::vec path_submat = allpaths_submat.col(0) - 1;
  arma::uvec allpathsubmat_idx;
  if(std::find(grp1.begin(), grp1.end(), path) != grp1.end())
  {
    Rcpp::Rcout << "Here1" << std::endl;
    allpathsubmat_idx = arma::find(path_submat == 0);
  }
  else if(std::find(grp2.begin(), grp2.end(), path) != grp2.end())
  {
    Rcpp::Rcout << "Here2" << std::endl;
    allpathsubmat_idx = arma::find(path_submat == 1);
  }
  else if(std::find(grp3.begin(), grp3.end(), path) != grp3.end())
  {
    Rcpp::Rcout << "Here3" << std::endl;
    allpathsubmat_idx = arma::find( path_submat == 3||path_submat == 4 );
  }
  else if(std::find(grp4.begin(), grp4.end(), path) != grp4.end())
  {
    Rcpp::Rcout << "Here4" << std::endl;
    allpathsubmat_idx = arma::find(path_submat == 2 || path_submat == 5);
  }
  
 Rcpp::Rcout << "allpathsubmat_idx.n_elem=" << allpathsubmat_idx.n_elem << std::endl;
  
  arma::mat allpath_submat2 = allpaths_submat.rows(allpathsubmat_idx);
  arma::vec sample = allpath_submat2.col(3);
  std::vector<double> q = quartiles(arma::conv_to<std::vector<double>>::from(sample));
  arma::uvec final_sample_ids = arma::find(sample >= q[0] && sample <= q[2]);
  double probability = (double) 1/(double) final_sample_ids.n_elem;
  arma::vec pvec(final_sample_ids.n_elem); 
  pvec.fill(probability);
  arma::uword sampled_id = Rcpp::RcppArmadillo::sample(final_sample_ids, 1, true, pvec)[0];
  int actionNb = allpath_submat2(sampled_id,5);
  arma::uvec turnIdx = arma::find(turnTimes.col(0) == actionNb);
  arma::mat turn_submat = turnTimes.rows(turnIdx);
  arma::uvec colIds = {0,5}; //3 = ActionNb, 5 = actionNb
  arma::mat turnDurations = turn_submat.cols(colIds);
  return(turnDurations);
}
