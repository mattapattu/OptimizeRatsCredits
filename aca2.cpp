#include "TurnsNew.h"
#include "aca3CreditUpdate.hpp"
//#include "utils.h"
//using namespace Rcpp;

//Function simulateTurnTimeFromR = Environment::global_env()["simulateTurnTime"];



//namespace aca3 {
Rcpp::List simulateAca2TurnsModels(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, Rcpp::S4 turnModel, arma::vec turnStages, bool debug)
{
  //Rcpp::Rcout << "Inside simulateAca2TurnsModels" << std::endl;
  arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  std::string model = Rcpp::as<std::string>(modelData.slot("Model"));
  //Rcpp::Rcout << "model=" << model << std::endl;
  arma::mat turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));;
  
  //Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(testModel.slot("nodeGroups"));
  
  double alpha = Rcpp::as<double>(modelData.slot("alpha"));
  double gamma = Rcpp::as<double>(modelData.slot("gamma1"));
  double reward = Rcpp::as<double>(modelData.slot("gamma2"));
  reward = reward * 10;  
  int episodeNb = 0;
  
  //Rcpp::Rcout << "alpha=" << alpha << ", gamma1=" << gamma << std::endl;
  
  Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(turnModel.slot("nodeGroups"));
  
  
  
  arma::mat R = arma::zeros(2, 6);
  R(0, 3) = reward;
  R(1, 3) = reward;
  arma::mat generated_PathData;
  arma::mat generated_TurnData;
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  //arma::vec allpath_rewards = allpaths.col(2);
  arma::vec allpath_duration = allpaths.col(3);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  arma::vec pathNb = allpaths.col(5);
  
  //arma::vec all_turns = turnTimes.col(3);
  //arma::vec turns_sessions = turnTimes.col(4);
  
  //Rcpp::Rcout << "sessionVec=" << sessionVec << std::endl;
  //Rcpp::Rcout << "uniqSessIdx=" << uniqSessIdx << std::endl;
  Graph S0(testModel, 0);
  Graph S1(testModel, 1);
  
  Graph turnsS0(turnModel, 0);
  Graph turnsS1(turnModel, 1);
  
  int actionNb = 0;
  
  // Loop through each session
  for (unsigned int session = 0; session < (uniqSessIdx.n_elem); session++)
  {
    
    int sessId = uniqSessIdx(session);
    //Rcpp::Rcout << "session=" << session << ", sessId=" << sessId << std::endl;
    arma::uvec sessionIdx = arma::find(sessionVec == (sessId));
    arma::vec actions_sess = allpath_actions.elem(sessionIdx);
    arma::vec states_sess = allpath_states.elem(sessionIdx);
    
    //arma::uvec turns_sessIdx = arma::find(turns_sessions == (sessId));
    //arma::vec turns_sess = all_turns.elem(turns_sessIdx);
    
    int initState = 0;
    bool changeState = false;
    bool returnToInitState = false;
    bool resetVector = true;
    int nrow = actions_sess.n_rows;
    double avg_score = 0;
    double score_episode = 0;
    int episode = 1;
    
    int S = states_sess(0) - 1;
    std::vector<std::string> episodeTurns;
    std::vector<int> episodeTurnStates;
    std::vector<double> episodeTurnTimes;
    
    arma::mat generated_PathData_sess(nrow, 7);
    arma::mat generated_TurnsData_sess((nrow * 3), 7);
    generated_PathData_sess.fill(-1);
    generated_TurnsData_sess.fill(-1);
    unsigned int turnIdx = 0; // counter for turn model
    //All episodes in new session
    //Rcpp::Rcout << "nrow=" << nrow << std::endl;
    for (int i = 0; i < nrow; i++)
    {
      actionNb++;
      if (resetVector)
      {
        initState = S;
        //Rcpp::Rcout <<"initState="<<initState<<std::endl;
        resetVector = false;
      }
      //Rcpp::Rcout <<"i=" <<i <<", S=" <<S  << std::endl;
      Node *currNode;
      std::vector<Edge> *edges;
      Rcpp::StringVector turnNames;
      Rcpp::StringVector testTurnNames;
      Graph graph;
      Graph turnsGraph;
      if (S == 0)
      {
        graph = S0;
        edges = graph.getOutgoingEdges("E");
        turnsGraph = turnsS0;
      }
      else
      {
        graph = S1;
        edges = graph.getOutgoingEdges("I");
        turnsGraph = turnsS1;
      }
      
      
      double pathDuration = 0;
      //HERE  the nodes based on testModel from one reward box to another are selected 
      while (!edges->empty())
      {
        Edge edgeSelected = softmax_action_sel(graph, *edges);
        std::string turnSelected = edgeSelected.dest->node;
        //Rcpp::Rcout << "Turn=" << turnSelected <<std::endl;
        
        //Convert the selected edge to TurnModel components
        //Rcpp::Rcout << "Turn=" << turnSelected  << ", turnNb=" << turnNb <<std::endl;
        currNode = edgeSelected.dest;
        //Rcpp::Rcout << "turnSelected =" << turnSelected <<std::endl;
        
        //Rcpp::Rcout << "turnTime=" << turnTime <<std::endl;
        testTurnNames.push_back(turnSelected);
        episodeTurns.push_back(currNode->node);
        episodeTurnStates.push_back(S);
        //Rcpp::Rcout << "model=" << model <<std::endl;
        if(model == "Turns")
        {
          //std::string turnName = currNode->node;
          //turnNames.push_back(turnSelected);
          int componentId = turnsGraph.getNodeIndex(turnSelected);     
          
          arma::vec durationVec = simulateTurnDuration(turnTimes, allpaths, componentId, (turnIdx+1), turnStages,nodeGroups,debug);
          double turnTime = durationVec(1);
          
          
          pathDuration = pathDuration + turnTime;
          episodeTurnTimes.push_back(turnTime);
          
          generated_TurnsData_sess(turnIdx, 0) = componentId;
          generated_TurnsData_sess(turnIdx, 1) = S;
          generated_TurnsData_sess(turnIdx, 2) = 0;
          generated_TurnsData_sess(turnIdx, 3) = turnTime;
          //Rcpp::Rcout << "Turn=" << turnSelected <<", turnDuration="<< turnTime<<std::endl;
          generated_TurnsData_sess(turnIdx, 4) = sessId;
          generated_TurnsData_sess(turnIdx, 5) = actionNb;
          generated_TurnsData_sess(turnIdx, 6) = durationVec(0);
          turnIdx++;
          
        }
        
        edges = graph.getOutgoingEdges(currNode->node);
      }
      
      //Rcpp::Rcout << "testTurnNames=" << testTurnNames <<std::endl;
      //graph.printPaths();
      int A = graph.getPathFromTurns(testTurnNames);
      //Rcpp::Rcout << "S=" <<S << ", A=" << A <<std::endl;
      
      if(model != "Turns")
      {
        Rcpp::StringVector turnComponents = turnsGraph.getTurnsFromPaths(A);
        //Rcpp::Rcout << "turnComponents=" << turnComponents <<std::endl;
        arma::mat turnDurationMat(turnComponents.size(),2);
        turnDurationMat.zeros();
        
        arma::mat testTurnDurationMat(testTurnNames.size(),2);
        testTurnDurationMat.zeros();
        
        // To generate TurnModel durations, the path is converted to TurnModel components
        // For each TurnModel component, a duration is simulated
        for(int k=0; k < turnComponents.size(); k++ ){
          
          std::string turnName1 = Rcpp::as<std::string>(turnComponents[k]);
          turnNames.push_back(turnName1);
          int componentId = turnsGraph.getNodeIndex(turnName1);     
          //Rcpp::Rcout << "turnName=" << turnName1 << ", turnId=" << componentId << std::endl;
          arma::vec durationVec = simulateTurnDuration(turnTimes, allpaths, componentId, (turnIdx+1), turnStages,nodeGroups,debug);
          double turnTime = durationVec(1);
          
          turnDurationMat(k,0) = componentId;
          turnDurationMat(k,1) = turnTime;
          //episodeTurnTimes.push_back(turnTime);
          //Rcpp::Rcout << "turnDurationMat is" << std::endl << turnDurationMat << std::endl;
          pathDuration = pathDuration + turnTime;
          
          generated_TurnsData_sess(turnIdx, 0) = componentId;
          generated_TurnsData_sess(turnIdx, 1) = S;
          generated_TurnsData_sess(turnIdx, 2) = 0;
          generated_TurnsData_sess(turnIdx, 3) = turnTime;
          //Rcpp::Rcout << "Turn=" << turnName1 <<", turnDuration="<< turnTime<<std::endl;
          generated_TurnsData_sess(turnIdx, 4) = sessId;
          generated_TurnsData_sess(turnIdx, 5) = actionNb;
          generated_TurnsData_sess(turnIdx, 6) = durationVec(0);
          turnIdx++;
          
        }
        
        // Based on the TurnModel durations, the durations of the testModel components 
        // are determined.
        
        for(int k=0; k < testTurnNames.size(); k++ )
        {
          std::string node = Rcpp::as<std::string>(testTurnNames(k));
          
          //Decompose hybrid node to get the TurnModel nodes
          Rcpp::StringVector turnNodes  = graph.getTurnNodes(node);
          //Rcpp::Rcout <<"S=" << S << ", testTurn=" << node << ", turnNodes=" << turnNodes <<std::endl;
          // Get the nodeIds corresponding to the TurnModel
          Rcpp::IntegerVector turnNodeIds = turnsGraph.getNodeIds(turnNodes);
          double testNodeDuration = 0;
          //Rcpp::Rcout << "turnDurationMat is" << std::endl << turnDurationMat << std::endl;
          //Rcpp::Rcout << "turnNodeIds:" <<  turnNodeIds << std::endl;
          for(int j=0; j<turnNodeIds.size();j++)
          {
            arma::uvec id = arma::find(turnDurationMat.col(0) == turnNodeIds[j]);
            //Rcpp::Rcout << "id:" << id << std::endl;
            testNodeDuration = testNodeDuration + turnDurationMat(id(0),1);
          }
          //Rcpp::Rcout << "testNode=" << node <<", testNodeDuration="<< testNodeDuration<<std::endl;
          
          episodeTurnTimes.push_back(testNodeDuration);
        }
        
      }
      
      
      
      //arma::mat durationMat = simulatePathTime(turnTimes, allpaths, actionNb, A, pathStages,nodeGroups);
      
      //Rcpp::Rcout <<"A=" << A << ", S=" << S << ", sessId=" <<sessId<< std::endl;
      generated_PathData_sess(i, 0) = A;
      generated_PathData_sess(i, 1) = S;
      //Rcpp::Rcout <<"R(S, A)=" <<R(S, A)<< std::endl;
      generated_PathData_sess(i, 2) = R(S, A);
      generated_PathData_sess(i, 3) = pathDuration;
      generated_PathData_sess(i, 4) = sessId;
      generated_PathData_sess(i, 5) = actionNb;
      
      
      if (R(S, A) > 0)
      {
        //Rcpp::Rcout << "turnNb=" << generated_TurnsData_sess((turnIdx - 1), 0) << ", receives reward"<< std::endl;
        generated_TurnsData_sess((turnIdx - 1), 2) = reward;
        score_episode = score_episode + reward;
      }
      
      int last_turn = generated_TurnsData_sess((turnIdx - 1), 0);
      
      int S_prime = aca_getNextState(S, A, last_turn);
      
      if (S_prime != initState)
      {
        changeState = true;
      }
      else if (S_prime == initState && changeState)
      {
        returnToInitState = true;
      }
      
      if (returnToInitState || (i==nrow-1))
      {
        //Rcpp::Rcout << "Inside end episode" << std::endl;
        changeState = false;
        returnToInitState = false;
        //Rcpp::Rcout << "episodeTurns.size=" <<episodeTurns.size() << ", episodeTurnStates.size()=" <<episodeTurnStates.size() << ", episodeTurnTimes.size=" << episodeTurnTimes.size()<< std::endl;
        // for(unsigned int i=0; i<episodeTurns.size(); i++)
        // {
        //   std::cout << episodeTurns[i] << "," <<episodeTurnStates[i] << "," <<episodeTurnTimes[i] << "; ";
        // }
        
        episodeNb = episodeNb+1;
        double alpha_prime = getAlphaPrime(alpha,episodeNb);       
        
        Aca3CreditUpdate(episodeTurns, episodeTurnStates, episodeTurnTimes, alpha_prime, score_episode, &S0, &S1);
        //Rcpp::Rcout << "Here7" << std::endl;
        S0.updateEdgeProbs(false);
        S1.updateEdgeProbs(false);
        //Rcpp::Rcout <<  "H="<<H<<std::endl;
        score_episode = 0;
        episode = episode + 1;
        resetVector = true;
        episodeTurns.clear();
        episodeTurnStates.clear();
        episodeTurnTimes.clear();
      }
      
      //Rcpp::Rcout << "Here1" << std::endl;
      //Rcpp::Rcout << "Here2" << std::endl;
      
      S = S_prime;
    }
    
    //Rcpp::Rcout << "Here3" << std::endl;
    S0.decayCredits(gamma);
    S1.decayCredits(gamma);
    S0.updateEdgeProbs(false);
    S1.updateEdgeProbs(false);
    
    
    generated_TurnData = arma::join_cols(generated_TurnData, generated_TurnsData_sess.rows(0, (turnIdx - 1)));
    generated_PathData = arma::join_cols(generated_PathData, generated_PathData_sess);
  }
  return (Rcpp::List::create(Named("PathData") = generated_PathData, _["TurnData"] = generated_TurnData));
}

std::vector<double> getAca2Likelihood(Rcpp::S4 ratdata, std::vector<double> params, Rcpp::S4 testModel, int sim, bool debug)
{
  arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  std::string model = Rcpp::as<std::string>(testModel.slot("Name"));
  arma::mat turnTimes;
  
  if(model == "Paths")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  }
  else if(model == "Turns")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
  }
  else if(model == "Hybrid1")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel1"));
  }
  else if(model == "Hybrid2")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel2"));
  }
  else if(model == "Hybrid3")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
  }
  else if(model == "Hybrid4")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel4"));
  }
  
  //Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(testModel.slot("nodeGroups"));
  
  double alpha = params[0];
  double gamma = params[1];
  double reward = params[2];
  reward = reward * 10;
  double power = params[3];
  
  
  int episodeNb = 0; 
  
  //Rcpp::Rcout <<  "allpaths.col(4)="<<allpaths.col(4) <<std::endl;
  
  std::vector<double> mseMatrix;
  //int mseRowIdx = 0;
  
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  
  arma::vec turnTime_method;
  if (sim == 1)
  {
    turnTime_method = turnTimes.col(3);
  }
  else if (model == "Paths")
  {
    turnTime_method = turnTimes.col(3);
  }
  else
  {
    turnTime_method = turnTimes.col(5);
  }
  
  int episode = 1;
  
  Graph S0(testModel, 0);
  //S0.printGraph();
  Graph S1(testModel, 1);
  //S1.printGraph();
  
  //Rcpp::Rcout <<"rootS1.turn ="<<rootS1->turn<<std::endl;
  //Rcpp::Rcout <<"rootS2.turn ="<<rootS2->turn<<std::endl;
  
  for (unsigned int session = 0; session < uniqSessIdx.n_elem; session++)
  {
    
    int sessId = uniqSessIdx(session);
    //Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
    arma::uvec sessionIdx = arma::find(sessionVec == sessId);
    arma::vec actions_sess = allpath_actions.elem(sessionIdx);
    arma::vec states_sess = allpath_states.elem(sessionIdx);
    arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);
    
    arma::uvec turnTimes_idx; 
    if (model == "Paths")
    {
      turnTimes_idx = arma::find(sessionVec == sessId); ;
    }
    else
    {
      turnTimes_idx = arma::find(turnTimes.col(4) == sessId); 
    }
    arma::vec turn_times_session = turnTime_method.elem(turnTimes_idx);
    arma::uword session_turn_count = 0;
    
    int initState = 0;
    bool changeState = false;
    bool returnToInitState = false;
    int score_episode = 0;
    float avg_score = 0;
    bool resetVector = true;
    int nrow = actions_sess.n_rows;
    int S;
    if(sim == 1)
    {
      S = states_sess(0); 
    }
    else
    {
      S = states_sess(0) - 1; 
    }
    int A = 0;
    std::vector<std::string> episodeTurns;
    std::vector<int> episodeTurnStates;
    std::vector<double> episodeTurnTimes;
    
    for (int i = 0; i < nrow; i++)
    {
      
      if (resetVector)
      {
        initState = S;
        //Rcpp::Rcout <<"initState="<<initState<<std::endl;
        resetVector = false;
      }
      
      int R = rewards_sess(i);
      
      if (R > 0)
      {
        score_episode = score_episode + reward;
      }
      
      if (sim == 1)
      {
        A = actions_sess(i);
      }
      else
      {
        A = actions_sess(i) - 1;
      }
      
      int S_prime = 0;
      if(i < (nrow-1))
      {
        if (sim == 1)
        {
          S_prime = states_sess(i + 1);
        }
        else
        {
          S_prime = states_sess(i + 1) - 1;
        }
      }
      
      if (S_prime != initState)
      {
        changeState = true;
      }
      else if (S_prime == initState && changeState)
      {
        returnToInitState = true;
      }
      
      if(debug)
      {
        Rcpp::Rcout <<"i="<< i << ", S=" << S <<", A=" << A<<std::endl;
      }
      
      
      Rcpp::StringVector turns;
      if(S==0)
      {
        turns = S0.getTurnsFromPaths(A);
      } 
      else
      {
        turns = S1.getTurnsFromPaths(A);
      }
      int nbOfTurns = turns.length();
      //Rcpp::Rcout <<"turns="<< turns << std::endl;
      
      Node *prevNode;
      Node *currNode;
      Graph graph;
      if (S == 0)
      {
        graph = S0;
        prevNode = graph.getNode("E");
      }
      else
      {
        graph = S1;
        prevNode = graph.getNode("I");
      }
      double pathProb = 1;
      for (int j = 0; j < nbOfTurns; j++)
      {
        std::string currTurn = Rcpp::as<std::string>(turns(j));
        currNode = graph.getNode(currTurn);
        //currNode->credit = currNode->credit + 1; //Test
        //Rcpp::Rcout <<"currNode="<< currNode->node<<std::endl;
        episodeTurns.push_back(currNode->node);
        episodeTurnStates.push_back(S);
        episodeTurnTimes.push_back(turn_times_session(session_turn_count));
        
        Edge edge = graph.getEdge(prevNode->node, currNode->node);
        double prob_a = edge.probability;
        pathProb = pathProb* prob_a;      
        //Rcpp::Rcout <<"prob_a="<< prob_a << ", pathProb=" <<pathProb <<std::endl;
        
        session_turn_count++;
        prevNode = currNode;
      }
      if(A != 6)
      {
        double logProb = log(pathProb);
        mseMatrix.push_back(logProb);
      }
      else
      {
        mseMatrix.push_back(0);
      }
      
      
      //log_lik=log_lik+ logProb;
      
      //Check if episode ended
      if (returnToInitState || (i==nrow-1))
      {
        //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
        if(debug)
        {
          Rcpp::Rcout <<  "End of episode"<<std::endl;
        }
        changeState = false;
        returnToInitState = false;
        
        avg_score = avg_score + (score_episode - avg_score) / episode;
        //Rcpp::Rcout <<  "score_episode=" << score_episode<<std::endl;
        // for (unsigned int index = 0; index < episodeTurns.size(); ++index)
        // {
        //   Rcpp::Rcout << "index=" <<index << ", episodeTurns[index]= " <<episodeTurns[index]->node <<std::endl;
        // }
        
        
        episodeNb = episodeNb+1;
        double alpha_prime = getAlphaPrime(alpha,episodeNb);
        
        Aca3CreditUpdate(episodeTurns, episodeTurnStates, episodeTurnTimes, alpha_prime, score_episode, &S0, &S1);
        //Aca3CreditUpdate(episodeTurns, episodeTurnStates, episodeTurnTimes, alpha, score_episode, &S0, &S1);
        //Rcpp::Rcout << "Update S0 probabilities"<<std::endl;
        S0.updateEdgeProbs(false);
        //Rcpp::Rcout << "Update S1 probabilities"<<std::endl;
        S1.updateEdgeProbs(false);
        //Rcpp::Rcout <<  "H="<<H<<std::endl;
        //Rcpp::Rcout << "S0 credits: ";
        //S0.printCredits(true);
        //Rcpp::Rcout << std::endl;
        //Rcpp::Rcout << "S1 credits: ";
        //Rcpp::Rcout << std::endl;
        //S1.printCredits(true);
        
        score_episode = 0;
        episode = episode + 1;
        resetVector = true;
        episodeTurns.clear();
        episodeTurnStates.clear();
        episodeTurnTimes.clear();
      }
      S = S_prime;
      //trial=trial+1;
    }
    S0.decayCredits(gamma);
    S1.decayCredits(gamma);
    S0.updateEdgeProbs(false);
    S1.updateEdgeProbs(false);
    
  }
  //Rcpp::Rcout << "S0 credits: " << std::endl;
  S0.printCredits(debug);
  //Rcpp::Rcout << "S1 credits: " << std::endl;
  S1.printCredits(debug);
  return (mseMatrix);
}

arma::mat getAca2ProbMatrix(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug)
{
  arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  std::string model = Rcpp::as<std::string>(modelData.slot("Model"));
  arma::mat turnTimes;
  
  if(model == "Paths")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  }
  else if(model == "Turns")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
  }
  else if(model == "Hybrid1")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel1"));
  }
  else if(model == "Hybrid2")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel2"));
  }
  else if(model == "Hybrid3")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
  }
  else if(model == "Hybrid4")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel4"));
  }
  //Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(testModel.slot("nodeGroups"));
  
  double alpha = Rcpp::as<double>(modelData.slot("alpha"));
  double gamma = Rcpp::as<double>(modelData.slot("gamma1"));
  double reward = Rcpp::as<double>(modelData.slot("gamma2"));
  reward = reward * 10;
  
  //Rcpp::Rcout <<  "allpaths.col(4)="<<allpaths.col(4) <<std::endl;
  int episodeNb = 0; 
  arma::mat mseMatrix;
  //int mseRowIdx = 0;
  
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec allpaths_pathNb = allpaths.col(5);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  
  arma::vec turnTime_method;
  if (sim == 1)
  {
    turnTime_method = turnTimes.col(3);
  }
  else if (model == "Paths")
  {
    turnTime_method = turnTimes.col(3);
  }
  else
  {
    turnTime_method = turnTimes.col(5);
  }
  
  int episode = 1;
  
  Graph S0(testModel, 0);
  Graph S1(testModel, 1);
  
  //Debugger logger;
  //logger.setDebug(debug);
  //Rcpp::Rcout <<"Print S0"<<std::endl;
  //S0.printGraph();
  //Rcpp::Rcout <<"Print S1"<<std::endl;
  //S1.printGraph();
  
  //Rcpp::Rcout <<"rootS1.turn ="<<rootS1->turn<<std::endl;
  //Rcpp::Rcout <<"rootS2.turn ="<<rootS2->turn<<std::endl;
  
  for (unsigned int session = 0; session < uniqSessIdx.n_elem; session++)
  {
    
    int sessId = uniqSessIdx(session);
    //Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
    arma::uvec sessionIdx = arma::find(sessionVec == sessId);
    arma::vec actions_sess = allpath_actions.elem(sessionIdx);
    arma::vec states_sess = allpath_states.elem(sessionIdx);
    arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);
    arma::vec allpaths_pathNb_sess = allpaths_pathNb.elem(sessionIdx);
    
    arma::uvec turnTimes_idx; 
    if (model == "Paths")
    {
      turnTimes_idx = arma::find(sessionVec == sessId); ;
    }
    else
    {
      turnTimes_idx = arma::find(turnTimes.col(4) == sessId); 
    }
    arma::vec turn_times_session = turnTime_method.elem(turnTimes_idx);
    
    arma::uword session_turn_count = 0;
    
    int initState = 0;
    bool changeState = false;
    bool returnToInitState = false;
    int score_episode = 0;
    float avg_score = 0;
    bool resetVector = true;
    int nrow = actions_sess.n_rows;
    int S;
    if(sim == 1)
    {
      S = states_sess(0); 
    }
    else
    {
      S = states_sess(0) - 1; 
    }
    int A = 0;
    std::vector<std::string> episodeTurns;
    std::vector<int> episodeTurnStates;
    std::vector<double> episodeTurnTimes;
    //Rcpp::Rcout <<"nrow="<<nrow<<std::endl;
    for (int i = 0; i < nrow; i++)
    {
      
      if (resetVector)
      {
        initState = S;
        //Rcpp::Rcout <<"initState="<<initState<<std::endl;
        resetVector = false;
      }
      
      int R = rewards_sess(i);
      
      if (R > 0)
      {
        score_episode = score_episode + reward;
      }
      
      
      if (sim == 1)
      {
        A = actions_sess(i);
      }
      else
      {
        A = actions_sess(i) - 1;
      }
      
      int S_prime = 0;
      if(i < (nrow-1))
      {
        if (sim == 1)
        {
          S_prime = states_sess(i + 1);
        }
        else
        {
          S_prime = states_sess(i + 1) - 1;
        }
      }
      
      
      if (S_prime != initState)
      {
        changeState = true;
      }
      else if (S_prime == initState && changeState)
      {
        returnToInitState = true;
      }
      
      
      //Rcpp::Rcout <<"i="<< i << ", S=" << S <<", A=" << A<<std::endl;
      std::ostringstream msg; 
      msg << "i="<< i << ", S=" << S <<", A=" << A <<", pathNb=" << allpaths_pathNb_sess(i);;
      //logger.Print(msg.str()); 
      
      Rcpp::StringVector turns;
      if(S==0)
      {
        turns = S0.getTurnsFromPaths(A);
      } 
      else
      {
        turns = S1.getTurnsFromPaths(A);
      }
      int nbOfTurns = turns.length();
      //Rcpp::Rcout <<"Path="<< A << ", turns=" << turns<<std::endl;
      // msg.str("");
      // msg << "turns=";
      //logger.PrintRcppVec(msg.str(),turns); 
      
      Node *prevNode;
      Node *currNode;
      Graph graph;
      if (S == 0)
      {
        graph = S0;
        prevNode = graph.getNode("E");
      }
      else
      {
        graph = S1;
        prevNode = graph.getNode("I");
      }
      
      double pathProb = 1;
      for (int j = 0; j < nbOfTurns; j++)
      {
        std::string currTurn = Rcpp::as<std::string>(turns(j));
        currNode = graph.getNode(currTurn);
        //Edge edge = graph.getEdge(prevNode->node, currNode->node);
        int turnId = graph.getNodeIndex(currTurn);
        //msg.str("");
        //msg << "currTurn=" << currTurn << ", turnId=" << turnId <<", session_turn_count=" <<session_turn_count;
        //logger.Print(msg.str()); 
        
        double turntime = turn_times_session(session_turn_count);
        episodeTurns.push_back(currNode->node);
        episodeTurnStates.push_back(S);
        episodeTurnTimes.push_back(turntime);
        
        //msg  << ", turntime=" << turntime;
        //logger.Print(msg.str()); 
        
        
        Edge edge = graph.getEdge(prevNode->node, currNode->node);
        double prob_a = edge.probability;
        
        pathProb = pathProb* prob_a; 
        //Rcpp::Rcout <<"src=" <<prevNode->node << ", dest =" <<  currNode->node  << ", prob_a="<< prob_a << ", pathProb=" <<pathProb <<std::endl;     
        prevNode = currNode;
        session_turn_count++;
      }
      
      //Rcpp::Rcout << "pathProb=" << pathProb << std::endl;
      arma::rowvec probRow(13);
      probRow.fill(-1);
      probRow(12) = allpaths_pathNb_sess(i);
      
      for (int path = 0; path < 6; path++)
      {
        //Rcpp::Rcout << "path=" << path << ", state=" << S << std::endl;
        
        for (int box = 0; box < 2; box++)
        {
          Rcpp::StringVector turnVec;
          if (box == 0)
          {
            turnVec = S0.getTurnsFromPaths(path);
            turnVec.push_front("E");
          }
          else
          {
            turnVec = S1.getTurnsFromPaths(path);
            turnVec.push_front("I");
          }
          
          double pathProb = 1;
          
          for (int k = 0; k < (turnVec.length() - 1); k++)
          {
            std::string turn1 = Rcpp::as<std::string>(turnVec[k]);
            std::string turn2 = Rcpp::as<std::string>(turnVec[k + 1]);
            //Rcpp::Rcout << "turn1=" << turn1 << ", turn2=" << turn2 << std::endl;
            
            Edge e;
            if (box == 0)
            {
              e = S0.getEdge(turn1, turn2);
            }
            else
            {
              e = S1.getEdge(turn1, turn2);
            }
            
            //Rcpp::Rcout << "Edge prob=" << e.probability << std::endl;
            pathProb = e.probability * pathProb;
          }
          
          int index = path + (6 * box);
          probRow[index] = pathProb;
          
        }
        
      }
      
      //Rcpp::Rcout << "probRow=" << probRow << std::endl;
      mseMatrix = arma::join_vert(mseMatrix, probRow);
      
      //log_lik=log_lik+ logProb;
      
      //Check if episode ended
      if (returnToInitState || (i==nrow-1))
      {
        if(debug)
        {
          Rcpp::Rcout <<  "End of episode"<<std::endl;
        }
        
        // msg.str("");
        // msg <<"Inside end episode";
        // logger.Print(msg.str()); 
        changeState = false;
        returnToInitState = false;
        
        episodeNb = episodeNb+1;
        double alpha_prime = getAlphaPrime(alpha,episodeNb);
        
        Aca3CreditUpdate(episodeTurns, episodeTurnStates, episodeTurnTimes, alpha_prime, score_episode, &S0, &S1);
        //Aca3CreditUpdate(episodeTurns, episodeTurnStates, episodeTurnTimes, alpha, score_episode, &S0, &S1);
        S0.updateEdgeProbs(false);
        S1.updateEdgeProbs(false);
        score_episode = 0;
        episode = episode + 1;
        resetVector = true;
        episodeTurns.clear();
        episodeTurnStates.clear();
        episodeTurnTimes.clear();
      }
      // msg.str("");
      // msg <<"Here1";
      // logger.Print(msg.str());
      
      S = S_prime;
      // msg.str("");
      // msg <<"Here2";
      // logger.Print(msg.str());
      //trial=trial+1;
    }
    S0.decayCredits(gamma);
    S1.decayCredits(gamma);
    S0.updateEdgeProbs(false);
    S1.updateEdgeProbs(false);
    
    //Rcpp::Rcout <<  "Here3"<<std::endl;
  }
  
  return (mseMatrix);
}


arma::mat getAca2ProbMatrix2(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug)
{
  arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  std::string model = Rcpp::as<std::string>(modelData.slot("Model"));
  arma::mat turnTimes;
  
  if(model == "Paths")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  }
  else if(model == "Turns")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
  }
  else if(model == "Hybrid1")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel1"));
  }
  else if(model == "Hybrid2")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel2"));
  }
  else if(model == "Hybrid3")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
  }
  else if(model == "Hybrid4")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel4"));
  }
  //Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(testModel.slot("nodeGroups"));
  
  double alpha = Rcpp::as<double>(modelData.slot("alpha"));
  double gamma = Rcpp::as<double>(modelData.slot("gamma1"));
  //Rcpp::Rcout <<  "allpaths.col(4)="<<allpaths.col(4) <<std::endl;
  int episodeNb = 0; 
  arma::mat mseMatrix;
  //int mseRowIdx = 0;
  
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec allpaths_pathNb = allpaths.col(5);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  
  arma::vec turnTime_method;
  if (sim == 1)
  {
    turnTime_method = turnTimes.col(3);
  }
  else if (model == "Paths")
  {
    turnTime_method = turnTimes.col(3);
  }
  else
  {
    turnTime_method = turnTimes.col(5);
  }
  
  int episode = 1;
  
  Graph S0(testModel, 0);
  Graph S1(testModel, 1);
  
  //Debugger logger;
  //logger.setDebug(debug);
  //Rcpp::Rcout <<"Print S0"<<std::endl;
  //S0.printGraph();
  //Rcpp::Rcout <<"Print S1"<<std::endl;
  //S1.printGraph();
  
  //Rcpp::Rcout <<"rootS1.turn ="<<rootS1->turn<<std::endl;
  //Rcpp::Rcout <<"rootS2.turn ="<<rootS2->turn<<std::endl;
  
  for (unsigned int session = 0; session < uniqSessIdx.n_elem; session++)
  {
    
    int sessId = uniqSessIdx(session);
    //Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
    arma::uvec sessionIdx = arma::find(sessionVec == sessId);
    arma::vec actions_sess = allpath_actions.elem(sessionIdx);
    arma::vec states_sess = allpath_states.elem(sessionIdx);
    arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);
    arma::vec allpaths_pathNb_sess = allpaths_pathNb.elem(sessionIdx);
    
    arma::uvec turnTimes_idx; 
    if (model == "Paths")
    {
      turnTimes_idx = arma::find(sessionVec == sessId); ;
    }
    else
    {
      turnTimes_idx = arma::find(turnTimes.col(4) == sessId); 
    }
    arma::vec turn_times_session = turnTime_method.elem(turnTimes_idx);
    
    arma::uword session_turn_count = 0;
    
    int initState = 0;
    bool changeState = false;
    bool returnToInitState = false;
    int score_episode = 0;
    float avg_score = 0;
    bool resetVector = true;
    int nrow = actions_sess.n_rows;
    int S;
    if(sim == 1)
    {
      S = states_sess(0); 
    }
    else
    {
      S = states_sess(0) - 1; 
    }
    int A = 0;
    std::vector<std::string> episodeTurns;
    std::vector<int> episodeTurnStates;
    std::vector<double> episodeTurnTimes;
    //Rcpp::Rcout <<"nrow="<<nrow<<std::endl;
    for (int i = 0; i < nrow; i++)
    {
      
      if (resetVector)
      {
        initState = S;
        //Rcpp::Rcout <<"initState="<<initState<<std::endl;
        resetVector = false;
      }
      
      int R = rewards_sess(i);
      
      if (R > 0)
      {
        score_episode = score_episode + 1;
      }
      
      
      if (sim == 1)
      {
        A = actions_sess(i);
      }
      else
      {
        A = actions_sess(i) - 1;
      }
      
      int S_prime = 0;
      if(i < (nrow-1))
      {
        if (sim == 1)
        {
          S_prime = states_sess(i + 1);
        }
        else
        {
          S_prime = states_sess(i + 1) - 1;
        }
      }
      
      
      if (S_prime != initState)
      {
        changeState = true;
      }
      else if (S_prime == initState && changeState)
      {
        returnToInitState = true;
      }
      
      
      //Rcpp::Rcout <<"i="<< i << ", S=" << S <<", A=" << A<<std::endl;
      // std::ostringstream msg; 
      // msg << "i="<< i << ", S=" << S <<", A=" << A <<", pathNb=" << allpaths_pathNb_sess(i);;
      // logger.Print(msg.str()); 
      
      Rcpp::StringVector turns;
      if(S==0)
      {
        turns = S0.getTurnsFromPaths(A);
      } 
      else
      {
        turns = S1.getTurnsFromPaths(A);
      }
      int nbOfTurns = turns.length();
      //Rcpp::Rcout <<"Path="<< A << ", turns=" << turns<<std::endl;
      // msg.str("");
      // msg << "turns=";
      // logger.PrintRcppVec(msg.str(),turns); 
      
      Node *prevNode;
      Node *currNode;
      Graph graph;
      if (S == 0)
      {
        graph = S0;
        prevNode = graph.getNode("E");
      }
      else
      {
        graph = S1;
        prevNode = graph.getNode("I");
      }
      
      double pathProb = 1;
      for (int j = 0; j < nbOfTurns; j++)
      {
        std::string currTurn = Rcpp::as<std::string>(turns(j));
        currNode = graph.getNode(currTurn);
        //Edge edge = graph.getEdge(prevNode->node, currNode->node);
        int turnId = graph.getNodeIndex(currTurn);
        // msg.str("");
        // msg << "currTurn=" << currTurn << ", turnId=" << turnId <<", session_turn_count=" <<session_turn_count;
        // logger.Print(msg.str()); 
        
        double turntime = turn_times_session(session_turn_count);
        episodeTurns.push_back(currNode->node);
        episodeTurnStates.push_back(S);
        episodeTurnTimes.push_back(turntime);
        
        // msg  << ", turntime=" << turntime;
        // logger.Print(msg.str()); 
        
        
        Edge edge = graph.getEdge(prevNode->node, currNode->node);
        double prob_a = edge.probability;
        
        pathProb = pathProb* prob_a; 
        //Rcpp::Rcout <<"src=" <<prevNode->node << ", dest =" <<  currNode->node  << ", prob_a="<< prob_a << ", pathProb=" <<pathProb <<std::endl;     
        prevNode = currNode;
        session_turn_count++;
      }
      
      //Rcpp::Rcout << "pathProb=" << pathProb << std::endl;
      arma::rowvec probRow(13);
      probRow.fill(-1);
      probRow(12) = allpaths_pathNb_sess(i);
      
      for (int path = 0; path < 6; path++)
      {
        //Rcpp::Rcout << "path=" << path << ", state=" << S << std::endl;
        
        for (int box = 0; box < 2; box++)
        {
          Rcpp::StringVector turnVec;
          if (box == 0)
          {
            turnVec = S0.getTurnsFromPaths(path);
            turnVec.push_front("E");
          }
          else
          {
            turnVec = S1.getTurnsFromPaths(path);
            turnVec.push_front("I");
          }
          
          double pathProb = 1;
          
          for (int k = 0; k < (turnVec.length() - 1); k++)
          {
            std::string turn1 = Rcpp::as<std::string>(turnVec[k]);
            std::string turn2 = Rcpp::as<std::string>(turnVec[k + 1]);
            //Rcpp::Rcout << "turn1=" << turn1 << ", turn2=" << turn2 << std::endl;
            
            Edge e;
            if (box == 0)
            {
              e = S0.getEdge(turn1, turn2);
            }
            else
            {
              e = S1.getEdge(turn1, turn2);
            }
            
            //Rcpp::Rcout << "Edge prob=" << e.probability << std::endl;
            pathProb = e.probability * pathProb;
          }
          
          int index = path + (6 * box);
          probRow[index] = pathProb;
          
        }
        
      }
      //Rcpp::Rcout << "probRow=" << probRow << std::endl;
      mseMatrix = arma::join_vert(mseMatrix, probRow);
      
      //log_lik=log_lik+ logProb;
      
      //Check if episode ended
      if (returnToInitState || (i==nrow-1))
      {
        //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
        // msg.str("");
        // msg <<"Inside end episode";
        // logger.Print(msg.str()); 
        changeState = false;
        returnToInitState = false;
        
        episodeNb = episodeNb+1;
        double alpha_prime = getAlphaPrime(alpha,episodeNb);
        
        Aca3CreditUpdate(episodeTurns, episodeTurnStates, episodeTurnTimes, alpha_prime, score_episode, &S0, &S1);
        //Aca3CreditUpdate(episodeTurns, episodeTurnStates, episodeTurnTimes, alpha, score_episode, &S0, &S1);
        S0.updateEdgeProbs(false);
        S1.updateEdgeProbs(false);
        score_episode = 0;
        episode = episode + 1;
        resetVector = true;
        episodeTurns.clear();
        episodeTurnStates.clear();
        episodeTurnTimes.clear();
      }
      // msg.str("");
      // msg <<"Here1";
      // logger.Print(msg.str());
      
      S = S_prime;
      // msg.str("");
      // msg <<"Here2";
      // logger.Print(msg.str());
      //trial=trial+1;
    }
    S0.decayCredits(gamma);
    S1.decayCredits(gamma);
    S0.updateEdgeProbs(false);
    S1.updateEdgeProbs(false);
    
    //Rcpp::Rcout <<  "Here3"<<std::endl;
  }
  
  return (mseMatrix);
}