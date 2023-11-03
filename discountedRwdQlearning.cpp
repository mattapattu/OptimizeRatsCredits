#include "TurnsNew.h"
//#include "utils.h"
//using namespace Rcpp;

//Function simulateTurnTimeFromR = Environment::global_env()["simulateTurnTime"];


//namespace aca3 {
Rcpp::List simulateDiscountedRwdQlearning(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, Rcpp::S4 turnModel, arma::vec turnStages, bool debug)
{
  //Rcpp::Rcout << "Inside simulateAca2TurnsModels" << std::endl;
  arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  std::string model = Rcpp::as<std::string>(modelData.slot("Model"));
  //Rcpp::Rcout << "model=" << model << std::endl;
  arma::mat turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));;
  arma::mat mseMatrix;
  //Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(testModel.slot("nodeGroups"));
  
  double alpha = Rcpp::as<double>(modelData.slot("alpha"));
  double beta = Rcpp::as<double>(modelData.slot("gamma1"));
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
  
  std::vector<int> rewardTurnsS0{2,5,6};
  std::vector<int> rewardTurnsS1{2,5,10};
  
  int actionNb = 0;
  int episode = 1;
  
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
      
      std::vector<Edge> *edges;
      Rcpp::StringVector turnNames;
      Rcpp::StringVector testTurnNames;
      Graph graph;
      Graph turnsGraph;
      Node *prevNode=nullptr;
      Node *currNode=nullptr;
      Node * rootNode=nullptr; 
      std::vector<int> turnsVector;
      std::vector<int>  rewardVec;
      
      if (S == 0)
      {
        graph = S0;
        edges = graph.getOutgoingEdges("E");
        turnsGraph = turnsS0;
        rewardVec = rewardTurnsS0;
        rootNode = graph.getNode("E"); 
        
      }
      else
      {
        graph = S1;
        edges = graph.getOutgoingEdges("I");
        turnsGraph = turnsS1;
        rewardVec = rewardTurnsS1;
        rootNode = graph.getNode("I"); 
        
      }
      
      int S_prime = -1;
      double pathDuration = 0;
      //HERE  the nodes based on testModel from one reward box to another are selected 
      while (!edges->empty())
      {
        Edge edgeSelected = softmax_action_sel(graph, *edges);
        std::string turnSelected = edgeSelected.dest->node;
        //Rcpp::Rcout << "Turn=" << turnSelected <<std::endl;
        
        // if(debug)
        // {
        //   Rcpp::Rcout <<"currNode="<< turnSelected<<std::endl;
        // }
        
        //Convert the selected edge to TurnModel components
        //Rcpp::Rcout << "Turn=" << turnSelected  << ", turnNb=" << turnNb <<std::endl;
        currNode = edgeSelected.dest;
        //Rcpp::Rcout << "turnSelected =" << turnSelected <<std::endl;
        double testNodeDuration = 0;
        double turnReward = 0;
        
        //Rcpp::Rcout << "turnTime=" << turnTime <<std::endl;
        testTurnNames.push_back(turnSelected);
        //Rcpp::Rcout << "model=" << model <<std::endl;
        if(model == "Turns")
        {
          //std::string turnName = currNode->node;
          //turnNames.push_back(turnSelected);
          int componentId = turnsGraph.getNodeIndex(turnSelected);     
          turnsVector.push_back(componentId);
          arma::vec durationVec = simulateTurnDuration(turnTimes, allpaths, componentId, (turnIdx+1), turnStages,nodeGroups,false);
          double turnTime = durationVec(1);
          testNodeDuration = turnTime;
          //Rcpp::Rcout << "Turn duration="<< turnTime<<std::endl; 
          pathDuration = pathDuration + turnTime;
          
          if(turnsVector==rewardVec)
          {
            turnReward = reward;
          }
          
          generated_TurnsData_sess(turnIdx, 0) = componentId;
          generated_TurnsData_sess(turnIdx, 1) = S;
          generated_TurnsData_sess(turnIdx, 2) = turnReward;
          generated_TurnsData_sess(turnIdx, 3) = turnTime;
          //Rcpp::Rcout << "Turn=" << turnSelected <<", turnDuration="<< turnTime<<std::endl;
          generated_TurnsData_sess(turnIdx, 4) = sessId;
          generated_TurnsData_sess(turnIdx, 5) = actionNb;
          generated_TurnsData_sess(turnIdx, 6) = durationVec(0);
          turnIdx++;
          
        }
        else{
          Rcpp::StringVector turnNodes  = graph.getTurnNodes(currNode->node);
          Rcpp::IntegerVector turnNodeIds = turnsGraph.getNodeIds(turnNodes);
          for(int j=0; j<turnNodeIds.size();j++)
          {
            turnsVector.push_back(turnNodeIds[j]);
            arma::vec durationVec = simulateTurnDuration(turnTimes, allpaths, turnNodeIds[j], (turnIdx+1), turnStages,nodeGroups,false);
            double turnTime = durationVec(1);
            pathDuration = pathDuration + turnTime;
            testNodeDuration = testNodeDuration + turnTime;
            //Rcpp::Rcout << "turnNodeId=" << turnNodeIds[j] <<", turnDuration="<< turnTime<<std::endl;
            
            
            if(turnsVector==rewardVec)
            {
              turnReward = reward;
            }
            generated_TurnsData_sess(turnIdx, 0) = turnNodeIds[j];
            generated_TurnsData_sess(turnIdx, 1) = S;
            generated_TurnsData_sess(turnIdx, 2) = turnReward;
            generated_TurnsData_sess(turnIdx, 3) = turnTime;
            generated_TurnsData_sess(turnIdx, 4) = sessId;
            generated_TurnsData_sess(turnIdx, 5) = actionNb;
            generated_TurnsData_sess(turnIdx, 6) = durationVec(0);
            
            turnIdx++;
          }
        }
        
        double actionDuration = testNodeDuration/ (double) 1;
        edges = graph.getOutgoingEdges(currNode->node);
        
        Edge edge;
        std::vector<Edge> *siblings = nullptr;
        if(prevNode==nullptr)
        {
          //Rcpp::Rcout << "prevNode is null. This is the first node of the path."<< model<<std::endl; 
          edge = graph.getEdge(rootNode->node, currNode->node);
          siblings = graph.getOutgoingEdges(rootNode->node);
        }
        else
        {
          //Rcpp::Rcout << "prevNode is not null. prevNode="<< prevNode->node<<std::endl; 
          edge = graph.getEdge(prevNode->node, currNode->node);
          siblings = graph.getOutgoingEdges(prevNode->node);
        }
        
        double qMax = -100000;
        std::vector<Edge> *edges = graph.getOutgoingEdges(currNode->node);
        
        //Rcpp::Rcout <<"Current node: " << currNode->node << ", edges nb = " << edges->size() << std::endl;
        if(edges->size() > 0) //If curr turn is an intermediate turn in the maze, determine qmax using edges
        {
          //Rcpp::Rcout <<"Number of edges greater than zero for current turn " << currNode->node << std::endl;
          Node* selectedNode = nullptr;
          for (auto it = edges->begin(); it != edges->end(); ++it)
          {
            
            double destNodeCredit = it->dest->credit;
            if((destNodeCredit - qMax) >= 1e-9)
            {
              qMax = destNodeCredit;
              selectedNode = it->dest;
            }
            
          }
          if(selectedNode == nullptr)
          {
            //Rcpp::Rcout <<"No edge is selected because all edges have 0 value " << std::endl;
            Edge selectedEdge =  edges->at(0);
            selectedNode = selectedEdge.dest;
          }
          qMax = selectedNode->credit;
          //Rcpp::Rcout <<"Max value at node "<< currNode->node <<" is: " << selectedNode->node << std::endl;
        }
        else if(edges->empty())  //If curr turn leads to next box, then select qmax using actions from next box
        {
          //Rcpp::Rcout <<"Final turn of the path" << std::endl;
          
          int A = graph.getPathFromTurns(testTurnNames);
          
          int last_turn = generated_TurnsData_sess((turnIdx - 1), 0);
          
          S_prime = aca_getNextState(S, A, last_turn);
          
          if (S_prime != initState)
          {
            changeState = true;
          }
          else if (S_prime == initState && changeState)
          {
            returnToInitState = true;
          }
          
          
          if(i != (nrow-1) && (!returnToInitState))
          {
            //Rcpp::Rcout <<"End of a path, but not end of the episode or session" << std::endl;
            
            
            int S_prime = aca_getNextState(S, A, last_turn);
            Node * newRootNode;
            Graph * newGraph=nullptr;
            if (S_prime == 0)
            {
              //Rcpp::Rcout <<"Next state is box E" << std::endl;
              newGraph = &S0;
              newRootNode = newGraph->getNode("E"); 
            }
            else
            {
              //Rcpp::Rcout <<"Next state is box I" << std::endl;
              newGraph = &S1;
              newRootNode = newGraph->getNode("I"); 
            }
            std::vector<Edge> *edges = newGraph->getOutgoingEdges(newRootNode->node);
            
            Node* selectedNode=nullptr;
            for (auto it = edges->begin(); it != edges->end(); ++it)
            {
              double destNodeCredit = it->dest->credit;
              if((destNodeCredit - qMax) >= 1e-9)
              {
                qMax = destNodeCredit;
                selectedNode = it->dest;
              }
              
            }
            if(selectedNode == nullptr)
            {
              //Rcpp::Rcout <<"All actions is box "<< newRootNode->node << " have value 0." << std::endl;
              // If all edges have same qval, select edge[0]
              Edge selectedEdge =  edges->at(0);
              selectedNode = selectedEdge.dest;
            }
            qMax = selectedNode->credit;
            //Rcpp::Rcout <<"Max q-val in box "<< newRootNode->node << " is: " << selectedNode->node << std::endl;
          }else{
            //Rcpp::Rcout <<"End of an episode or end of a session, returnToInitState=" << returnToInitState  << std::endl;
            qMax = 0;
          }
        }
        
        double currTurnReward = exp(-beta*actionDuration)*turnReward;
        double td_err = currTurnReward +  exp(-beta*actionDuration)*qMax -currNode->credit;
        
        // if(debug)
        // {
        //   Rcpp::Rcout <<"episode=" << episode << ", S=" << S << ", currTurn="  << currNode->node <<", currTurnReward=" << currTurnReward  << ", turntime=" <<actionDuration << ", qMax=" <<  qMax <<  ", qCurrNode="<< currNode->credit  << ", td_err=" <<td_err << std::endl;
        // }
        
        currNode->credit = currNode->credit + (alpha * td_err);
        
        S0.updateEdgeProbs();
        S1.updateEdgeProbs();
        
        prevNode = currNode;
        
      }
      turnsVector.clear();
      //Rcpp::Rcout << "testTurnNames=" << testTurnNames <<std::endl;
      //graph.printPaths();
      
      //Rcpp::Rcout << "S=" <<S << ", A=" << A <<std::endl;
      
      int A = graph.getPathFromTurns(testTurnNames);
      
      generated_PathData_sess(i, 0) = A;
      generated_PathData_sess(i, 1) = S;
      //Rcpp::Rcout <<"R(S, A)=" <<R(S, A)<< std::endl;
      generated_PathData_sess(i, 2) = R(S, A);
      generated_PathData_sess(i, 3) = pathDuration;
      generated_PathData_sess(i, 4) = sessId;
      generated_PathData_sess(i, 5) = actionNb;
      
      // int last_turn = generated_TurnsData_sess((turnIdx - 1), 0);
      // 
      // int S_prime = aca_getNextState(S, A, last_turn);
      // 
      // if (S_prime != initState)
      // {
      //   changeState = true;
      // }
      // else if (S_prime == initState && changeState)
      // {
      //   returnToInitState = true;
      // }
      
      // if(debug)
      // {
      //   Rcpp::Rcout <<"episode=" <<episode << ", i=" <<i <<", S=" <<S << ", A=" << A << ", changeState=" << changeState << ", returnToInitState=" << returnToInitState << ", resetVector=" << resetVector << std::endl;
      // }
      
      if (returnToInitState || (i==nrow-1))
      {
        
        episode = episode +1;
        changeState = false;
        returnToInitState = false;
        resetVector = true;
      }
      
      
      if(debug)
      {
        arma::rowvec probRow(13);
        probRow.fill(-1);
        probRow(12) = 0;
        
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
        
      }
      
      S = S_prime;
    }
    
    generated_TurnData = arma::join_cols(generated_TurnData, generated_TurnsData_sess.rows(0, (turnIdx - 1)));
    generated_PathData = arma::join_cols(generated_PathData, generated_PathData_sess);
  }
  return (Rcpp::List::create(Named("PathData") = generated_PathData, _["TurnData"] = generated_TurnData, _["probMat"] = mseMatrix));
}

std::vector<double> getDiscountedRwdQlearningLik(Rcpp::S4 ratdata, std::vector<double> params, Rcpp::S4 testModel, int sim, bool debug)
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
  
  int episodeNb = 0; 
  
  double alpha = params[0];
  double beta = params[1];
  double reward = params[2];
  reward = reward * 10;
  double power = params[3];
    
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
  int pathCounter=0;
  Graph S0(testModel, 0);
  //S0.printGraph();
  Graph S1(testModel, 1);
  //S1.printGraph();
  
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
        R = reward;
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
      // if(debug)
      // {
      //   Rcpp::Rcout <<"i="<< i << ", S=" << S <<", A=" << A<<std::endl;
      // }
      
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
      Graph *graph;
      double currTurnReward = 0;
      if (S == 0)
      {
        graph = &S0;
        prevNode = graph->getNode("E");
      }
      else
      {
        graph = &S1;
        prevNode = graph->getNode("I");
      }
      double pathProb = 1;
      for (int j = 0; j < nbOfTurns; j++)
      {
        if (j == (nbOfTurns - 1))
        {
          currTurnReward = R;
        }
        else
        {
          currTurnReward = 0;
        }
        
        std::string currTurn = Rcpp::as<std::string>(turns(j));
        currNode = graph->getNode(currTurn);
        //currNode->credit = currNode->credit + 1; //Test
        //Rcpp::Rcout <<"currNode="<< currNode->node<<std::endl;
        //episodeTurns.push_back(currNode->node);
        //episodeTurnStates.push_back(S);
        //episodeTurnTimes.push_back(turn_times_session(session_turn_count));
        double turntime = turn_times_session(session_turn_count);
        
        Edge edge = graph->getEdge(prevNode->node, currNode->node);
        double prob_a = edge.probability;
        pathProb = pathProb* prob_a;      
        //Rcpp::Rcout <<"prob_a="<< prob_a << ", pathProb=" <<pathProb <<std::endl;
        
        double qMax = -100000;
        std::vector<Edge> *edges = graph->getOutgoingEdges(currTurn);
        
        if(edges->size() > 0) //If curr turn is an intermediate turn in the maze, determine qmax using edges
        {
          //Rcpp::Rcout <<"Number of edges greater than zero for current turn " << currTurn << std::endl;
          Node* selectedNode = nullptr;
          for (auto it = edges->begin(); it != edges->end(); ++it)
          {
            
            double destNodeCredit = it->dest->credit;
            if((destNodeCredit - qMax) >= 1e-9)
            {
              qMax = destNodeCredit;
              selectedNode = it->dest;
            }
            
          }
          if(selectedNode == nullptr)
          {
            //Rcpp::Rcout <<"No edge is selected because all edges have 0 value " << std::endl;
            Edge selectedEdge =  edges->at(0);
            selectedNode = selectedEdge.dest;
          }
          qMax = selectedNode->credit;
          //Rcpp::Rcout <<"Edge with max value is: " << selectedNode->node << std::endl;
        }
        else if(j == (nbOfTurns - 1))  //If curr turn leads to next box, then select qmax using actions from next box
        {
          //Rcpp::Rcout <<"CurrTurn is final action of the path. Searching for Qmax in next box" << std::endl;
          if(i != (nrow-1) && (!returnToInitState))
          {
            //Rcpp::Rcout <<"Not the final action of the episode or session" << std::endl;
            int S_prime = states_sess(i + 1);
            Node * newRootNode;
            Graph * newGraph=nullptr;
            if (S_prime == 0)
            {
              //Rcpp::Rcout <<"Next state is box E" << std::endl;
              newGraph = &S0;
              newRootNode = newGraph->getNode("E"); 
            }
            else
            {
              //Rcpp::Rcout <<"Next state is box I" << std::endl;
              newGraph = &S1;
              newRootNode = newGraph->getNode("I"); 
            }
            std::vector<Edge> *edges = newGraph->getOutgoingEdges(newRootNode->node);
            
            Node* selectedNode=nullptr;
            for (auto it = edges->begin(); it != edges->end(); ++it)
            {
              double destNodeCredit = it->dest->credit;
              if((destNodeCredit - qMax) >= 1e-9)
              {
                qMax = destNodeCredit;
                selectedNode = it->dest;
              }
              
            }
            if(selectedNode == nullptr)
            {
              //Rcpp::Rcout <<"All actions is box "<< rootNode->node << " have value 0." << std::endl;
              // If all edges have same qval, select edge[0]
              Edge selectedEdge =  edges->at(0);
              selectedNode = selectedEdge.dest;
            }
            qMax = selectedNode->credit;
            //Rcpp::Rcout <<"Max value action in box "<< rootNode->node << " is: " << selectedNode->node << std::endl;
          }else{
            qMax = 0;
          }
        }
        
        currTurnReward = exp(-beta*turntime)*currTurnReward;
        double td_err = currTurnReward +  exp(-beta*turntime)*qMax -currNode->credit;
        if(debug)
        {
          Rcpp::Rcout <<"S=" << S << ", A=" << A  <<", currTurn="  << currTurn <<", currTurnReward=" << currTurnReward  << ", turntime=" <<turntime << ", qMax=" <<  qMax <<  ", qCurrNode="<< currNode->credit  << ", td_err=" <<td_err << ", pathProb=" << pathProb<< std::endl;
        }
        currNode->credit = currNode->credit + (alpha * td_err);
        
        S0.updateEdgeProbs();
        S1.updateEdgeProbs();
        
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
      pathCounter = pathCounter+1;
      
      if(debug)
      {
        Rcpp::Rcout << "PathIdx=" << pathCounter<<", S=" << S << ", A=" << A << std::endl;
        S0.printCredits(debug);
        S1.printCredits(debug);
        Rcpp::Rcout<< "## Tree Node Probs: " << std::endl;
        S0.printProbabilities(debug);
        S1.printProbabilities(debug);
      }
      
      
      //Check if episode ended
      if (returnToInitState || (i==nrow-1))
      {
        //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
        // if(debug)
        // {
        //   Rcpp::Rcout <<  "End of episode"<<std::endl;
        // }
        changeState = false;
        returnToInitState = false;
        
        episode = episode + 1;
        resetVector = true;
      }
      
      S = S_prime;
      //trial=trial+1;
      
    }
    
  }
  return (mseMatrix);
}


arma::mat getDiscountedRwdQlearningProbMat(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug)
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
  
  int episodeNb = 0; 
  
  double alpha = Rcpp::as<double>(modelData.slot("alpha"));
  double beta = Rcpp::as<double>(modelData.slot("gamma1"));
  double reward = Rcpp::as<double>(modelData.slot("gamma2"));
  reward = reward * 10;
  
  //Rcpp::Rcout <<  "allpaths.col(4)="<<allpaths.col(4) <<std::endl;
  
  arma::mat mseMatrix;
  //int mseRowIdx = 0;
  
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  arma::vec allpaths_pathNb = allpaths.col(5);
  
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
        R = reward;
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
      // if(debug)
      // {
      //   Rcpp::Rcout <<"i="<< i << ", S=" << S <<", A=" << A<<std::endl;
      // }
      
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
      Graph *graph;
      double currTurnReward = 0;
      if (S == 0)
      {
        graph = &S0;
        prevNode = graph->getNode("E");
      }
      else
      {
        graph = &S1;
        prevNode = graph->getNode("I");
      }
      double pathProb = 1;
      for (int j = 0; j < nbOfTurns; j++)
      {
        if (j == (nbOfTurns - 1))
        {
          currTurnReward = R;
        }
        else
        {
          currTurnReward = 0;
        }
        
        std::string currTurn = Rcpp::as<std::string>(turns(j));
        currNode = graph->getNode(currTurn);
        //currNode->credit = currNode->credit + 1; //Test
        //
        // if(debug)
        // {
        //   Rcpp::Rcout <<"currNode="<< currNode->node<<std::endl;
        // }
        
        //episodeTurns.push_back(currNode->node);
        //episodeTurnStates.push_back(S);
        //episodeTurnTimes.push_back(turn_times_session(session_turn_count));
        double turntime = turn_times_session(session_turn_count);
        
        Edge edge = graph->getEdge(prevNode->node, currNode->node);
        double prob_a = edge.probability;
        pathProb = pathProb* prob_a;      
        //Rcpp::Rcout <<"prob_a="<< prob_a << ", pathProb=" <<pathProb <<std::endl;
        
        double qMax = -100000;
        std::vector<Edge> *edges = graph->getOutgoingEdges(currTurn);
        
        //Rcpp::Rcout <<"Current node: " << currNode->node << ", edges nb = " << edges->size() << std::endl;
        
        if(edges->size() > 0) //If curr turn is an intermediate turn in the maze, determine qmax using edges
        {
          //Rcpp::Rcout <<"Number of edges greater than zero for current turn " << currTurn << std::endl;
          Node* selectedNode = nullptr;
          for (auto it = edges->begin(); it != edges->end(); ++it)
          {
            
            double destNodeCredit = it->dest->credit;
            if((destNodeCredit - qMax) >= 1e-9)
            {
              qMax = destNodeCredit;
              selectedNode = it->dest;
            }
            
          }
          if(selectedNode == nullptr)
          {
            //Rcpp::Rcout <<"No edge is selected because all edges have 0 value " << std::endl;
            Edge selectedEdge =  edges->at(0);
            selectedNode = selectedEdge.dest;
          }
          qMax = selectedNode->credit;
          //Rcpp::Rcout <<"Max value at node "<< currNode->node <<" is: " << selectedNode->node << std::endl;
        }
        else if(j == (nbOfTurns - 1))  //If curr turn leads to next box, then select qmax using actions from next box
        {
          //Rcpp::Rcout <<"CurrTurn is final action of the path. Searching for Qmax in next box" << std::endl;
          if(i != (nrow-1) && (!returnToInitState))
          {
            //Rcpp::Rcout <<"End of a path, but not end of the episode or session" << std::endl;
            int S_prime = states_sess(i + 1);
            Node * newRootNode;
            Graph * newGraph=nullptr;
            if (S_prime == 0)
            {
              //Rcpp::Rcout <<"Next state is box E" << std::endl;
              newGraph = &S0;
              newRootNode = newGraph->getNode("E"); 
            }
            else
            {
              //Rcpp::Rcout <<"Next state is box I" << std::endl;
              newGraph = &S1;
              newRootNode = newGraph->getNode("I"); 
            }
            std::vector<Edge> *edges = newGraph->getOutgoingEdges(newRootNode->node);
            
            Node* selectedNode=nullptr;
            for (auto it = edges->begin(); it != edges->end(); ++it)
            {
              double destNodeCredit = it->dest->credit;
              if((destNodeCredit - qMax) >= 1e-9)
              {
                qMax = destNodeCredit;
                selectedNode = it->dest;
              }
              
            }
            if(selectedNode == nullptr)
            {
              //Rcpp::Rcout <<"All actions is box "<< newRootNode->node << " have value 0." << std::endl;
              // If all edges have same qval, select edge[0]
              Edge selectedEdge =  edges->at(0);
              selectedNode = selectedEdge.dest;
            }
            qMax = selectedNode->credit;
            //Rcpp::Rcout <<"Max q-val in box "<< newRootNode->node << " is: " << selectedNode->node << std::endl;
          }else{
            //Rcpp::Rcout <<"End of an episode or end of a session, returnToInitState=" << returnToInitState  << std::endl;
            qMax = 0;
          }
        }
        
        currTurnReward = exp(-beta*turntime)*currTurnReward;
        double td_err = currTurnReward +  exp(-beta*turntime)*qMax -currNode->credit;
        currNode->credit = currNode->credit + (alpha * td_err);
        
        S0.updateEdgeProbs();
        S1.updateEdgeProbs();
        
        if(debug)
        {
          Rcpp::Rcout <<"episode=" << episode << ", S=" << S << ", currTurn="  << currNode->node <<", currTurnReward=" << currTurnReward  << ", turntime=" <<turntime << ", qMax=" <<  qMax <<  ", qCurrNode="<< currNode->credit  << ", td_err=" <<td_err << std::endl;
          S0.printCredits(debug);
          S1.printCredits(debug);
          Rcpp::Rcout<< "## Tree Node Probs: " << std::endl;
          S0.printCredits(debug);
          S1.printCredits(debug);
        }
        
        session_turn_count++;
        prevNode = currNode;
      }
      
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
      // if(debug)
      // {
      //   Rcpp::Rcout <<"episode=" <<episode << ", i=" <<i <<", S=" <<S << ", A=" << A << ", changeState=" << changeState << ", returnToInitState=" << returnToInitState << ", resetVector=" << resetVector << std::endl;
      // }
      
      //Check if episode ended
      if (returnToInitState || (i==nrow-1))
      {
        //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
        // if(debug)
        // {
        //   Rcpp::Rcout <<  "End of episode"<<std::endl;
        // }
        episode = episode+1;
        changeState = false;
        returnToInitState = false;
        
        resetVector = true;
      }
      
      S = S_prime;
      //trial=trial+1;
    }
    
  }
  return (mseMatrix);
}


Rcpp::List getDiscountedRwdQlearningProbMat2(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug)
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
  
  int episodeNb = 0; 
  
  double alpha = Rcpp::as<double>(modelData.slot("alpha"));
  double beta = Rcpp::as<double>(modelData.slot("gamma1"));
  double reward = Rcpp::as<double>(modelData.slot("gamma2"));
  reward = reward * 10;
  
  //Rcpp::Rcout <<  "allpaths.col(4)="<<allpaths.col(4) <<std::endl;
  
  arma::mat mseMatrix;
  std::vector<double> likVec;
  //int mseRowIdx = 0;
  
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  arma::vec allpaths_pathNb = allpaths.col(5);
  
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
        R = reward;
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
      // if(debug)
      // {
      //   Rcpp::Rcout <<"i="<< i << ", S=" << S <<", A=" << A<<std::endl;
      // }
      
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
      Graph *graph;
      double currTurnReward = 0;
      if (S == 0)
      {
        graph = &S0;
        prevNode = graph->getNode("E");
      }
      else
      {
        graph = &S1;
        prevNode = graph->getNode("I");
      }
      double pathProb = 1;
      for (int j = 0; j < nbOfTurns; j++)
      {
        if (j == (nbOfTurns - 1))
        {
          currTurnReward = R;
        }
        else
        {
          currTurnReward = 0;
        }
        
        std::string currTurn = Rcpp::as<std::string>(turns(j));
        currNode = graph->getNode(currTurn);
        //currNode->credit = currNode->credit + 1; //Test
        //
        // if(debug)
        // {
        //   Rcpp::Rcout <<"currNode="<< currNode->node<<std::endl;
        // }
        
        //episodeTurns.push_back(currNode->node);
        //episodeTurnStates.push_back(S);
        //episodeTurnTimes.push_back(turn_times_session(session_turn_count));
        double turntime = turn_times_session(session_turn_count);
        
        Edge edge = graph->getEdge(prevNode->node, currNode->node);
        double prob_a = edge.probability;
        pathProb = pathProb* prob_a;      
        //Rcpp::Rcout <<"prob_a="<< prob_a << ", pathProb=" <<pathProb <<std::endl;
        
        double qMax = -100000;
        std::vector<Edge> *edges = graph->getOutgoingEdges(currTurn);
        
        //Rcpp::Rcout <<"Current node: " << currNode->node << ", edges nb = " << edges->size() << std::endl;
        
        if(edges->size() > 0) //If curr turn is an intermediate turn in the maze, determine qmax using edges
        {
          //Rcpp::Rcout <<"Number of edges greater than zero for current turn " << currTurn << std::endl;
          Node* selectedNode = nullptr;
          for (auto it = edges->begin(); it != edges->end(); ++it)
          {
            
            double destNodeCredit = it->dest->credit;
            if((destNodeCredit - qMax) >= 1e-9)
            {
              qMax = destNodeCredit;
              selectedNode = it->dest;
            }
            
          }
          if(selectedNode == nullptr)
          {
            //Rcpp::Rcout <<"No edge is selected because all edges have 0 value " << std::endl;
            Edge selectedEdge =  edges->at(0);
            selectedNode = selectedEdge.dest;
          }
          qMax = selectedNode->credit;
          //Rcpp::Rcout <<"Max value at node "<< currNode->node <<" is: " << selectedNode->node << std::endl;
        }
        else if(j == (nbOfTurns - 1))  //If curr turn leads to next box, then select qmax using actions from next box
        {
          //Rcpp::Rcout <<"CurrTurn is final action of the path. Searching for Qmax in next box" << std::endl;
          if(i != (nrow-1) && (!returnToInitState))
          {
            //Rcpp::Rcout <<"End of a path, but not end of the episode or session" << std::endl;
            int S_prime = states_sess(i + 1);
            Node * newRootNode;
            Graph * newGraph=nullptr;
            if (S_prime == 0)
            {
              //Rcpp::Rcout <<"Next state is box E" << std::endl;
              newGraph = &S0;
              newRootNode = newGraph->getNode("E"); 
            }
            else
            {
              //Rcpp::Rcout <<"Next state is box I" << std::endl;
              newGraph = &S1;
              newRootNode = newGraph->getNode("I"); 
            }
            std::vector<Edge> *edges = newGraph->getOutgoingEdges(newRootNode->node);
            
            Node* selectedNode=nullptr;
            for (auto it = edges->begin(); it != edges->end(); ++it)
            {
              double destNodeCredit = it->dest->credit;
              if((destNodeCredit - qMax) >= 1e-9)
              {
                qMax = destNodeCredit;
                selectedNode = it->dest;
              }
              
            }
            if(selectedNode == nullptr)
            {
              //Rcpp::Rcout <<"All actions is box "<< newRootNode->node << " have value 0." << std::endl;
              // If all edges have same qval, select edge[0]
              Edge selectedEdge =  edges->at(0);
              selectedNode = selectedEdge.dest;
            }
            qMax = selectedNode->credit;
            //Rcpp::Rcout <<"Max q-val in box "<< newRootNode->node << " is: " << selectedNode->node << std::endl;
          }else{
            //Rcpp::Rcout <<"End of an episode or end of a session, returnToInitState=" << returnToInitState  << std::endl;
            qMax = 0;
          }
        }
        
        currTurnReward = exp(-beta*turntime)*currTurnReward;
        double td_err = currTurnReward +  exp(-beta*turntime)*qMax -currNode->credit;
        // if(debug)
        // {
        //   Rcpp::Rcout <<"episode=" << episode << ", S=" << S << ", currTurn="  << currNode->node <<", currTurnReward=" << currTurnReward  << ", turntime=" <<turntime << ", qMax=" <<  qMax <<  ", qCurrNode="<< currNode->credit  << ", td_err=" <<td_err << std::endl;
        // }
        currNode->credit = currNode->credit + (alpha * td_err);
        
        S0.updateEdgeProbs();
        S1.updateEdgeProbs();
        
        session_turn_count++;
        prevNode = currNode;
      }
      
      if(A != 6)
      {
        double logProb = log(pathProb);
        likVec.push_back(logProb);
      }
      else
      {
        likVec.push_back(0);
      }
      
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
      // if(debug)
      // {
      //   Rcpp::Rcout <<"episode=" <<episode << ", i=" <<i <<", S=" <<S << ", A=" << A << ", changeState=" << changeState << ", returnToInitState=" << returnToInitState << ", resetVector=" << resetVector << std::endl;
      // }
      
      //Check if episode ended
      if (returnToInitState || (i==nrow-1))
      {
        //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
        // if(debug)
        // {
        //   Rcpp::Rcout <<  "End of episode"<<std::endl;
        // }
        episode = episode+1;
        changeState = false;
        returnToInitState = false;
        
        resetVector = true;
      }
      
      S = S_prime;
      //trial=trial+1;
    }
    
  }
  return (Rcpp::List::create(Named("probMat") = mseMatrix, _["lik"] = likVec ));
  
}




