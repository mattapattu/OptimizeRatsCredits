#include "TurnsNew.h"
#include "BoostGraph.h"
// #include "utils.h"
using namespace Rcpp;

// using namespace Rcpp;

// Function simulateQLearning
Rcpp::List simulateQLearn(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, Rcpp::S4 turnModel, arma::vec turnStages, bool debug)
{
  // Rcpp::Rcout << "Inside simulateAca2TurnsModels" << std::endl;
  arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  std::string model = Rcpp::as<std::string>(modelData.slot("Model"));
  // Rcpp::Rcout << "model=" << model << std::endl;
  arma::mat turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
  ;
  arma::mat mseMatrix;

  double alpha = Rcpp::as<double>(modelData.slot("alpha"));
  double beta = Rcpp::as<double>(modelData.slot("gamma1"));
  double reward = Rcpp::as<double>(modelData.slot("gamma2"));
  reward = reward * 10;
  double power = Rcpp::as<double>(modelData.slot("lambda"));
  // double reward = 4;
  // double power = 0.5;

  // Rcpp::Rcout << "alpha=" << alpha << ", beta=" << beta << std::endl;

  Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(turnModel.slot("nodeGroups"));

  arma::mat R = arma::zeros(2, 6);
  R(0, 3) = 1;
  R(1, 3) = 1;
  arma::mat generated_PathData;
  arma::mat generated_TurnData;
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  // arma::vec allpath_rewards = allpaths.col(2);
  arma::vec allpath_duration = allpaths.col(3);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  arma::vec pathNb = allpaths.col(5);

  Graph S0(testModel, 0);
  Graph S1(testModel, 1);

  Graph turnsS0(turnModel, 0);
  Graph turnsS1(turnModel, 1);

  std::vector<int> rewardTurnsS0{2, 5, 6};
  std::vector<int> rewardTurnsS1{2, 5, 10};

  int actionNb = 0;
  int actionCounter = 0;
  double averageReward = 0;
  double rewardSum = 0;
  double durationSum = 0;

  // Loop through each session
  for (unsigned int session = 0; session < (uniqSessIdx.n_elem); session++)
  {

    int sessId = uniqSessIdx(session);
    // Rcpp::Rcout << "session=" << session << ", sessId=" << sessId << std::endl;
    arma::uvec sessionIdx = arma::find(sessionVec == (sessId));
    arma::vec actions_sess = allpath_actions.elem(sessionIdx);
    arma::vec states_sess = allpath_states.elem(sessionIdx);

    // int initState = 0;
    int nrow = actions_sess.n_rows;

    int S = states_sess(0) - 1;
    std::vector<std::string> episodeTurns;
    // std::vector<int> episodeTurnStates;
    // std::vector<double> episodeTurnTimes;

    arma::mat generated_PathData_sess(nrow, 7);
    arma::mat generated_TurnsData_sess((nrow * 3), 7);
    generated_PathData_sess.fill(-1);
    generated_TurnsData_sess.fill(-1);
    unsigned int turnIdx = 0; // counter for turn model
    // All episodes in new session
    // Rcpp::Rcout << "nrow=" << nrow << std::endl;
    for (int i = 0; i < nrow; i++)
    {
      actionNb++;
      // Rcpp::Rcout <<"i=" <<i <<", S=" <<S  << std::endl;
      std::vector<Edge> *edges;
      Rcpp::StringVector turnNames;
      Rcpp::StringVector testTurnNames;
      Graph graph;
      Graph turnsGraph;
      Node *prevNode = nullptr;
      Node *currNode = nullptr;
      Node *rootNode = nullptr;
      std::vector<int> turnsVector;
      std::vector<int> rewardVec;

      if (S == 0)
      {
        // Rcpp::Rcout <<"Rat in box E"<< std::endl;
        graph = S0;
        edges = graph.getOutgoingEdges("E");
        turnsGraph = turnsS0;
        rewardVec = rewardTurnsS0;
        rootNode = graph.getNode("E");
      }
      else
      {
        // Rcpp::Rcout <<"Rat in box I"<< std::endl;
        graph = S1;
        edges = graph.getOutgoingEdges("I");
        turnsGraph = turnsS1;
        rewardVec = rewardTurnsS1;
        rootNode = graph.getNode("I");
      }

      // Rcpp::Rcout <<"Selecting actions until reaching reward box."<< std::endl;
      double pathProb = 1;
      double pathDuration = 0;
      // HERE  the nodes based on testModel from one reward box to another are selected
      while (!edges->empty())
      {
        actionCounter = actionCounter + 1;
        Edge edgeSelected = softmax_action_sel(graph, *edges);
        std::string turnSelected = edgeSelected.dest->node;
        // Rcpp::Rcout << "Turn=" << turnSelected <<std::endl;

        double prob_a = edgeSelected.probability;
        pathProb = pathProb * prob_a;
        // Rcpp::Rcout << "Turn=" << turnSelected  << ", prob_a=" << prob_a <<std::endl;

        // Convert the selected edge to TurnModel components
        // Rcpp::Rcout << "Turn=" << turnSelected  << ", turnNb=" << turnNb <<std::endl;
        currNode = edgeSelected.dest;
        // Rcpp::Rcout << "turnSelected =" << turnSelected <<std::endl;

        // Rcpp::Rcout << "turnTime=" << turnTime <<std::endl;
        testTurnNames.push_back(turnSelected);
        // episodeTurns.push_back(currNode->node);
        // episodeTurnStates.push_back(S);
        // Rcpp::Rcout << "model=" << model <<std::endl;
        double testNodeDuration = 0;
        double turnReward = 0;

        if (model == "Turns")
        {
          // std::string turnName = currNode->node;
          // turnNames.push_back(turnSelected);
          // Rcpp::Rcout << "Model is Turns"<<std::endl;
          int componentId = turnsGraph.getNodeIndex(turnSelected);
          turnsVector.push_back(componentId);
          arma::vec durationVec = simulateTurnDuration(turnTimes, allpaths, componentId, (turnIdx + 1), turnStages, nodeGroups, false);
          double turnTime = durationVec(1);
          testNodeDuration = turnTime;
          // Rcpp::Rcout << "Turn duration="<< turnTime<<std::endl;
          pathDuration = pathDuration + turnTime;
          // episodeTurnTimes.push_back(turnTime);

          if (turnsVector == rewardVec)
          {
            turnReward = reward;
          }

          generated_TurnsData_sess(turnIdx, 0) = componentId;
          generated_TurnsData_sess(turnIdx, 1) = S;
          generated_TurnsData_sess(turnIdx, 2) = turnReward;
          generated_TurnsData_sess(turnIdx, 3) = turnTime;
          // Rcpp::Rcout << "Turn=" << turnSelected <<", turnDuration="<< turnTime<<std::endl;
          generated_TurnsData_sess(turnIdx, 4) = sessId;
          generated_TurnsData_sess(turnIdx, 5) = actionNb;
          generated_TurnsData_sess(turnIdx, 6) = durationVec(0);
          turnIdx++;
        }
        else
        {
          // Rcpp::Rcout << "Model is "<< model << " , splitting "<< currNode->node << " into turns."<<std::endl;
          Rcpp::StringVector turnNodes = graph.getTurnNodes(currNode->node);
          Rcpp::IntegerVector turnNodeIds = turnsGraph.getNodeIds(turnNodes);

          // Rcpp::Rcout << "turnDurationMat is" << std::endl << turnDurationMat << std::endl;
          // Rcpp::Rcout << "turnNodes:" <<  turnNodes << std::endl;
          // Rcpp::Rcout << "turnNodeIds:" <<  turnNodeIds << std::endl;

          for (int j = 0; j < turnNodeIds.size(); j++)
          {
            turnsVector.push_back(turnNodeIds[j]);
            arma::vec durationVec = simulateTurnDuration(turnTimes, allpaths, turnNodeIds[j], (turnIdx + 1), turnStages, nodeGroups, false);
            double turnTime = durationVec(1);
            pathDuration = pathDuration + turnTime;
            testNodeDuration = testNodeDuration + turnTime;
            // Rcpp::Rcout << "turnNodeId=" << turnNodeIds[j] <<", turnDuration="<< turnTime<<std::endl;

            if (turnsVector == rewardVec)
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

        double actionDuration = testNodeDuration / (double)1;
        edges = graph.getOutgoingEdges(currNode->node);

        Edge edge;
        std::vector<Edge> *siblings = nullptr;
        if (prevNode == nullptr)
        {
          // Rcpp::Rcout << "prevNode is null. This is the first node of the path."<< model<<std::endl;
          edge = graph.getEdge(rootNode->node, currNode->node);
          siblings = graph.getOutgoingEdges(rootNode->node);
        }
        else
        {
          // Rcpp::Rcout << "prevNode is not null. prevNode="<< prevNode->node<<std::endl;
          edge = graph.getEdge(prevNode->node, currNode->node);
          siblings = graph.getOutgoingEdges(prevNode->node);
        }

        // Determine sibling with maxQVal to check if selected action is greedy
        bool isCurrTurnGreedy = false;
        double maxProb = 0;
        Node *maxSibling = nullptr;
        for (auto it = siblings->begin(); it != siblings->end(); ++it)
        {
          double prob = it->probability;
          if ((prob - maxProb) >= 1e-9)
          {
            maxSibling = it->dest;
            maxProb = prob;
          }
        }

        if (maxSibling->node == currNode->node)
        {
          // Rcpp::Rcout <<"Max Sibling is curr node, action is greedy" << std::endl;
          isCurrTurnGreedy = true;
        }
        else
        {
          // Rcpp::Rcout <<"Action selected is not greedy" << std::endl;
        }

        double qMax = -100000;
        std::vector<Edge> *edges = graph.getOutgoingEdges(currNode->node);

        if (edges->size() > 0) // If curr turn is an intermediate turn in the maze, determine qmax using edges
        {
          // Rcpp::Rcout <<"Number of edges greater than zero for current turn " << currNode->node << std::endl;
          Node *selectedNode = nullptr;
          for (auto it = edges->begin(); it != edges->end(); ++it)
          {

            double destNodeCredit = it->dest->credit;
            if ((destNodeCredit - qMax) >= 1e-9)
            {
              qMax = destNodeCredit;
              selectedNode = it->dest;
            }
          }
          if (selectedNode == nullptr)
          {
            // Rcpp::Rcout <<"No edge is selected because all edges have 0 value " << std::endl;
            Edge selectedEdge = edges->at(0);
            selectedNode = selectedEdge.dest;
          }
          qMax = selectedNode->credit;
          // Rcpp::Rcout <<"Edge with max value is: " << selectedNode->node << std::endl;
        }
        else if (edges->empty()) // If curr turn leads to next box, then select qmax using actions from next box
        {
          // Rcpp::Rcout <<"Final turn of the path" << std::endl;
          if (i != (nrow - 1))
          {
            // Rcpp::Rcout <<"Not the final path of the session" << std::endl;
            int A = graph.getPathFromTurns(testTurnNames);
            int last_turn = generated_TurnsData_sess((turnIdx - 1), 0);
            int S_prime = aca_getNextState(S, A, last_turn);
            Node *newRootNode;
            Graph *newGraph = nullptr;
            if (S_prime == 0)
            {
              // Rcpp::Rcout <<"Next state is box E" << std::endl;
              newGraph = &S0;
              newRootNode = newGraph->getNode("E");
            }
            else
            {
              // Rcpp::Rcout <<"Next state is box I" << std::endl;
              newGraph = &S1;
              newRootNode = newGraph->getNode("I");
            }
            std::vector<Edge> *edges = newGraph->getOutgoingEdges(newRootNode->node);

            Node *selectedNode = nullptr;
            for (auto it = edges->begin(); it != edges->end(); ++it)
            {
              double destNodeCredit = it->dest->credit;
              if ((destNodeCredit - qMax) >= 1e-9)
              {
                qMax = destNodeCredit;
                selectedNode = it->dest;
              }
            }
            if (selectedNode == nullptr)
            {
              // Rcpp::Rcout <<"All actions is box "<< rootNode->node << " have value 0." << std::endl;
              //  If all edges have same qval, select edge[0]
              Edge selectedEdge = edges->at(0);
              selectedNode = selectedEdge.dest;
            }
            qMax = selectedNode->credit;
            // Rcpp::Rcout <<"Max value action in box "<< rootNode->node << " is: " << selectedNode->node << std::endl;
          }
          else
          {
            qMax = 0;
          }
        }

        // rewardSum = rewardSum + turnReward;
        // durationSum = durationSum + actionDuration;

        // double avgRewardEst = rewardSum/durationSum;

        // turnReward=turnReward*4;
        double td_err = turnReward - (averageReward * actionDuration) + qMax - currNode->credit;
        // Rcpp::Rcout <<"currTurn="  << currNode->node <<", turnReward=" << turnReward  << ", turntime=" <<actionDuration <<  ", averageReward=" <<averageReward <<  ", qMax=" <<  qMax << ", td_err=" <<td_err << std::endl;

        double alpha_prime = alpha / (double)std::pow(actionCounter, power);
        currNode->credit = currNode->credit + (alpha_prime * td_err);

        // if(isCurrTurnGreedy)
        //{
        //   rewardSum = rewardSum + turnReward;
        //   durationSum = durationSum + actionDuration;
        //   averageReward = rewardSum/(double) durationSum;
        // }

        double beta_prime = beta / (double)std::pow(actionCounter, power);
        averageReward = averageReward + (beta_prime * td_err);

        if (debug)
        {
          Rcpp::Rcout << "currTurn=" << currNode->node << ", currTurnReward=" << turnReward << ", turntime=" << actionDuration << ", averageReward=" << averageReward << ", qMax=" << qMax << ", qCurrNode=" << currNode->credit << ", td_err=" << td_err << ", pathProb=" << pathProb << ",actionNb=" << actionNb << ", alpha_prime=" << alpha_prime << ", beta_prime=" << beta_prime << std::endl;
        }

        S0.updateEdgeProbs();
        S1.updateEdgeProbs();

        // S0.printCredits(true);
        // S1.printCredits(true);

        // S0.printProbabilities();
        // S1.printProbabilities();

        prevNode = currNode;
      }

      turnsVector.clear();
      // Rcpp::Rcout << "testTurnNames=" << testTurnNames <<std::endl;
      // graph.printPaths();
      int A = graph.getPathFromTurns(testTurnNames);
      // Rcpp::Rcout << "S=" <<S << ", A=" << A <<std::endl;

      // arma::mat durationMat = simulatePathTime(turnTimes, allpaths, actionNb, A, pathStages,nodeGroups);

      // Rcpp::Rcout <<"i="<< i << ", S=" << S <<", A=" << A << ", ses=" << sessId<<std::endl;

      generated_PathData_sess(i, 0) = A;
      generated_PathData_sess(i, 1) = S;
      // Rcpp::Rcout <<"R(S, A)=" <<R(S, A)<< std::endl;
      generated_PathData_sess(i, 2) = R(S, A);
      generated_PathData_sess(i, 3) = pathDuration;
      generated_PathData_sess(i, 4) = sessId;
      generated_PathData_sess(i, 5) = actionNb;

      int last_turn = generated_TurnsData_sess((turnIdx - 1), 0);

      int S_prime = aca_getNextState(S, A, last_turn);

      if (debug)
      {
        arma::rowvec probRow(13);
        probRow.fill(-1);
        probRow(12) = 0;

        for (int path = 0; path < 6; path++)
        {
          // Rcpp::Rcout << "path=" << path << ", state=" << S << std::endl;

          Rcpp::StringVector turnVec;
          if (S == 0)
          {
            turnVec = S0.getTurnsFromPaths(path);
            turnVec.push_front("E");
          }
          else
          {
            turnVec = S1.getTurnsFromPaths(path);
            turnVec.push_front("I");
          }
          // Rcpp::Rcout << "turnVec=" << turnVec << std::endl;
          //  msg.str("");
          //  msg << "turnVec=";
          //  logger.PrintRcppVec(msg.str(),turnVec);
          double pathProb = 1;
          for (int k = 0; k < (turnVec.length() - 1); k++)
          {
            std::string turn1 = Rcpp::as<std::string>(turnVec[k]);
            std::string turn2 = Rcpp::as<std::string>(turnVec[k + 1]);
            // Rcpp::Rcout << "turn1=" << turn1 << ", turn2=" << turn2 << std::endl;

            Edge e;
            if (S == 0)
            {
              e = S0.getEdge(turn1, turn2);
            }
            else
            {
              e = S1.getEdge(turn1, turn2);
            }

            // Rcpp::Rcout << "Edge prob=" << e.probability << std::endl;
            pathProb = e.probability * pathProb;
          }
          int index = path + (6 * S);
          // Rcpp::Rcout << "index=" << index << ", pathProb=" << pathProb << std::endl;
          probRow[index] = pathProb;
        }
        // Rcpp::Rcout << "probRow=" << probRow << std::endl;
        mseMatrix = arma::join_vert(mseMatrix, probRow);
      }

      S = S_prime;
    }

    // if (turnIdx < (nrow * 2) - 1)
    // {
    //   generated_TurnsData_sess.shed_rows((turnIdx), ((nrow * 2) - 1));
    // }
    generated_TurnData = arma::join_cols(generated_TurnData, generated_TurnsData_sess.rows(0, (turnIdx - 1)));
    generated_PathData = arma::join_cols(generated_PathData, generated_PathData_sess);
  }
  // Rcpp::Rcout << "generated_PathData is" << std::endl << generated_PathData.rows(0,10) << std::endl;
  return (Rcpp::List::create(Named("PathData") = generated_PathData, _["TurnData"] = generated_TurnData, _["probMat"] = mseMatrix));
}

std::vector<double> getQLearningLik(Rcpp::S4 ratdata, std::vector<double> params, Rcpp::S4 testModel, int sim, bool debug)
{
  arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  std::string model = Rcpp::as<std::string>(testModel.slot("Name"));
  arma::mat turnTimes;

  if (model == "Paths")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  }
  else if (model == "Turns")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
  }
  else if (model == "Hybrid1")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel1"));
  }
  else if (model == "Hybrid2")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel2"));
  }
  else if (model == "Hybrid3")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
  }
  else if (model == "Hybrid4")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel4"));
  }

  // Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(testModel.slot("nodeGroups"));

  double alpha = params[0];
  double beta = params[1];
  double reward = params[2];
  reward = reward * 10;
  double power = params[3];

  // Rcpp::Rcout <<  "alpha="<<alpha << ", beta=" <<beta  << ", reward=" <<reward << ", power="<< power<<std::endl;

  std::vector<double> mseMatrix;
  // int mseRowIdx = 0;

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

  double averageReward = 0;
  double rewardSum = 0;
  double durationSum = 0;
  int actionCounter = 0;
  int pathCounter = 0;

  // Graph S0(testModel, 0);
  // S0.printGraph();
  // Graph S1(testModel, 1);
  // S1.printGraph();
  BoostGraph S0(testModel, 0);
  BoostGraph S1(testModel, 1);

  // Rcpp::Rcout <<"rootS1.turn ="<<rootS1->turn<<std::endl;
  // Rcpp::Rcout <<"rootS2.turn ="<<rootS2->turn<<std::endl;

  for (unsigned int session = 0; session < uniqSessIdx.n_elem; session++)
  {

    int sessId = uniqSessIdx(session);
    // Rcpp::Rcout <<"sessId="<<sessId<<std::endl;

    arma::uvec sessionIdx = arma::find(sessionVec == sessId);
    arma::vec actions_sess = allpath_actions.elem(sessionIdx);
    arma::vec states_sess = allpath_states.elem(sessionIdx);
    if (sim != 1)
    {
      states_sess = states_sess - 1;
      actions_sess = actions_sess - 1;
    }

    arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);

    arma::uvec turnTimes_idx;
    if (model == "Paths")
    {
      turnTimes_idx = arma::find(sessionVec == sessId);
      ;
    }
    else
    {
      turnTimes_idx = arma::find(turnTimes.col(4) == sessId);
    }
    arma::vec turn_times_session = turnTime_method.elem(turnTimes_idx);
    arma::uword session_turn_count = 0;

    int initState = 0;
    bool changeState = false;
    bool endOfCurrEpisode = false;

    int nrow = actions_sess.n_rows;
    int S = 0;
    int A = 0;
    std::vector<std::string> episodeTurns;
    std::vector<int> episodeTurnStates;
    BoostGraph::Vertex prevNode;
    BoostGraph::Vertex currNode;
    BoostGraph::Vertex rootNode;
    BoostGraph *graph;
    // int prevTurnReward = 0;
    double currTurnReward = 0;
    int i = 0;
    while (i < (nrow))
    {
      initState = states_sess(i);

      S = states_sess(i);
      A = actions_sess(i);
      // int S_prime = states_sess(i + 1);

      if (S == 0)
      {
        graph = &S0;
        rootNode = graph->findNode("E");
      }
      else
      {
        graph = &S1;
        rootNode = graph->findNode("I");
      }
      double R = rewards_sess(i);
      if (R > 0)
      {
        R = reward;
      }
      // Rcpp::Rcout <<"ses=" << sessId << ", i=" << i << ", S=" <<S <<  ", A=" << A << ", R="<< R << std::endl;

      //Rcpp::StringVector turns;
      std::vector<std::string> turns = graph->getTurnsFromPaths(A);
      int nbOfTurns = turns.size();

      // Rcpp::Rcout <<"nbOfTurns=" << nbOfTurns << std::endl;
      double pathProb = 0.0;
      for (int j = 0; j < nbOfTurns; j++)
      {

        actionCounter = actionCounter + 1;
        if (j == (nbOfTurns - 1))
        {
          currTurnReward = R;
        }
        else
        {
          currTurnReward = 0;
        }

        std::string currTurn = turns[j];
        currNode = graph->findNode(currTurn);
        double turntime = turn_times_session(session_turn_count);
        turntime = turntime / (double)1;


        BoostGraph::Edge edge;
        if(j==0)
        {
          edge = graph->findEdge(rootNode, currNode);
        }
        else
        {
          edge = graph->findEdge(prevNode, currNode);
        }

        // Rcpp::Rcout <<"currTurn=" << currTurn << ", turnTime=" << turntime << std::endl;
        double prob_a = graph->getEdgeProbability(edge);
        pathProb = pathProb + prob_a;

        bool isCurrTurnGreedy = false;
        BoostGraph::Vertex maxSibling = graph->findMaxCreditSibling(currNode);
        if (graph->getNodeName(maxSibling) == currTurn)
        {
          // Rcpp::Rcout <<"Max Sibling is curr node, action is greedy" << std::endl;
          isCurrTurnGreedy = true;
        }

        double qMax = -100000;

        // CASE1: IF CURR TURN IS AN INERMEDIATE TURN IN THE MAZE, DETERMINE QMAX USING EDGES
        if (!graph->isTerminalVertex(currTurn))
        {
          BoostGraph::Vertex maxChild = graph->getChildWithMaxCredit(currTurn);
          qMax = graph->getNodeCredits(maxChild);
        }
        else if(i != (nrow - 1)) // CASE2: If curr turn leads to next box, then select qmax using actions from next box
        {
          int S_prime = states_sess(i + 1);
          BoostGraph *newGraph;
          BoostGraph::Vertex newRootNode;
          if (S_prime == 0)
          {
            // Rcpp::Rcout <<"Next state is box E" << std::endl;
            newGraph = &S0;
            newRootNode = newGraph->findNode("E");
          }
          else
          {
            // Rcpp::Rcout <<"Next state is box I" << std::endl;
            newGraph = &S1;
            newRootNode = newGraph->findNode("I");
          }
          std::string rootNode = newGraph->getNodeName(newRootNode);
          BoostGraph::Vertex maxChild = newGraph->getChildWithMaxCredit(rootNode);
          qMax = newGraph->getNodeCredits(maxChild);
        }
        else // CASE3: i is final turn of the session
        {
          qMax = 0;
        }

        // currTurnReward = currTurnReward;
        // currTurnReward = currTurnReward*4;
        double currNode_credit = graph->getNodeCredits(currNode);
        double td_err = currTurnReward - (averageReward * turntime) + qMax - currNode_credit;
        // if(debug)
        // {
        //   Rcpp::Rcout <<"S=" << S << ", A=" << A  <<", currTurn="  << currTurn <<", currTurnReward=" << currTurnReward  << ", turntime=" <<turntime <<  ", averageReward=" <<averageReward << ", qMax=" <<  qMax <<  ", qCurrNode="<< currNode->credit  << ", td_err=" <<td_err << ", pathProb=" << pathProb<< std::endl;
        // }


        currNode_credit = currNode_credit + (alpha * td_err);
        graph->setNodeCredits(currNode, currNode_credit);
            // averageReward = averageReward + (beta*td_err);
            //  if(isCurrTurnGreedy)
            //  {
            //    rewardSum = rewardSum + currTurnReward;
            //    durationSum = durationSum + turntime;
            //    averageReward = rewardSum/(double) durationSum;
            //  }

        //double beta_prime = beta / (double)std::pow(actionCounter, power);
        averageReward = averageReward + (beta * td_err);

        prevNode = currNode;
        // prevTurnReward = currTurnReward;
        S0.updateEdgeProbabilitiesSoftmax();
        S1.updateEdgeProbabilitiesSoftmax();

        // S0.printCredits(true);
        // S1.printCredits(true);

        // S0.printProbabilities();
        // S1.printProbabilities();

        session_turn_count++;
      }

      if (A != 6)
      {
        // double logProb = log(pathProb);
        if (std::isnan(pathProb) || std::isinf(pathProb))
        {
          std::cout << "Non-integer logProb=" << pathProb << ", at Path i=" << i << ", session=" << session << std::endl;
        }
        //Path prob is already log(pathprob)
        mseMatrix.push_back(pathProb);
      }
      else
      {
        mseMatrix.push_back(0);
      }

      pathCounter = pathCounter + 1;

      if (debug)
      {
        Rcpp::Rcout << "PathIdx=" << pathCounter << ", S=" << S << ", A=" << A << std::endl;
        // S0.printCredits(debug);
        // S1.printCredits(debug);
        Rcpp::Rcout << "## Tree Node Probs: " << std::endl;
        // S0.printProbabilities(debug);
        // S1.printProbabilities(debug);
      }

      // S = S_prime;
      i++;
    }
    // S0.printCredits(debug);
    // S1.printCredits(debug);
    //
    // S0.printProbabilities(debug);
    // S1.printProbabilities(debug);
  }
  
  // for (int i = 0; i < mseMatrix.size(); i++) {
  //  std::cout << mseMatrix[i] << " ";
  // }

  
  double result = std::accumulate(mseMatrix.begin(), mseMatrix.end(), 0.0);
  //std::cout << "alpha =" << alpha << ", beta=" << beta << ", reward=" << reward << ", lik=" << result << std::endl;

  return (mseMatrix);
}

arma::mat getQLearningProbMat(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug = false)
{
  arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  std::string model = Rcpp::as<std::string>(modelData.slot("Model"));
  arma::mat turnTimes;

  if (model == "Paths")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  }
  else if (model == "Turns")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
  }
  else if (model == "Hybrid1")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel1"));
  }
  else if (model == "Hybrid2")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel2"));
  }
  else if (model == "Hybrid3")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
  }
  else if (model == "Hybrid4")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel4"));
  }

  double alpha = Rcpp::as<double>(modelData.slot("alpha"));
  double beta = Rcpp::as<double>(modelData.slot("gamma1"));
  double reward = Rcpp::as<double>(modelData.slot("gamma2"));
  reward = reward * 10;
  double power = Rcpp::as<double>(modelData.slot("lambda"));

  arma::mat mseMatrix;

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

  double averageReward = 0;
  double rewardSum = 0;
  double durationSum = 0;
  int actionCounter = 0;

  Graph S0(testModel, 0);
  Graph S1(testModel, 1);

  for (unsigned int session = 0; session < uniqSessIdx.n_elem; session++)
  {

    int sessId = uniqSessIdx(session);
    // Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
    arma::uvec sessionIdx = arma::find(sessionVec == sessId);
    arma::vec actions_sess = allpath_actions.elem(sessionIdx);
    arma::vec states_sess = allpath_states.elem(sessionIdx);
    arma::vec allpaths_pathNb_sess = allpaths_pathNb.elem(sessionIdx);

    if (sim != 1)
    {
      states_sess = states_sess - 1;
      actions_sess = actions_sess - 1;
    }

    arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);

    arma::uvec turnTimes_idx;
    if (model == "Paths")
    {
      turnTimes_idx = arma::find(sessionVec == sessId);
      ;
    }
    else
    {
      turnTimes_idx = arma::find(turnTimes.col(4) == sessId);
    }
    arma::vec turn_times_session = turnTime_method.elem(turnTimes_idx);
    arma::uword session_turn_count = 0;

    int initState = 0;
    bool changeState = false;
    bool endOfCurrEpisode = false;

    int nrow = actions_sess.n_rows;
    int S = 0;
    int A = 0;
    std::vector<std::string> episodeTurns;
    std::vector<int> episodeTurnStates;
    Node *prevNode = nullptr;
    Node *currNode = nullptr;
    Node *rootNode = nullptr;
    Graph *graph = nullptr;
    // int prevTurnReward = 0;
    double currTurnReward = 0;
    int i = 0;
    while (i < (nrow))
    {
      initState = states_sess(i);

      S = states_sess(i);
      A = actions_sess(i);
      // int S_prime = states_sess(i + 1);

      if (S == 0)
      {
        graph = &S0;
        rootNode = graph->getNode("E");
      }
      else
      {
        graph = &S1;
        rootNode = graph->getNode("I");
      }
      double R = rewards_sess(i);
      if (R == 1)
      {
        R = reward;
      }
      // Rcpp::Rcout <<"ses=" << sessId << ", i=" << i << ", S=" <<S <<  ", A=" << A << ", R="<< R << std::endl;

      Rcpp::StringVector turns;
      turns = graph->getTurnsFromPaths(A);
      int nbOfTurns = turns.length();

      // Rcpp::Rcout <<"S=" <<S << ", R="<< R << ", nbOfTurns=" << nbOfTurns << std::endl;

      double pathProb2 = 1;
      for (int j = 0; j < nbOfTurns; j++)
      {
        actionCounter = actionCounter + 1;
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
        double turntime = turn_times_session(session_turn_count);
        turntime = turntime / (double)1;
        // Rcpp::Rcout <<"currTurn="<< currTurn << ", currTurnReward=" <<currTurnReward<<std::endl;

        Edge edge;
        std::vector<Edge> *siblings = nullptr;
        if (j == 0)
        {
          edge = graph->getEdge(rootNode->node, currNode->node);
          siblings = graph->getOutgoingEdges(rootNode->node);
        }
        else
        {
          edge = graph->getEdge(prevNode->node, currNode->node);
          siblings = graph->getOutgoingEdges(prevNode->node);
        }

        pathProb2 = pathProb2 * edge.probability;

        bool isCurrTurnGreedy = false;
        double maxProb = 0;
        Node *maxSibling = nullptr;
        for (auto it = siblings->begin(); it != siblings->end(); ++it)
        {
          // Rcpp::Rcout <<"edge.src=" << it->src->node << ", edge.dest=" << it->dest->node << ",credit=" << it->dest->credit <<std::endl;

          double prob = it->probability;
          if ((prob - maxProb) >= 1e-9)
          {
            maxSibling = it->dest;
            maxProb = prob;
          }
        }
        if (maxSibling->node == currNode->node)
        {
          // Rcpp::Rcout <<"Max Sibling is curr node, action is greedy" << std::endl;
          isCurrTurnGreedy = true;
        }

        double qMax = -100000;
        std::vector<Edge> *edges = graph->getOutgoingEdges(currTurn);

        if (edges->size() > 0) // If curr turn is an intermediate turn in the maze, determine qmax using edges
        {
          // Rcpp::Rcout <<"Number of edges greater than zero for current turn " << currTurn << std::endl;
          Node *selectedNode = nullptr;
          for (auto it = edges->begin(); it != edges->end(); ++it)
          {
            // Rcpp::Rcout <<"edge.src=" << it->src->node << ", edge.dest=" << it->dest->node << ",credit=" << it->dest->credit <<std::endl;

            double destNodeCredit = it->dest->credit;
            if ((destNodeCredit - qMax) >= 1e-9)
            {
              qMax = destNodeCredit;
              selectedNode = it->dest;
            }
          }
          if (selectedNode == nullptr)
          {
            // Rcpp::Rcout <<"No edge is selected because all edges have 0 value " << std::endl;
            Edge selectedEdge = edges->at(0);
            selectedNode = selectedEdge.dest;
          }
          qMax = selectedNode->credit;
          // Rcpp::Rcout <<"Edge with max value is: " << selectedNode->node << std::endl;
        }
        else if (j == (nbOfTurns - 1)) // If curr turn leads to next box, then select qmax using actions from next box
        {
          // Rcpp::Rcout <<"Final turn of the path" << std::endl;
          if (i != (nrow - 1))
          {
            // Rcpp::Rcout <<"Not the final path of the session" << std::endl;
            int S_prime = states_sess(i + 1);
            Node *newRootNode;
            Graph *newGraph = nullptr;
            if (S_prime == 0)
            {
              // Rcpp::Rcout <<"Next state is box E" << std::endl;
              newGraph = &S0;
              newRootNode = newGraph->getNode("E");
            }
            else
            {
              // Rcpp::Rcout <<"Next state is box I" << std::endl;
              newGraph = &S1;
              newRootNode = newGraph->getNode("I");
            }
            std::vector<Edge> *edges = newGraph->getOutgoingEdges(newRootNode->node);

            Node *selectedNode = nullptr;
            for (auto it = edges->begin(); it != edges->end(); ++it)
            {
              double destNodeCredit = it->dest->credit;
              if ((destNodeCredit - qMax) >= 1e-9)
              {
                qMax = destNodeCredit;
                selectedNode = it->dest;
              }
            }
            if (selectedNode == nullptr)
            {
              // Rcpp::Rcout <<"All actions is box "<< rootNode->node << " have value 0." << std::endl;
              //  If all edges have same qval, select edge[0]
              Edge selectedEdge = edges->at(0);
              selectedNode = selectedEdge.dest;
            }
            qMax = selectedNode->credit;
            // Rcpp::Rcout <<"Max value action in box "<< rootNode->node << " is: " << selectedNode->node << std::endl;
          }
          else
          {
            qMax = 0;
          }
        }

        // currTurnReward = currTurnReward*4;
        double td_err = currTurnReward - (averageReward * turntime) + qMax - currNode->credit;
        // Rcpp::Rcout <<"currTurn="  << currTurn <<", currTurnReward=" << currTurnReward  << ", turntime=" <<turntime <<  ", averageReward=" <<averageReward << ", qMax=" <<  qMax << ", td_err=" <<td_err << std::endl;

        double alpha_prime = alpha / (double)std::pow(actionCounter, power);
        currNode->credit = currNode->credit + (alpha_prime * td_err);
        // averageReward = averageReward + (beta*td_err);
        //  if(isCurrTurnGreedy)
        //  {
        //    rewardSum = rewardSum + currTurnReward;
        //    durationSum = durationSum + turntime;
        //    averageReward = rewardSum/(double) durationSum;
        //  }

        double beta_prime = beta / (double)std::pow(actionCounter, power);
        averageReward = averageReward + (beta_prime * td_err);

        // Rcpp::Rcout <<"turntime=" <<turntime <<  " ,currTurnReward=" << currTurnReward <<  ", averageReward=" <<averageReward << ", qMax=" <<  qMax << ", td_err=" <<td_err << std::endl;
        if (debug)
        {
          Rcpp::Rcout << "currTurn=" << currNode->node << ", currTurnReward=" << currTurnReward << ", turntime=" << turntime << ", averageReward=" << averageReward << ", qMax=" << qMax << ", qCurrNode=" << currNode->credit << ", td_err=" << td_err << ", pathProb=" << pathProb2 << ",actionNb=" << actionCounter << ", alpha_prime=" << alpha_prime << ", beta_prime=" << beta_prime << std::endl;
        }

        prevNode = currNode;
        // prevTurnReward = currTurnReward;
        S0.updateEdgeProbs();
        S1.updateEdgeProbs();

        // S0.printCredits(true);
        // S1.printCredits(true);

        // S0.printProbabilities();
        // S1.printProbabilities();

        session_turn_count++;
      }

      arma::rowvec probRow(13);
      probRow.fill(-1);
      probRow(12) = allpaths_pathNb_sess(i);

      for (int path = 0; path < 6; path++)
      {
        // Rcpp::Rcout << "path=" << path << ", state=" << S << std::endl;

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
            // Rcpp::Rcout << "turn1=" << turn1 << ", turn2=" << turn2 << std::endl;

            Edge e;
            if (box == 0)
            {
              e = S0.getEdge(turn1, turn2);
            }
            else
            {
              e = S1.getEdge(turn1, turn2);
            }

            // Rcpp::Rcout << "Edge prob=" << e.probability << std::endl;
            pathProb = e.probability * pathProb;
          }

          int index = path + (6 * box);
          probRow[index] = pathProb;
        }
      }

      // Rcpp::Rcout << "probRow=" << probRow << std::endl;
      mseMatrix = arma::join_vert(mseMatrix, probRow);
      i++;
    }
  }
  // Rcpp::Rcout << "mseMatrix=" << mseMatrix << std::endl;
  return (mseMatrix);
}

arma::mat getQLearningProbMat2(Rcpp::S4 ratdata, Rcpp::S4 modelData, Rcpp::S4 testModel, int sim, bool debug = false)
{
  arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  std::string model = Rcpp::as<std::string>(modelData.slot("Model"));
  arma::mat turnTimes;

  if (model == "Paths")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
  }
  else if (model == "Turns")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
  }
  else if (model == "Hybrid1")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel1"));
  }
  else if (model == "Hybrid2")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel2"));
  }
  else if (model == "Hybrid3")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
  }
  else if (model == "Hybrid4")
  {
    turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel4"));
  }

  double alpha = Rcpp::as<double>(modelData.slot("alpha"));
  double beta = Rcpp::as<double>(modelData.slot("gamma1"));
  double reward = Rcpp::as<double>(modelData.slot("gamma2"));
  reward = reward * 10;
  double power = Rcpp::as<double>(modelData.slot("lambda"));

  arma::mat mseMatrix;

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

  double averageReward = 0;
  double rewardSum = 0;
  double durationSum = 0;
  int actionCounter = 0;

  Graph S0(testModel, 0);
  Graph S1(testModel, 1);

  for (unsigned int session = 0; session < uniqSessIdx.n_elem; session++)
  {

    int sessId = uniqSessIdx(session);
    // Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
    arma::uvec sessionIdx = arma::find(sessionVec == sessId);
    arma::vec actions_sess = allpath_actions.elem(sessionIdx);
    arma::vec states_sess = allpath_states.elem(sessionIdx);
    arma::vec allpaths_pathNb_sess = allpaths_pathNb.elem(sessionIdx);

    if (sim != 1)
    {
      states_sess = states_sess - 1;
      actions_sess = actions_sess - 1;
    }

    arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);

    arma::uvec turnTimes_idx;
    if (model == "Paths")
    {
      turnTimes_idx = arma::find(sessionVec == sessId);
      ;
    }
    else
    {
      turnTimes_idx = arma::find(turnTimes.col(4) == sessId);
    }
    arma::vec turn_times_session = turnTime_method.elem(turnTimes_idx);
    arma::uword session_turn_count = 0;

    int initState = 0;
    bool changeState = false;
    bool endOfCurrEpisode = false;

    int nrow = actions_sess.n_rows;
    int S = 0;
    int A = 0;
    std::vector<std::string> episodeTurns;
    std::vector<int> episodeTurnStates;
    Node *prevNode = nullptr;
    Node *currNode = nullptr;
    Node *rootNode = nullptr;
    Graph *graph = nullptr;
    // int prevTurnReward = 0;
    double currTurnReward = 0;
    int i = 0;
    while (i < (nrow))
    {
      initState = states_sess(i);

      S = states_sess(i);
      A = actions_sess(i);
      // int S_prime = states_sess(i + 1);

      if (S == 0)
      {
        graph = &S0;
        rootNode = graph->getNode("E");
      }
      else
      {
        graph = &S1;
        rootNode = graph->getNode("I");
      }
      double R = rewards_sess(i);
      if (R == 1)
      {
        R = reward;
      }
      // Rcpp::Rcout <<"ses=" << sessId << ", i=" << i << ", S=" <<S <<  ", A=" << A << ", R="<< R << std::endl;

      Rcpp::StringVector turns;
      turns = graph->getTurnsFromPaths(A);
      int nbOfTurns = turns.length();

      // Rcpp::Rcout <<"S=" <<S << ", R="<< R << ", nbOfTurns=" << nbOfTurns << std::endl;

      double pathProb2 = 1;
      for (int j = 0; j < nbOfTurns; j++)
      {
        actionCounter = actionCounter + 1;
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
        double turntime = turn_times_session(session_turn_count);
        turntime = turntime / (double)1;
        // Rcpp::Rcout <<"currTurn="<< currTurn << ", currTurnReward=" <<currTurnReward<<std::endl;

        Edge edge;
        std::vector<Edge> *siblings = nullptr;
        if (j == 0)
        {
          edge = graph->getEdge(rootNode->node, currNode->node);
          siblings = graph->getOutgoingEdges(rootNode->node);
        }
        else
        {
          edge = graph->getEdge(prevNode->node, currNode->node);
          siblings = graph->getOutgoingEdges(prevNode->node);
        }

        pathProb2 = pathProb2 * edge.probability;

        bool isCurrTurnGreedy = false;
        double maxProb = 0;
        Node *maxSibling = nullptr;
        for (auto it = siblings->begin(); it != siblings->end(); ++it)
        {
          // Rcpp::Rcout <<"edge.src=" << it->src->node << ", edge.dest=" << it->dest->node << ",credit=" << it->dest->credit <<std::endl;

          double prob = it->probability;
          if ((prob - maxProb) >= 1e-9)
          {
            maxSibling = it->dest;
            maxProb = prob;
          }
        }
        if (maxSibling->node == currNode->node)
        {
          // Rcpp::Rcout <<"Max Sibling is curr node, action is greedy" << std::endl;
          isCurrTurnGreedy = true;
        }

        double qMax = -100000;
        std::vector<Edge> *edges = graph->getOutgoingEdges(currTurn);

        if (edges->size() > 0) // If curr turn is an intermediate turn in the maze, determine qmax using edges
        {
          // Rcpp::Rcout <<"Number of edges greater than zero for current turn " << currTurn << std::endl;
          Node *selectedNode = nullptr;
          for (auto it = edges->begin(); it != edges->end(); ++it)
          {
            // Rcpp::Rcout <<"edge.src=" << it->src->node << ", edge.dest=" << it->dest->node << ",credit=" << it->dest->credit <<std::endl;

            double destNodeCredit = it->dest->credit;
            if ((destNodeCredit - qMax) >= 1e-9)
            {
              qMax = destNodeCredit;
              selectedNode = it->dest;
            }
          }
          if (selectedNode == nullptr)
          {
            // Rcpp::Rcout <<"No edge is selected because all edges have 0 value " << std::endl;
            Edge selectedEdge = edges->at(0);
            selectedNode = selectedEdge.dest;
          }
          qMax = selectedNode->credit;
          // Rcpp::Rcout <<"Edge with max value is: " << selectedNode->node << std::endl;
        }
        else if (j == (nbOfTurns - 1)) // If curr turn leads to next box, then select qmax using actions from next box
        {
          // Rcpp::Rcout <<"Final turn of the path" << std::endl;
          if (i != (nrow - 1))
          {
            // Rcpp::Rcout <<"Not the final path of the session" << std::endl;
            int S_prime = states_sess(i + 1);
            Node *newRootNode;
            Graph *newGraph = nullptr;
            if (S_prime == 0)
            {
              // Rcpp::Rcout <<"Next state is box E" << std::endl;
              newGraph = &S0;
              newRootNode = newGraph->getNode("E");
            }
            else
            {
              // Rcpp::Rcout <<"Next state is box I" << std::endl;
              newGraph = &S1;
              newRootNode = newGraph->getNode("I");
            }
            std::vector<Edge> *edges = newGraph->getOutgoingEdges(newRootNode->node);

            Node *selectedNode = nullptr;
            for (auto it = edges->begin(); it != edges->end(); ++it)
            {
              double destNodeCredit = it->dest->credit;
              if ((destNodeCredit - qMax) >= 1e-9)
              {
                qMax = destNodeCredit;
                selectedNode = it->dest;
              }
            }
            if (selectedNode == nullptr)
            {
              // Rcpp::Rcout <<"All actions is box "<< rootNode->node << " have value 0." << std::endl;
              //  If all edges have same qval, select edge[0]
              Edge selectedEdge = edges->at(0);
              selectedNode = selectedEdge.dest;
            }
            qMax = selectedNode->credit;
            // Rcpp::Rcout <<"Max value action in box "<< rootNode->node << " is: " << selectedNode->node << std::endl;
          }
          else
          {
            qMax = 0;
          }
        }

        // currTurnReward = currTurnReward*4;
        double td_err = currTurnReward - (averageReward * turntime) + qMax - currNode->credit;
        // Rcpp::Rcout <<"currTurn="  << currTurn <<", currTurnReward=" << currTurnReward  << ", turntime=" <<turntime <<  ", averageReward=" <<averageReward << ", qMax=" <<  qMax << ", td_err=" <<td_err << std::endl;

        double alpha_prime = alpha / (double)std::pow(actionCounter, power);
        currNode->credit = currNode->credit + (alpha_prime * td_err);
        // averageReward = averageReward + (beta*td_err);
        //  if(isCurrTurnGreedy)
        //  {
        //    rewardSum = rewardSum + currTurnReward;
        //    durationSum = durationSum + turntime;
        //    averageReward = rewardSum/(double) durationSum;
        //  }

        double beta_prime = beta / (double)std::pow(actionCounter, power);
        averageReward = averageReward + (beta_prime * td_err);

        // Rcpp::Rcout <<"turntime=" <<turntime <<  " ,currTurnReward=" << currTurnReward <<  ", averageReward=" <<averageReward << ", qMax=" <<  qMax << ", td_err=" <<td_err << std::endl;
        if (debug)
        {
          Rcpp::Rcout << "currTurn=" << currNode->node << ", currTurnReward=" << currTurnReward << ", turntime=" << turntime << ", averageReward=" << averageReward << ", qMax=" << qMax << ", qCurrNode=" << currNode->credit << ", td_err=" << td_err << ", pathProb=" << pathProb2 << ",actionNb=" << actionCounter << ", alpha_prime=" << alpha_prime << ", beta_prime=" << beta_prime << std::endl;
        }

        prevNode = currNode;
        // prevTurnReward = currTurnReward;
        S0.updateEdgeProbs();
        S1.updateEdgeProbs();

        // S0.printCredits(true);
        // S1.printCredits(true);

        // S0.printProbabilities();
        // S1.printProbabilities();

        session_turn_count++;
      }

      arma::rowvec probRow(13);
      probRow.fill(-1);
      probRow(12) = allpaths_pathNb_sess(i);

      for (int st = 0; st < 2; st++)
      {
        for (int path = 0; path < 6; path++)
        {
          // Rcpp::Rcout << "path=" << path << ", state=" << S << std::endl;
          Rcpp::StringVector turnVec;
          if (st == 0)
          {
            turnVec = S0.getTurnsFromPaths(path);
            turnVec.push_front("E");
          }
          else
          {
            turnVec = S1.getTurnsFromPaths(path);
            turnVec.push_front("I");
          }
          // Rcpp::Rcout << "turnVec=" << turnVec << std::endl;
          //  msg.str("");
          //  msg << "turnVec=";
          //  logger.PrintRcppVec(msg.str(),turnVec);
          double pathProb = 1;
          for (int k = 0; k < (turnVec.length() - 1); k++)
          {
            std::string turn1 = Rcpp::as<std::string>(turnVec[k]);
            std::string turn2 = Rcpp::as<std::string>(turnVec[k + 1]);
            // Rcpp::Rcout << "turn1=" << turn1 << ", turn2=" << turn2 << std::endl;

            Edge e;
            if (st == 0)
            {
              e = S0.getEdge(turn1, turn2);
            }
            else
            {
              e = S1.getEdge(turn1, turn2);
            }

            // Rcpp::Rcout << "Edge prob=" << e.probability << std::endl;
            pathProb = e.probability * pathProb;
          }
          int index = path + (6 * st);
          // Rcpp::Rcout << "index=" << index << ", pathProb=" << pathProb << std::endl;
          probRow[index] = pathProb;
        }
      }
      // Rcpp::Rcout << "probRow=" << probRow << std::endl;
      mseMatrix = arma::join_vert(mseMatrix, probRow);
      i++;
    }
  }
  // Rcpp::Rcout << "mseMatrix=" << mseMatrix << std::endl;
  return (mseMatrix);
}
