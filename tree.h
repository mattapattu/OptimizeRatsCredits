#ifndef TREE_H
#define TREE_H

#include <iostream>
#include <vector>
#include <regex>
#include <RcppArmadillo.h>
#include <R.h>





struct Node
{
    std::string node;
    double credit;
};
struct Edge
{
    Node *src;
    Node *dest;
    double probability;
};

struct Paths
{
    Rcpp::StringVector Path0;
    Rcpp::StringVector Path1;
    Rcpp::StringVector Path2;
    Rcpp::StringVector Path3;
    Rcpp::StringVector Path4;
    Rcpp::StringVector Path5;
};

// A class to represent a graph object
class Graph
{
public:
    // a vector of vectors to represent an adjacency list
    std::vector<Node> nodes;
    std::vector<std::vector<Edge>> adjList;
    Paths mazePaths = Paths();
    Rcpp::List turnNodes;
    

    int getNodeIndex(std::string nodeName)
    {
        int index = -1;
        for(unsigned int i=0;i<nodes.size();i++)
        {

            if (nodes[i].node == nodeName)
            {
                index = i;
                break;
            }
        }
        return (index);
    }

    Node *getNode(std::string nodeName)
    {
        for (auto &node : nodes)
        {
            if (node.node == nodeName)
            {
                return (&node);
            }
        }

        return (nullptr);
    }
    
    

    Graph() {}

    Graph(Rcpp::S4 turnModel, int state)
    {
        Rcpp::S4 graph;
        Rcpp::List rcppNodeList;
        Rcpp::List rcppEdgeList;
        

        if (state == 0)
        {
            graph = Rcpp::as<Rcpp::S4>(turnModel.slot("S0"));
            rcppNodeList = turnModel.slot("nodes.S0");
            rcppEdgeList = turnModel.slot("edges.S0");
            turnNodes = turnModel.slot("turnNodes.S0");

        }
        else
        {
            graph = Rcpp::as<Rcpp::S4>(turnModel.slot("S1"));
            rcppNodeList = turnModel.slot("nodes.S1");
            rcppEdgeList = turnModel.slot("edges.S1");
            turnNodes = turnModel.slot("turnNodes.S1");
        }

        mazePaths.Path0 = Rcpp::as<Rcpp::StringVector>(graph.slot("Path0"));
        mazePaths.Path1 = Rcpp::as<Rcpp::StringVector>(graph.slot("Path1"));
        mazePaths.Path2 = Rcpp::as<Rcpp::StringVector>(graph.slot("Path2"));
        mazePaths.Path3 = Rcpp::as<Rcpp::StringVector>(graph.slot("Path3"));
        mazePaths.Path4 = Rcpp::as<Rcpp::StringVector>(graph.slot("Path4"));
        mazePaths.Path5 = Rcpp::as<Rcpp::StringVector>(graph.slot("Path5"));

        for (int i = 0; i < rcppNodeList.size(); i++)
        {
            Node n = {Rcpp::as<std::string>(rcppNodeList[i]), 0};
            n.credit = 0;   // Initialize all node credits to zero.
            //Rcpp::Rcout <<"i=" << i <<", node="<<n.node<<std::endl;
            nodes.push_back(n);
        }
        int N = nodes.size();
        //Rcpp::Rcout <<"N=" << N<<std::endl;
        adjList.resize(N);
        for (int i = 0; i < rcppEdgeList.size(); i++)
        {
            Rcpp::S4 edge = rcppEdgeList[i];
            SEXP edgeVec = edge.slot("edge");
            Rcpp::StringVector vec(edgeVec);
            SEXP prob = edge.slot("prob");
            Rcpp::NumericVector probVec(prob);

            
            Node *src = getNode(Rcpp::as<std::string>(vec[0]));
            Node *dest = getNode(Rcpp::as<std::string>(vec[1]));
            Edge e = {src, dest, probVec[0]};
            int nodeIndex = getNodeIndex(src->node);
            //Rcpp::Rcout <<"nodeIndex=" << nodeIndex <<", vec="<<vec << ", probVec=" << probVec<<std::endl;
            adjList[nodeIndex].push_back(e);
        }
        
    }

    Edge getEdge(std::string src, std::string dest)
    {
        int nodeIndex = getNodeIndex(src);
        std::vector<Edge> edges = adjList[nodeIndex];
        for (auto &edge : edges)
        {
            if (edge.dest->node == dest)
            {
                return (edge);
            }
        }
        return Edge();
    }

    std::vector<Edge>* getOutgoingEdges(std::string src)
    {
        int nodeIndex = getNodeIndex(src);

        std::vector<Edge>* edges = &adjList[nodeIndex];
        return (edges);
    }

    
    void printGraph()
    {
        int N = nodes.size();
        //Rcpp::Rcout <<"N=" << N<<std::endl;
        for (int i = 0; i < N; i++)
        {
            // print the current vertex number
            std::cout << "i=" << i << ", " << nodes[i].node << " ——> ";

            // print all neighboring vertices of a vertex `i`
            for (auto v : adjList[i])
            {
                std::cout << v.dest->node << " ";
            }
            std::cout << std::endl;
        }
    }
    
    void printProbabilities(bool debug)
    {
      if(debug)
      {
        int N = nodes.size();
        Rcpp::Rcout <<"N=" << N<<std::endl;
        for (int i = 0; i < N; i++)
        {
          
          // print all neighboring vertices of a vertex `i`
          for (auto v : adjList[i])
          {
            Rcpp::Rcout <<"v.src=" << v.src->node << ", v.dest=" << v.dest->node << ", v.probability=" << v.probability <<std::endl;
            std::cout << "i=" << i << ", " << nodes[i].node << " ——> " << v.dest->node << ":" <<v.probability << std::endl;
          }
        }
      }
      
    }

    void decayCredits(double gamma)
    {
        for (auto &node : nodes)
        {
            node.credit = gamma*node.credit;
        }
    }

    void printCredits(bool debug)
    {
        if(debug)
        {
            for (auto &node : nodes)
            {
                Rcpp::Rcout << "node = " << node.node << ", credit = " << node.credit << std::endl;
            }
            //Rcpp::Rcout << std::endl;
        }
        
    }
    

    void printPaths()
    {
      Rcpp::Rcout <<  mazePaths.Path0 << std::endl;
      Rcpp::Rcout <<  mazePaths.Path1 << std::endl;
      Rcpp::Rcout <<  mazePaths.Path2 << std::endl;
      Rcpp::Rcout <<  mazePaths.Path3 << std::endl;
      Rcpp::Rcout <<  mazePaths.Path4 << std::endl;
      Rcpp::Rcout <<  mazePaths.Path5 << std::endl;
    }

    Rcpp::List getNodeCredits()
    {
        Rcpp::StringVector v1;
        Rcpp::NumericVector v2;
        for (auto &node : nodes)
        {
            v1.push_back(node.node);
            v2.push_back(node.credit);
        }

        return Rcpp::List::create(Rcpp::Named("nodes") = v1 , Rcpp::Named("credits") = v2);
    }

    void updateEdgeProbs(bool debug = false)
    {
        for (auto &node : nodes)
        {
            std::vector<Edge>* outgoingEdges = getOutgoingEdges(node.node);
            if (!outgoingEdges->empty())
            {
                std::vector<double> edgeCredits;
                Rcpp::StringVector nodeNames;
                for (auto it = outgoingEdges->begin(); it != outgoingEdges->end(); ++it)
                {
                    edgeCredits.push_back((*it).dest->credit);
                    nodeNames.push_back((*it).dest->node);
                    if(debug)
                    {
                        Rcpp::Rcout << (*it).dest->node<< "=" << (*it).dest->credit << std::endl;

                    }
                }
                arma::vec v(edgeCredits);
                
                double max = v.max();
                //Rcpp::Rcout << "v" << v<< std::endl;
                
                
                arma::vec exp_v= exp(v);
                double exp_v_sum = arma::accu(exp_v);
                
                arma::vec v_new=v-max;
                //Rcpp::Rcout << "v-max=" << v_new<< std::endl; 
                arma::vec exp_v_new= exp(v_new);
                //Rcpp::Rcout << "exp_v_new=" << exp_v_new<< std::endl; 
                
                double exp_v_new_sum = arma::accu(exp_v_new);
                //Rcpp::Rcout << "exp_v_new_sum=" << exp_v_new_sum<< std::endl; 
                
                //arma::vec test1_exp_v= arma::exp(v_new);
                //Rcpp::Rcout << "test1_exp_v=" << test1_exp_v<< std::endl; 
                //arma::vec test1_probVec = test1_exp_v / arma::accu(test1_exp_v);
                //Rcpp::Rcout << "test1_probVec=" << test1_probVec<< std::endl; 
                
                //arma::vec test2_exp_v= arma::exp(v);
                //Rcpp::Rcout << "test2_exp_v=" << test2_exp_v<< std::endl; 
                //double test2_exp_sum = arma::accu(test2_exp_v);
                //arma::vec test2_probVec = test2_exp_v / arma::accu(test2_exp_v);
                //Rcpp::Rcout << "test2_probVec=" << test2_probVec<< std::endl; 
                
                
                //arma::vec probVec = exp_v_new / exp_v_new_sum;
                arma::vec probVec = exp_v / exp_v_sum;
                
                //Rcpp::Rcout << "probVec=" << probVec << std::endl;
                
                //if(arma::accu(test1_probVec-probVec) !=0)
                //{
                  //Rcpp::Rcout<< "Diff between exp_v_new and test1_exp_v="<< arma::accu(test1_probVec-probVec) << std::endl;
                //}
                
                //if(arma::accu(test2_probVec-probVec) !=0)
                //{
                  //Rcpp::Rcout<< "Diff between exp_v_new and test2_exp_v=" << arma::accu(test2_probVec-probVec) << std::endl;
                //}
                
                if(debug)
                {
                    Rcpp::Rcout << "nodeNames=" << nodeNames << std::endl;
                    Rcpp::Rcout << "outgoingEdges=" << outgoingEdges<< std::endl;
                    Rcpp::Rcout << "v=" << v << std::endl;
                    Rcpp::Rcout << "probVec=" << probVec << std::endl;
                }
                int i=0;
                for (auto it = outgoingEdges->begin(); it != outgoingEdges->end(); ++it)
                {
                    (*it).probability = probVec[i];
                    i++;
                }
                
            }
        }
    }

    Rcpp::StringVector getTurnsFromPaths(int path)
    {
        Rcpp::StringVector turns;
        //Rcpp::Rcout << "path=" << path <<std::endl;
        if(path == 0)
        {
            turns = mazePaths.Path0;
        }
        else if(path == 1)
        {
            turns = mazePaths.Path1;
        }
        else if(path == 2)
        {
            turns = mazePaths.Path2;
        }
        else if(path == 3)
        {
            turns = mazePaths.Path3;
        }
        else if(path == 4)
        {
            turns = mazePaths.Path4;
        }
        else if(path == 5)
        {
            turns = mazePaths.Path5;
        }

        return(turns);
    }

    int getPathFromTurns(Rcpp::StringVector turns)
    {
        int path=-1;
        if(Rcpp::setequal(turns,mazePaths.Path0))
        {
            path = 0;
        }
        else if(Rcpp::setequal(turns,mazePaths.Path1))
        {
            path = 1;
        }
        else if(Rcpp::setequal(turns,mazePaths.Path2))
        {
            path = 2;
        }
        else if(Rcpp::setequal(turns,mazePaths.Path3))
        {
            path = 3;
        }
        else if(Rcpp::setequal(turns,mazePaths.Path4))
        {
            path = 4;
        }
        else if(Rcpp::setequal(turns,mazePaths.Path5))
        {
            path = 5;
        }

        return(path);
    }


    Rcpp::StringVector getTurnNodes(std::string nodeName)
    {
        
        return(turnNodes[nodeName]);

    }

    Rcpp::IntegerVector getNodeIds(Rcpp::StringVector nodenames)
    {
        Rcpp::IntegerVector nodeIds;
        //Rcpp::Rcout << "nodenames:" <<  nodenames << std::endl;
        for(int k=0; k<nodenames.size(); k++ )
        {
            std::string node = Rcpp::as<std::string>(nodenames[k]);
            int nodeId = getNodeIndex(node);
            nodeIds.push_back(nodeId);
        }

        return(nodeIds);
    }
    
    
};



#endif