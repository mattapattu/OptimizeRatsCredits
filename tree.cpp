#include "tree.h"



/* Graph initS0(Rcpp::S4 turnModel)
{
   Rcpp::S4 s0Graph = turnModel.slot("S0");
   Rcpp::List nodes = turnModel.slot("nodes.S0");

   
   
   Graph S0_graph(Rcpp::List &nodes);
   S0_graph.setEdges(Rcpp::List &edgeList);
   printGraph(S0_graph,nodeList.size());
    // for (auto& i: nodeList)
    // {
    //     std::cout << i.node << ' ';
    // }
        

    return (S0_graph);
}

Graph initS1(Rcpp::S4 turnModel)
{

   Rcpp::S4 s1Graph = turnModel.slot("S1");
   Rcpp::List nodes = turnModel.slot("nodes.S1");

   std::vector<Node> nodeList; 
   for (int i=0; i<nodes.size(); i++) 
   {
       //Rcpp::Rcout <<"node="<<Rcpp::as<std::string>(nodes[i])<<std::endl;
       Node n = {Rcpp::as<std::string>(nodes[i]),0};
       nodeList.push_back(n);
   }    
   
   Graph S1_graph(nodeList);

   Rcpp::List edgeList = turnModel.slot("edges.S1");
   std::vector<Edge> edges; 
    for (int i=0; i<edgeList.size(); i++) 
   {
       Rcpp::S4 edge =  edgeList[i];
       SEXP edgeVec = edge.slot("edge");
       Rcpp::StringVector vec(edgeVec);
       SEXP prob = edge.slot("prob");
       Rcpp::NumericVector probVec(prob);

       //Rcpp::Rcout <<"vec="<<vec << ", probVec=" << probVec<<std::endl;

       Node* src = S1_graph.getNode(Rcpp::as<std::string>(vec[0]));
       Node* dest = S1_graph.getNode(Rcpp::as<std::string>(vec[1]));
       Edge e = {src,dest,probVec[0]};
       edges.push_back(e);
   }    

    S1_graph.setEdges(edges);
    printGraph(S1_graph,nodeList.size());
    // for (auto& i: nodeList)
    // {
    //     std::cout << i.node << ' ';
    // }
    return (S1_graph);
}

 */


