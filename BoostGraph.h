#include <iostream>
#include <vector>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <RcppArmadillo.h>
#include <RInside.h>
//#include <mutex>



struct VertexProperties
{
    std::string node;
    double credit;
    int node_id;

};

struct EdgeProperties
{
    double probability;
};

class BoostGraph
{
public:
    typedef boost::adjacency_list<
        boost::vecS,      // OutEdgeList type
        boost::vecS,      // VertexList type
        boost::directedS, // Directed/Undirected graph
        VertexProperties, // Vertex properties
        EdgeProperties    // Edge properties
        >
        Graph;

    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef boost::graph_traits<Graph>::edge_descriptor Edge;
    typedef boost::graph_traits<Graph>::edge_iterator edge_iterator;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;

    BoostGraph() {
        std::lock_guard<std::mutex> lck (mutex_);

    }

    BoostGraph(const Rcpp::S4& turnModel, int state)
    {
        std::lock_guard<std::mutex> lck (mutex_);

        Rcpp::S4 s4graph;
        Rcpp::CharacterVector rcppNodeList;
        Rcpp::List rcppEdgeList;
        

        if (state == 0)
        {
            s4graph = Rcpp::as<Rcpp::S4>(turnModel.slot("S0"));
            rcppNodeList = turnModel.slot("nodes.S0");
            rcppEdgeList = turnModel.slot("edges.S0");
            turnNodes = turnModel.slot("turnNodes.S0");
        }
        else
        {
            s4graph = Rcpp::as<Rcpp::S4>(turnModel.slot("S1"));
            rcppNodeList = turnModel.slot("nodes.S1");
            rcppEdgeList = turnModel.slot("edges.S1");
            turnNodes = turnModel.slot("turnNodes.S1");
        }

        Path0 = Rcpp::as<std::vector<std::string>>(s4graph.slot("Path0"));
        Path1 = Rcpp::as<std::vector<std::string>>(s4graph.slot("Path1"));
        Path2 = Rcpp::as<std::vector<std::string>>(s4graph.slot("Path2"));
        Path3 = Rcpp::as<std::vector<std::string>>(s4graph.slot("Path3"));
        Path4 = Rcpp::as<std::vector<std::string>>(s4graph.slot("Path4"));
        Path5 = Rcpp::as<std::vector<std::string>>(s4graph.slot("Path5"));

        for (int i = 0; i < rcppNodeList.size(); i++)
        {
            addNode(Rcpp::as<std::string>(rcppNodeList[i]), i);
        }
        for (int i = 0; i < rcppEdgeList.size(); i++)
        {

            Rcpp::S4 edge = rcppEdgeList[i];
            SEXP edgeVec = edge.slot("edge");
            Rcpp::StringVector vec(edgeVec);
            SEXP prob = edge.slot("prob");
            Rcpp::NumericVector probVec(prob);
            addEdge(Rcpp::as<std::string>(vec[0]), Rcpp::as<std::string>(vec[1]), std::log(probVec[0]));
        }
    }

//     ~BoostGraph() {
//     // Destroy the mutex
//     mutex_.unlock();
//   }


    void addNode(const std::string &nodeName, int node_id, double initialCredit = 0.0)
    {
        Vertex v = boost::add_vertex(graph);
        graph[v].node = nodeName;
        graph[v].credit = initialCredit;
        graph[v].node_id = node_id;
    }

    void addEdge(const std::string &srcNodeName, const std::string &destNodeName, double probability)
    {
        //std::cout << "Adding edge between src:" <<srcNodeName << " and dest:" << destNodeName <<std::endl; 
        Vertex src = findNode(srcNodeName);
        Vertex dest = findNode(destNodeName);
        if (src != boost::graph_traits<Graph>::null_vertex() &&
            dest != boost::graph_traits<Graph>::null_vertex())
        {
            boost::add_edge(src, dest, EdgeProperties{probability}, graph);
        }
    }

    Vertex findNode(const std::string &nodeName)
    {
        vertex_iterator v, vend;
        for (boost::tie(v, vend) = boost::vertices(graph); v != vend; ++v)
        {
            if (graph[*v].node == nodeName)
            {
                return *v;
            }
        }
        return boost::graph_traits<Graph>::null_vertex();
    }

    Edge findEdge(const Vertex &src , const Vertex &dest)
    {
        Edge edge;  
        if (src != boost::graph_traits<Graph>::null_vertex() && dest != boost::graph_traits<Graph>::null_vertex())
        {
            edge_iterator e, eend;
            for (boost::tie(e, eend) = boost::edges(graph); e != eend; ++e)
            {
                if (boost::source(*e, graph) == src && boost::target(*e, graph) == dest)
                {
                    edge = *e;
                    break;
                }
            }
        }
        return edge;
    }

    double getEdgeProbability(const Edge &edge)
    {
        return graph[edge].probability;
    }

    double getNodeCredits(const Vertex &node)
    {
        return graph[node].credit;
    }
    std::string getNodeName(const Vertex &node)
    {
        return graph[node].node;
    }


    void setNodeCredits(const Vertex &node, double credits)
    {
        graph[node].credit = credits;
    }

    void setEdgeProbability(const Edge &edge, double probability)
    {
        graph[edge].probability = probability;
    }

    Vertex findParent(const Vertex &node)
    {
        // Iterate through the vertices to find the parent
        Vertex parent = boost::graph_traits<Graph>::null_vertex();
        Vertex target = boost::graph_traits<Graph>::null_vertex();
        Graph::out_edge_iterator outIt, outEnd;
        vertex_iterator v, vend;
        for (boost::tie(v, vend) = boost::vertices(graph); v != vend; ++v)
        {
          for (boost::tie(outIt, outEnd) = boost::out_edges(*v, graph); outIt != outEnd; ++outIt)
            {
                target = boost::target(*outIt, graph);
                if(target == node)
                {
                    parent = boost::source(*outIt, graph);
                    break; // Assuming each node has only one parent
                }
                
            }
        }

        
        return parent;
    }

    Vertex findMaxCreditSibling(const Vertex &node)
    {
        double maxCredits = -std::numeric_limits<double>::infinity();
        Vertex maxCreditSibling = boost::graph_traits<Graph>::null_vertex();

        // Get the parent node of the given node
        Vertex parent = findParent(node);

        if (parent != boost::graph_traits<Graph>::null_vertex())
        {
            // Iterate through the adjacent nodes (siblings)
            boost::graph_traits<Graph>::adjacency_iterator adjIt, adjEnd;
            for (boost::tie(adjIt, adjEnd) = boost::adjacent_vertices(parent, graph); adjIt != adjEnd; ++adjIt)
            {
                double credits = getNodeCredits(*adjIt);
                if (credits > maxCredits)
                {
                    maxCredits = credits;
                    maxCreditSibling = *adjIt;
                }
            }
        }

        return maxCreditSibling;
    }

    bool isTerminalVertex(const std::string& vertexName) {
        Vertex v = findNode(vertexName); // Assuming getNode returns the vertex descriptor for the given vertexName

        // Get the out-degree of the vertex
        int outDegree = out_degree(v, graph);

        // If the out-degree is 0, the vertex is a terminal vertex
        return outDegree == 0;
    }


    Vertex getChildWithMaxCredit(const std::string& parentVertexName) {
        Vertex parentVertex = findNode(parentVertexName); // Get the vertex descriptor of the parent node

        // Initialize variables to keep track of the child with max credit
        Vertex maxCreditChild = boost::graph_traits<Graph>::null_vertex();
        double maxCredit = -std::numeric_limits<double>::infinity(); // Initialize with negative infinity

        // Iterate through the outgoing edges of the parent vertex
        Graph::out_edge_iterator ei, ei_end;
        for (tie(ei, ei_end) = out_edges(parentVertex, graph); ei != ei_end; ++ei) {
            Vertex childVertex = target(*ei, graph); // Get the target vertex of the outgoing edge

            // Get the credit of the child node
            double childCredit = getNodeCredits(childVertex); // Implement this function to get the credit of a node

            // If the child has higher credit, update the max credit and maxCreditChild
            if (childCredit > maxCredit) {
                maxCredit = childCredit;
                maxCreditChild = childVertex;
            }
        }

        return maxCreditChild;
    }


    void decayCredits(double gamma)
    {
        for (vertex_iterator it = vertices(graph).first; it != vertices(graph).second; ++it)
        {
            Vertex node = *it;
            graph[node].credit *= gamma;
        }
    }

    std::vector<Vertex> findSiblings(Vertex node)
    {
        std::vector<Vertex> siblings;

        // Get the parent node of the given node
        Vertex parent = findParent(node);
        // Get the outgoing edges from the current node
        Graph::out_edge_iterator it, end;
        for (boost::tie(it, end) = out_edges(parent, graph); it != end; ++it)
        {
            Vertex sibling = boost::target(*it, graph);
            siblings.push_back(sibling);
        }

        return siblings;
    }

    void updateEdgeProbabilitiesSoftmax()
    {
        // Iterate through each node in the graph
        for (vertex_iterator it = vertices(graph).first; it != vertices(graph).second; ++it)
        {
            Vertex node = *it;

            std::vector<Vertex> children;
            Graph::out_edge_iterator ei, ei_end;
            //double sumExponentials = 0.0;
            std::vector<double> values;
            for (tie(ei, ei_end) = out_edges(node, graph); ei != ei_end; ++ei) {
                Vertex childVertex = target(*ei, graph); // Get the target vertex of the outgoing                
                children.push_back(childVertex);
                //sumExponentials += std::exp(graph[childVertex].credit);
                values.push_back(graph[childVertex].credit);
            }

            
            for (Vertex child : children)
            {
                Edge edge = findEdge(node, child);
                // Calculate the softmax probability
                //double softmaxProbability = std::exp(graph[child].credit) / sumExponentials;
                double logsumexp_value = logsumexp(values);


                // Update the probability of the edge
                graph[edge].probability = graph[child].credit-logsumexp_value;

                if (std::isnan(graph[edge].probability)) {
                    std::cout << "Edge src: " << graph[node].node << " dest: "  << graph[child].node << " logprob is nan. Check" << std::endl;
                }

                if (std::isinf(graph[edge].probability)) {
                    std::cout << "Edge src: " << graph[node].node << " dest: "  << graph[child].node << " logprob is infinity. Check" << std::endl;
                }

            }

        }
    }

    double logsumexp(const std::vector<double>& values) {
        if (values.empty()) {
            return -std::numeric_limits<double>::infinity();
        }

        double max_val = values[0];
        for (double val : values) {
            if (val > max_val) {
                max_val = val;
            }
        }

        double sum_exp = 0.0;
        for (double val : values) {
            sum_exp += std::exp(val - max_val);
        }

        return max_val + std::log(sum_exp);
    }


    std::vector<std::string> getTurnsFromPaths(int path)
    {
        //std::lock_guard<std::mutex> lck (mutex_);
        std::vector<std::string> turns;
        // Rcpp::Rcout << "path=" << path <<std::endl;
        if (path == 0)
        {
            turns = Path0;
        }
        else if (path == 1)
        {
            turns = Path1;
        }
        else if (path == 2)
        {
            turns = Path2;
        }
        else if (path == 3)
        {
            turns = Path3;
        }
        else if (path == 4)
        {
            turns = Path4;
        }
        else if (path == 5)
        {
            turns = Path5;
        }

        return (turns);
    }

    int getPathFromTurns(std::vector<std::string> turns)
    {
        //std::lock_guard<std::mutex> lck (mutex_);
        int path = -1;
        if (turns == Path0)
        {
            path = 0;
        }
        else if (turns == Path1)
        {
            path = 1;
        }
        else if (turns == Path2)
        {
            path = 2;
        }
        else if (turns == Path3)
        {
            path = 3;
        }
        else if (turns == Path4)
        {
            path = 4;
        }
        else if (turns == Path5)
        {
            path = 5;
        }

        return (path);
    }

    Rcpp::StringVector getTurnNodes(std::string nodeName)
    {
        //std::lock_guard<std::mutex> lck (mutex_);
        return (turnNodes[nodeName]);
    }

    // void printGraph()
    // {
    //     vertex_iterator v, vend;
    //     edge_iterator e, eend;
    //     for (boost::tie(v, vend) = boost::vertices(graph); v != vend; ++v)
    //     {
    //         std::cout << "Node: " << graph[*v].node << " (Credit: " << graph[*v].credit << ")\n";
    //         for (boost::tie(e, eend) = boost::out_edges(*v, graph); e != eend; ++e)
    //         {
    //             Vertex src = boost::source(*e, graph);
    //             Vertex dest = boost::target(*e, graph);
    //             std::cout << "  Edge to: " << graph[dest].node << " (Probability: " << graph[*e].probability << ")\n";
    //         }
    //     }
    // }

    void printGraph()
    {
        boost::dynamic_properties dp;
        dp.property("node_id", boost::get(&VertexProperties::node_id, graph));
        dp.property("label", boost::get(&VertexProperties::node, graph));
        dp.property("weight", boost::get(&EdgeProperties::probability, graph));
        std::ofstream graph_file_out("./out.gv");
        boost::write_graphviz_dp(graph_file_out, graph, dp);


    }

private:
    Graph graph;
    std::vector<std::string> Path0;
    std::vector<std::string> Path1;
    std::vector<std::string> Path2;
    std::vector<std::string> Path3;
    std::vector<std::string> Path4;
    std::vector<std::string> Path5;
    Rcpp::List turnNodes;
    std::mutex mutex_;



};

// int main()
// {
//     RInside R;
//     std::string cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/Hybrid3.Rdata')";
//     R.parseEvalQ(cmd);                  
//     Rcpp::S4 Hybrid3 = R.parseEval("get('Hybrid3')");
//     BoostGraph graph(Hybrid3, 1);
//     graph.printGraph();

//     return 0;
// }
