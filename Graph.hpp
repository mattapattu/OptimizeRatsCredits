#ifndef __GRAPH__
#define __GRAPH__


#include <vector>
#include <string>

#include <vector>
#include <string>

class Edge {
private:
    std::string edge;
    double prob;

public:
    Edge() : prob(0.0) {}

    Edge(const std::string& e, double p) : edge(e), prob(p) {}

    const std::string& getEdge() const {
        return edge;
    }

    void setEdge(const std::string& e) {
        edge = e;
    }

    double getProb() const {
        return prob;
    }

    void setProb(double p) {
        prob = p;
    }
};

class Graph : public Edge {
private:
    std::string Name;
    double State;
    std::vector<std::string> Path0;
    std::vector<std::string> Path1;
    std::vector<std::string> Path2;
    std::vector<std::string> Path3;
    std::vector<std::string> Path4;
    std::vector<std::string> Path5;

public:
    Graph() : State(0.0) {}

    Graph(const std::string& e, double p, const std::string& name, double state,
          const std::vector<std::string>& path0, const std::vector<std::string>& path1,
          const std::vector<std::string>& path2, const std::vector<std::string>& path3,
          const std::vector<std::string>& path4, const std::vector<std::string>& path5)
        : Edge(e, p), Name(name), State(state), Path0(path0), Path1(path1), Path2(path2),
          Path3(path3), Path4(path4), Path5(path5) {}

    const std::string& getName() const {
        return Name;
    }

    void setName(const std::string& name) {
        Name = name;
    }

    double getState() const {
        return State;
    }

    void setState(double state) {
        State = state;
    }

    const std::vector<std::string>& getPath0() const {
        return Path0;
    }

    void setPath0(const std::vector<std::string>& path0) {
        Path0 = path0;
    }

    // Define getters and setters for other PathN members similarly.
};

class Model : public Graph {
private:
    Graph S0;
    Graph S1;
    std::vector<std::string> nodeGroups;
    std::vector<std::vector<Edge>> edgesS0;
    std::vector<std::vector<Edge>> edgesS1;
    std::vector<std::string> nodesS0;
    std::vector<std::string> nodesS1;
    std::vector<std::vector<std::string>> turnNodesS0;
    std::vector<std::vector<std::string>> turnNodesS1;

public:
    Model() {}

    Model(const Graph& s0, const Graph& s1, const std::vector<std::string>& groups,
          const std::vector<std::vector<Edge>>& edges0, const std::vector<std::vector<Edge>>& edges1,
          const std::vector<std::string>& nS0, const std::vector<std::string>& nS1,
          const std::vector<std::vector<std::string>>& tNodesS0, const std::vector<std::vector<std::string>>& tNodesS1)
        : S0(s0), S1(s1), nodeGroups(groups), edgesS0(edges0), edgesS1(edges1), nodesS0(nS0),
          nodesS1(nS1), turnNodesS0(tNodesS0), turnNodesS1(tNodesS1) {}

    const Graph& getS0() const {
        return S0;
    }

    void setS0(const Graph& s0) {
        S0 = s0;
    }

    const Graph& getS1() const {
        return S1;
    }

    void setS1(const Graph& s1) {
        S1 = s1;
    }

    const std::vector<std::string>& getNodeGroups() const {
        return nodeGroups;
    }

    void setNodeGroups(const std::vector<std::string>& groups) {
        nodeGroups = groups;
    }


    Model convertRModelToCpp(Rcpp::S4 rModel) {
        Rcpp::S4 s0 = rModel.slot("S0");
        Rcpp::S4 s1 = rModel.slot("S1");
        Rcpp::List groups = rModel.slot("nodeGroups");
        Rcpp::List edgesS0 = rModel.slot("edges.S0");
        Rcpp::List edgesS1 = rModel.slot("edges.S1");
        Rcpp::CharacterVector nS0 = rModel.slot("nodes.S0");
        Rcpp::CharacterVector nS1 = rModel.slot("nodes.S1");
        Rcpp::LList tNodesS0 = rModel.slot("turnNodes.S0");
        Rcpp::List tNodesS1 = rModel.slot("turnNodes.S1");

        // Convert the S4 components to C++ objects and populate the Model
        Graph s0Cpp = convertRGraphToCpp(s0);
        Graph s1Cpp = convertRGraphToCpp(s1);
        std::vector<std::vector<Edge>> edgesS0Cpp = convertRListToVectorOfVectors(edgesS0);
        std::vector<std::vector<Edge>> edgesS1Cpp = convertRListToVectorOfVectors(edgesS1);
        std::vector<std::string> nS0Cpp = as<std::vector<std::string>>(nS0);
        std::vector<std::string> nS1Cpp = as<std::vector<std::string>>(nS1);
        std::vector<std::vector<std::string>> tNodesS0Cpp = convertRListToVectorOfVectorsOfString(tNodesS0);
        std::vector<std::vector<std::string>> tNodesS1Cpp = convertRListToVectorOfVectorsOfString(tNodesS1);

    // Create the C++ Model object
    Model cppModel(s0Cpp, s1Cpp, groups, edgesS0Cpp, edgesS1Cpp, nS0Cpp, nS1Cpp, tNodesS0Cpp, tNodesS1Cpp);

    return cppModel;
}

    // Define getters and setters for other members similarly.
};

class AllModels : public Graph {
private:
    Model Paths;
    Model Turns;
    Model Hybrid1;
    Model Hybrid2;
    Model Hybrid3;
    Model Hybrid4;

public:
    AllModels() {}

    AllModels(const Model& paths, const Model& turns, const Model& hybrid1,
              const Model& hybrid2, const Model& hybrid3, const Model& hybrid4)
        : Paths(paths), Turns(turns), Hybrid1(hybrid1), Hybrid2(hybrid2), Hybrid3(hybrid3), Hybrid4(hybrid4) {}

    const Model& getPaths() const {
        return Paths;
    }

    void setPaths(const Model& paths) {
        Paths = paths;
    }

    const Model& getTurns() const {
        return Turns;
    }

    void setTurns(const Model& turns) {
        Turns = turns;
    }

    // Define getters and setters for other members similarly.
};




#endif
