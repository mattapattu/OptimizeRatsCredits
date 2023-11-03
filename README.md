# OptimizeRatsCredits

- Install Pagmo and dependencies
- Install Rcpp, RInside, RcppArmadillo in R
- Compile and build optimize.cpp using command:
g++ -g -std=gnu++17 -o optimize optimize.cpp utils.cpp TurnsNew.cpp avgRewardQLearning.cpp discountedRwdQlearning.cpp aca2.cpp tree.cpp -I /home/mattapattu/.local/include -L /home/mattapattu/.local/lib -Wl,-R/home/mattapattu/.local/lib -I"/usr/share/R/include" -I"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/Rcpp/include" -I"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RcppArmadillo/include" -I"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RInside/include" -pthread -lpagmo -lboost_serialization -ltbb -L/usr/lib/R/lib -lR -L"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RInside/lib" -lRInside -Wl,-rpath,/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RInside/lib


To see maze graphs:
- Uncomment main in graph.cpp
 g++ -g -std=gnu++17 -o BoostGraph.o BoostGraph.cpp -I /home/mattapattu/.local/include -L /home/mattapattu/.local/lib -Wl,-R/home/mattapattu/.local/lib -I"/usr/share/R/include" -I"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/Rcpp/include" -I"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RcppArmadillo/include" -I"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RInside/include" -pthread -lpagmo -lboost_serialization -ltbb -L/usr/lib/R/lib -lR -L"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RInside/lib" -lRInside -Wl,-rpath,/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RInside/lib

