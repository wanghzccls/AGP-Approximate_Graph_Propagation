#ifndef AGP_H
#define AGP_H
#include<iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <queue>
#include <unordered_map>
#include <math.h>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <thread>
#include <string>
#include <unistd.h>
#include <sys/time.h>
#include<Eigen/Dense>

using namespace std;
using namespace Eigen;
typedef unsigned int uint;

namespace propagation{
    class Agp{
    public:
    	uint m,n; //edges and nodes
        int L; // propagation levels
    	double rmax,alpha,t;
        string dataset_name;
        vector<uint>el;
        vector<uint>pl;
        vector<double>rowsum_pos;
        vector<double>rowsum_neg;
        vector<int>random_w;
        vector<double>Du_a;
        vector<double>Du_b;
        //vector<double>Du;
        double agp_operation(string dataset,string agp_alg,uint mm,uint nn,int LL,double rmaxx,double alphaa,double tt,Eigen::Map<Eigen::MatrixXd> &feat);
        void sgc_agp(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed);
        void appnp_agp(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed);
        void gdc_agp(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed);
    };
}


#endif // AGP_H
