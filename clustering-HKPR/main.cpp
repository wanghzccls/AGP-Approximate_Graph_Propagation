#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <vector>
#include "SimStruct.h"
#include "util.h"
//#include "metric.h"
//#include "AGPstruct.h"
//#include "MCstruct.h"
//#include "ClusterHKPR.h"
#include <fstream>
#include <cstring>
#include <unordered_set>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>


typedef unsigned int uint;


void usage() {
    cerr << "./HKPR"<<endl
	 << "-d <data directory>"<<endl
	 << "-f <filelabel>"<<endl
	 << "-algo <algorithm>"<<endl
	 << "[-e <epsilon> (default 0.001)]"<<endl
 	 << "[-t <heat kernel parameter> (default 5)]"<<endl
	 << "[-qn <querynum> (default 10)]"<<endl;
}

uint check_inc(uint i, uint max) {
    if (i == max) {
        usage();
        exit(1);
    }
    return i + 1;
}

bool cmp(const pair<uint, double>& a, const pair<uint, double>& b){
    return a.second > b.second;
}



double calPrecision(vector<uint> algo_top, vector<pair<double,uint> > gt_answers, uint k){
    	if(gt_answers.size()<k){
		return 1;
    	} 
    	uint hitCount = 0;
    	uint real_gtsize=k;
    	for(uint i=k;i<gt_answers.size();i++){
		if(gt_answers[i].first!=gt_answers[k-1].first){
			break;
		}
		else{
			real_gtsize+=1;
    		}
    	}
    	for(uint i=0;i<k;i++){
		for(uint j=0;j<real_gtsize;j++){
			if(algo_top[i]==gt_answers[j].second){
				hitCount+=1;
				break;
			}
		}
    	}
    	return (hitCount/(double)k);
}


double conductance(vector<pair<double,uint> >algo_answers, Graph g){ 
	uint *cutcheck=new uint[g.n]();
	uint numcut=0;
	uint crt_vol=0;
	double crt_Phi=1.0;
	uint nonzero=algo_answers.size();
	for(uint i=0;i<nonzero;i++){
		if(algo_answers[i].first==0){
			break;
		}
		uint crt_node=algo_answers[i].second;
		uint crt_OutSize=g.getOutSize(crt_node);
		if(crt_OutSize==0){
			continue;
		}
		crt_vol+=crt_OutSize;
		cutcheck[crt_node]=1;
		for(uint j=0;j<crt_OutSize;j++){
			uint tmpOutV=g.getOutVert(crt_node,j);
			if(cutcheck[tmpOutV]==1){
				numcut-=1;
			}
			else{
				numcut+=1;
			}
		}
		if(numcut<=0){
			numcut=0;
			continue;
		}
		double min_vol=crt_vol;
		if(crt_vol>(g.m-crt_vol)){
			min_vol=g.m-crt_vol;
		}
		double tmp_Phi=numcut/((double)min_vol);
		if(tmp_Phi<=crt_Phi){
			crt_Phi=tmp_Phi;
		}
	}
	delete[] cutcheck;
	return crt_Phi;
}



double cal_maxError(uint vert, double *gtvalues, vector<pair<double,uint> > algoanswers){
    	double maxErr=-1;
	uint *nnzarr=new uint[vert]();
	double tmp_err;
	for(uint j=0;j<algoanswers.size();j++){
		uint tmpnode=algoanswers[j].second;
		tmp_err=abs(algoanswers[j].first-gtvalues[tmpnode]);
		if(maxErr<tmp_err){
			maxErr=tmp_err;
		}
		nnzarr[tmpnode]=1;
	}
	for(uint j=0;j<vert;j++){
		if(nnzarr[j]==0){
			tmp_err=gtvalues[j];
			if(maxErr<tmp_err){
				maxErr=tmp_err;
			}
		}
	}
	delete[] nnzarr;
	return maxErr;
}


void metric(string filedir, string filelabel, Graph g, string algoname, uint querynum, double eps, double t){
	double avg_max_error=0,avg_pre50=0,avg_conductance=0;
    	uint vert=g.n;

    	vector<uint> query_set;
    	uint query_node;
    	ifstream query_file;
    	query_file.open(filedir+"query/"+filelabel+".query");
	if(!query_file){
		cout<<"ERROR: fail to read query file"<<endl;
	}

    	while(query_file>>query_node){
    		query_set.push_back(query_node);
    	}

    	for (uint i = 0; i < querynum; ++i){
        	uint u = query_set[i];
		stringstream ss_gt,ss_gt_dir;
		ss_gt<<filedir<<"result/powermethod/"<<filelabel<<"/"<<t<<"/"<<u<<"_gt.txt";
		//cout<<ss_gt.str()<<endl;
		ifstream gtin;
		gtin.open(ss_gt.str());
		if(!gtin){
	    		cout<<"ERROR:unable to open groundtruth file."<<endl;
		}
	
		double *gtvalues=new double[vert]();
		vector<pair<double, uint> > gtanswers;

        	uint gtCnt = 0;
        	uint gt_tempNode;
        	double gt_tempSim; 
		while(gtin>>gt_tempNode>>gt_tempSim){
			double gt_tempOut=g.getOutSize(gt_tempNode);
			if((gt_tempSim>0.0)&&(gt_tempOut>0.0)){
            			gtvalues[gt_tempNode]=gt_tempSim/gt_tempOut;
	    			gtanswers.push_back(make_pair(gt_tempSim/gt_tempOut, gt_tempNode));
				gtCnt++;
            		}
		}

        	sort(gtanswers.begin(), gtanswers.end(), greater<pair<double, uint> >());

		uint precision_num=50;
		if(gtCnt<50){
			precision_num=gtCnt;
		}
	
        	stringstream ss_algo;
		ss_algo<<filedir<<"result/"<<algoname<<"/"<<filelabel<<"/"<<t<<"/"<<eps<<"/"<<u<<".txt";
        	//cout<<ss_algo.str()<<endl;
		ifstream algoin(ss_algo.str());
        	vector<uint> algoNodes;
		double *algovalues=new double[vert](); 
		bool *algocheck=new bool[vert]();
		vector<pair<double, uint> > algoanswers;
		uint realCnt = 0;
        	uint algo_tempNode;
        	double algo_tempSim;    
		double algo_tempOutSize;
		while(algoin>>algo_tempNode>>algo_tempSim){
            		algoNodes.push_back(algo_tempNode);
			algo_tempOutSize=g.getOutSize(algo_tempNode);
			if(algo_tempOutSize>0.0){
            			algovalues[algo_tempNode]=algo_tempSim/(double)algo_tempOutSize;
				algoanswers.push_back(make_pair((algo_tempSim/(double)algo_tempOutSize), algo_tempNode));
	  			algocheck[algo_tempNode]=true;
            			realCnt++;
			}
        	}
        	sort(algoanswers.begin(),algoanswers.end(),greater<pair<double, uint> >());
		uint topknum=precision_num;
		if((uint)algoanswers.size()<topknum){
			topknum=(uint)algoanswers.size();
		}

		vector<uint> topk_algo_Nodes;
        	for(uint x = 0; x < topknum; x++){
            		topk_algo_Nodes.push_back(algoanswers[x].second);
        	}
		if(topknum<precision_num){
			uint tmpCnt=topknum;
			for(uint supid=0;supid<vert;supid++){
				if(tmpCnt>=precision_num){
					break;
				}
				else if(algocheck[supid]==false){
					topk_algo_Nodes.push_back(supid);
					algocheck[supid]=true;
					tmpCnt+=1;
				}
			}
		}
		avg_conductance+=conductance(algoanswers,g);
		avg_max_error+=cal_maxError(vert,gtvalues,algoanswers);
		avg_pre50+=calPrecision(topk_algo_Nodes,gtanswers,50);
    		delete[] algovalues;
    		delete[] gtvalues;
    		delete[] algocheck;
    	}
    	avg_max_error/=(double)querynum;
    	avg_pre50/=(double)querynum;
    	avg_conductance/=(double)querynum;

    	cout << "Avg MaxError="<< avg_max_error<<endl;
    	cout << "Avg Precision@50="<<avg_pre50<<endl;
    	cout << "Avg Conductance="<<avg_conductance<<endl;
}



int main(int argc, char **argv){
    	uint i = 1;
    	char *endptr;
    	string filedir="./";
    	string filelabel="youtube";
    	string algo="AGP";
    	long querynum = 10;
    	double eps = 0.001;
    	double t=5;
      
    	while (i < argc) {
        	if (!strcmp(argv[i], "-d")) {
            		i = check_inc(i, argc);
            		filedir = argv[i];
	    		if(filedir[filedir.length()-1]!='/'){
				filedir=filedir+"/";
	    		}
	    		//mkpath(filedir);
        	} 
		else if (!strcmp(argv[i], "-f")) {
            		i = check_inc(i, argc);
            		filelabel = argv[i];
        	} 
		else if (!strcmp(argv[i], "-algo")) {
            		i = check_inc(i, argc);
            		algo = argv[i];
        	} 
		else if (!strcmp(argv[i], "-e")) {
            		i = check_inc(i, argc);
            		eps = strtod(argv[i], &endptr);
            		if ((eps == 0 || eps > 1) && endptr) {
                		cerr << "Invalid eps argument" << endl;
                		exit(1);
            		}
        	}
		else if (!strcmp(argv[i], "-t")) {
            		i = check_inc(i, argc);
            		t = strtod(argv[i], &endptr);
            		if ((t <0) && endptr) {
                		cerr << "Invalid t argument" << endl;
                		exit(1);
            		}
        	}
		else if (!strcmp(argv[i], "-qn")) {
            		i = check_inc(i, argc);
            		querynum = strtod(argv[i], &endptr);
            		if ((querynum < -2) && endptr) {
                		cerr << "Invalid querynum argument" << endl;
                		exit(1);
            		}
        	}
		else {
            		usage();
            		exit(1);
        	}
        	i++;
    	}

	cout<<"========="<<endl;
	cout<<"===Local clustering==="<<endl;
	cout<<"Dataset: "<<filelabel<<endl;
	cout<<"Algorithm: "<<algo<<endl;
	cout<<"eps="<<eps<<endl;
	cout<<"t="<<t<<endl;
	cout<<endl;
    	
	SimStruct *sim=NULL;
    	
	if(algo=="AGP"){
		sim = new AGPstruct(filedir, filelabel, eps, t);
    	}
    	else if(algo=="powermethod"){
		sim = new powermethod(filedir,filelabel,eps,t);
    	}

    	if(querynum > sim->vert){
		querynum=sim->vert;
    	}
	cout<<endl;
    	cout<<"querynum="<<querynum<<endl;
    	string queryname;
	queryname = filedir+"query/" + filelabel + ".query";    
	ifstream query;
	query.open(queryname);
	if(!query){
		cout<<"Generate query file..."<<endl;
		uint* check=new uint[sim->vert]();
		ofstream data_idx(queryname);
       		for(uint i = 0; i < querynum; i++){
			int tmpnode;
            		tmpnode = (sim->R.generateRandom() % sim->vert);
			//cout<<"tmpnode="<<tmpnode<<endl;
			while((check[tmpnode]==1)||((sim->g).getOutSize(tmpnode)==0)){
				//cout<<check[tmpnode]<<" "<<(sim->g).getOutSize(tmpnode)<<endl;
				tmpnode=(sim->R.generateRandom() % sim->vert);
				//cout<<"repeated tmpnode="<<tmpnode<<endl;
			}
			check[tmpnode]=1;
            		data_idx << tmpnode<<"\n";
		}
        	data_idx.close();
		query.open(queryname);
		if(!query){
			cout<<"ERROR:input query file:"<<queryname <<endl;
			return 0;
		}
	}
	cout<<"Input query file from: "<<queryname<<endl;

	for(uint i = 0; i < querynum; i++){
		uint nodeId;
	    	query >> nodeId;
	    	cout<<i<<": "<<nodeId<<endl;
		
	    	clock_t t0=clock();
 	    	sim->query(nodeId);
	    	clock_t t1=clock();
	    	sim->avg_time+=(t1-t0)/(double)CLOCKS_PER_SEC;
	    	cout<<"Query time for node "<<nodeId<<": "<<(t1-t0)/(double)CLOCKS_PER_SEC<<" s"<<endl;
	
	    	stringstream ss_dir,ss;
	    	if(algo=="powermethod"){
			ss_dir<<filedir<<"result/"<<algo<<"/"<<filelabel<<"/"<<t<<"/";
	    	}
		else{
			ss_dir<<filedir<<"result/"<<algo<<"/"<<filelabel<<"/"<<t<<"/"<<eps<<"/";
		}
		mkpath(ss_dir.str());
	    	if(algo=="powermethod"){
			ss<<ss_dir.str()<<nodeId<<"_gt.txt";
	    	}
		else{
			ss<<ss_dir.str()<<nodeId<<".txt";
		}
		cout<<"Write query results in file: "<<ss.str()<<endl;
	    	ofstream fout;
            	fout.open(ss.str());
            	fout.setf(ios::fixed,ios::floatfield);
            	fout.precision(15);
            	if(!fout) cout<<"Fail to open the writed file"<<endl;
   	    	for(uint j = 0; j < sim->finalreserve_count; j++){
    	    		fout<<sim->H[0][j]<<" "<<sim->finalReserve[sim->H[0][j]]<<endl;
	    		
		}
    	    	fout.close();
        }
	//util ut;
	//double peakm=ut.peak_mem()/1024.0/1024.0;
	//double overhead=peakm-sim->graph_size;

	cout<<endl;
	cout<<"query time: "<< sim->avg_time/ (double) querynum <<" s"<<endl;
	//cout<<"overhead memory: "<<overhead<<" GB"<<endl;
	
	if(algo!="powermethod"){
		cout<<endl;
    		cout<<"Evaluate "<<algo<<"'s performance..."<<endl;
		metric(filedir, filelabel,sim->g,algo,querynum,eps,t);
	}
	cout << "====Local clustering by "<<algo<<" on "<<filelabel<<" done!===="<<endl;
	cout<<endl<<endl<<endl;
	query.close();

    	return 0;

}
