#ifndef SIMSTRUCT_H
#define SIMSTRUCT_H

#include <vector>
#include <algorithm>
#include <queue>
#include <functional>
#include <iostream>
#include <thread>
#include <string>
#include <sstream>
#include <fstream>
#include "Graph.h"
#include "Random.h"
#include "alias.h"
#include "util.h"
#include <unordered_map>
#include <unordered_set>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include <errno.h>
#include <queue>
#include <cmath>
#include <random>
#include <ctime>


typedef unsigned int uint;

class SimStruct{
	public:	
    		Graph g;//Class Graph
    		Random R;//Class Random
    		util ut;//Class util
    		uint vert;//the number of vertice
    		string filelabel;
    		double t;
    		double eps;
    		double avg_time;
    		uint seed;
    		double vm,rss_s,rss_d;
    		double graph_size;
		uint *H[2];
    		double *finalReserve;
    		uint finalreserve_count;
    
   	SimStruct(string name, string file_label, double epsilon, double para_t) {
        	filelabel = file_label;
		ut.process_mem_usage(vm,rss_s);
        	g.inputGraph(name, file_label);
 		ut.process_mem_usage(vm,rss_d);
		graph_size=(rss_d-rss_s)/1024.0/1024.0;
		//cout<<"graph size="<<graph_size<<endl;        
        	R = Random();
        	vert = g.n;
		t=para_t;
        	eps = epsilon;
        	avg_time = 0;
		seed=(uint)time(0);
    	}
    	~SimStruct() {
 
    	}
    
    	virtual void query(uint u){ }      
};


class powermethod:public SimStruct{
   public:
    	uint * U[2];
    	uint *candidate_set[2];
    	uint candidate_count[2];
    	double *residue[2];

    	powermethod(string name, string file_label, double epsilon, double para_t):
		SimStruct(name,file_label,epsilon,para_t) {
		candidate_count[0]=0;
		candidate_count[1]=0;
        	H[0] = new uint[vert];
        	H[1] = new uint[vert];
        	U[0] = new uint[vert];
        	U[1] = new uint[vert];
		candidate_set[0] = new uint[vert];
		candidate_set[1] = new uint[vert];
		residue[0]=new double[vert];
		residue[1]=new double[vert];
		finalReserve = new double[vert];
		finalreserve_count=0;
        	for(uint i = 0; i < vert; i++){
	    		residue[0][i]=0;
	    		residue[1][i]=0;
	    		finalReserve[i]=0;
            		H[0][i] = 0;
            		H[1][i] = 0;
            		U[0][i] = 0;
            		U[1][i] = 0;
	    		candidate_set[0][i]=0;
	    		candidate_set[1][i]=0;
		}
    	}
    	~powermethod() {
        	delete[] H[0];
        	delete[] H[1];
        	delete[] U[0];
        	delete[] U[1];
        	delete[] residue[0];
		delete[] residue[1];
		delete[] finalReserve;
    		delete[] candidate_set[0];
		delete[] candidate_set[1];
    	}      
    

    	void query(uint u){
		for(uint j = 0; j < finalreserve_count; j++){
                	finalReserve[H[0][j]] = 0;
                	H[1][H[0][j]] = 0;
        	}
		finalreserve_count=0;
        	uint tempLevel = 0;
	
		double w_i=1.0*exp((-1)*t);
		double Y_i=1;

        	residue[0][u] = 1;
        	candidate_set[0][0]=u;
		candidate_count[0]=1;
	
		uint L=50;
        	//cout<<"L="<<L<<endl;
		//bool flag=false;
		while(tempLevel<=L){
	    		if(tempLevel>L){
 				break;
	    		}
	    		//if(flag==true){
			//	break;
	    		//}
			uint tempLevelID=tempLevel%2;
	    		uint newLevelID=(tempLevel+1)%2;
	    		uint candidateCnt=candidate_count[tempLevelID];
	    		if(candidateCnt==0){
				break;
	    		}
	    		candidate_count[tempLevelID]=0;
	    		for(uint j = 0; j < candidateCnt; j++){
				uint tempNode = candidate_set[tempLevelID][j];
                		double tempR = residue[tempLevelID][tempNode];
	        		U[tempLevelID][tempNode]=0;
                		residue[tempLevelID][tempNode] = 0;
				if(H[1][tempNode] == 0){
                    			H[0][finalreserve_count++] = tempNode;
                    			H[1][tempNode] = 1;
                		}
				finalReserve[tempNode]+=(w_i/Y_i)*tempR;
	        
				if(tempLevel==L){
	  				continue;
				}
				uint outSize = g.getOutSize(tempNode);
				double incre=tempR*(1-w_i/Y_i)/outSize;
				for(uint k = 0; k < outSize; k++){
                    			uint newNode = g.getOutVert(tempNode, k);
					residue[newLevelID][newNode] += incre;
					if(U[newLevelID][newNode] == 0){
                       				U[newLevelID][newNode] = 1;
						candidate_set[newLevelID][candidate_count[newLevelID]++]=newNode;
					}
                		}
            		}
	    		tempLevel++;
	    		Y_i-=w_i;
	    		w_i*=t/tempLevel;
		}
    	}
};




class AGPstruct:public SimStruct{
    public:
    	uint * U[2];
    	uint *candidate_set[2];
    	uint candidate_count[2];
    	double *residue[2];

    	AGPstruct(string name, string file_label, double epsilon, double para_t):
		SimStruct(name, file_label, epsilon, para_t) {
		candidate_count[0]=0;
		candidate_count[1]=0;
        	H[0] = new uint[vert];
        	H[1] = new uint[vert];
        	U[0] = new uint[vert];
        	U[1] = new uint[vert];
		candidate_set[0] = new uint[vert];
		candidate_set[1] = new uint[vert];
		residue[0]=new double[vert];
		residue[1]=new double[vert];
		finalReserve = new double[vert];
		finalreserve_count=0;
        	for(uint i = 0; i < vert; i++){
	    		residue[0][i]=0;
	    		residue[1][i]=0;
	    		finalReserve[i]=0;
            		H[0][i] = 0;
            		H[1][i] = 0;
            		U[0][i] = 0;
            		U[1][i] = 0;
	    		candidate_set[0][i]=0;
	    		candidate_set[1][i]=0;
		}
        	//cout << "====init done!====" << endl;
    	}
    	~AGPstruct() {
        	delete[] H[0];
        	delete[] H[1];
        	delete[] U[0];
        	delete[] U[1];
        	delete[] residue[0];
		delete[] residue[1];
		delete[] finalReserve;
    		delete[] candidate_set[0];
		delete[] candidate_set[1];
    	}      
    

    	void query(uint u){
		for(uint j = 0; j < finalreserve_count; j++){
                	finalReserve[H[0][j]] = 0;
                	H[1][H[0][j]] = 0;
        	}
		finalreserve_count=0;
        	uint tempLevel = 0;
	
		double w_i=1.0*exp((-1)*t);
		double Y_i=1;

        	residue[0][u] = 1;
        	candidate_set[0][0]=u;
		candidate_count[0]=1;

 		bool flag=false;
        	while(true){
            		uint tempLevelID=tempLevel%2;
	    		uint newLevelID=(tempLevel+1)%2;
	    		uint candidateCnt=candidate_count[tempLevelID];
	    		if(candidateCnt==0){
				break;
	    		}
	    		candidate_count[tempLevelID]=0;
	    		for(uint j = 0; j < candidateCnt; j++){
				uint tempNode = candidate_set[tempLevelID][j];
                		double tempR = residue[tempLevelID][tempNode];
	        		U[tempLevelID][tempNode]=0;
                		residue[tempLevelID][tempNode] = 0;
				if(H[1][tempNode] == 0){
                    			H[0][finalreserve_count++] = tempNode;
                    			H[1][tempNode] = 1;
                		}
				finalReserve[tempNode]+=(w_i/Y_i)*tempR;
	        		if(Y_i<eps){
					continue;
	    			}
				uint outSize = g.getOutSize(tempNode);
		
                		double incre = tempR* (1-w_i/Y_i)/outSize;
				if(tempR<eps*outSize){
					continue;
				}
				if(incre>eps){
					for(uint k = 0; k < outSize; k++){
                    				uint newNode = g.getOutVert(tempNode, k);
						residue[newLevelID][newNode] += incre;
                    				if(U[newLevelID][newNode] == 0){
                       					U[newLevelID][newNode] = 1;
							candidate_set[newLevelID][candidate_count[newLevelID]++]=newNode;
		    				}
					}
				}
				else{
					double sampling_pr=incre/eps;
					std::default_random_engine generator(seed);
					std::binomial_distribution<uint> distribution(outSize,(double)sampling_pr);
					uint expn=distribution(generator);
					if(expn==outSize){
						for(uint k = 0; k < outSize; k++){
                    					uint newNode = g.getOutVert(tempNode, k);
							residue[newLevelID][newNode] += incre;
                    					if(U[newLevelID][newNode] == 0){
                       						U[newLevelID][newNode] = 1;
								candidate_set[newLevelID][candidate_count[newLevelID]++]=newNode;
		    					}
						}
					}
					else{
						for(uint ri=0;ri<expn;ri++){
							uint tmpran=(uint)((R.drand())*(outSize-ri-1))+1;
							uint idchangefar=g.outPL[tempNode]+ri+tmpran;
							uint idchangenear=g.outPL[tempNode]+ri;
							uint tmpchange=g.outEL[idchangefar];
							g.outEL[idchangefar]=g.outEL[idchangenear];
							g.outEL[idchangenear]=tmpchange;						

							uint newNode=g.getOutVert(tempNode,ri);
							residue[newLevelID][newNode] += eps;
                    					if(U[newLevelID][newNode] == 0){
                       						U[newLevelID][newNode] = 1;
								candidate_set[newLevelID][candidate_count[newLevelID]++]=newNode;
		    					}
						}
					}
				}
            		}
	    		tempLevel++;
	    		Y_i-=w_i;
	    		w_i*=t/tempLevel;
		}
    	}
};





#endif
