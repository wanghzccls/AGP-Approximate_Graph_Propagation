#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <algorithm>
#include <queue>
#include <functional>
#include <iostream>
#include <thread>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include <errno.h>
#include <random>
#include <ctime>


typedef unsigned int uint;



class util{
  public:
    util(){
    
    }

    ~util(){
    
    }      
    

    unsigned long long peak_mem(){
	char buf[1024];
	FILE *fp;
    	unsigned long long peaksize;
	fp=fopen("/proc/self/status","r");
	if(fp==NULL){
		cout<<"fp = null"<<endl;
		return -1;
	}
	while(fgets(buf,sizeof(buf)-1,fp)!=NULL){
		if(sscanf(buf,"VmPeak:%llu",&peaksize)>0){
			break;
		}
	}
	fclose(fp);
	return peaksize;
    }


    void process_mem_usage(double& vm_usage, double& resident_set){
   	using std::ios_base;
   	using std::ifstream;
   	using std::string;

   	vm_usage     = 0.0;
   	resident_set = 0.0;

   	// 'file' stat seems to give the most reliable results
   	//
   	ifstream stat_stream("/proc/self/stat",ios_base::in);

   	// dummy vars for leading entries in stat that we don't care about
   	//
   	string pid, comm, state, ppid, pgrp, session, tty_nr;
   	string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   	string utime, stime, cutime, cstime, priority, nice;
   	string O, itrealvalue, starttime;

   	// the two fields we want
   	//
   	unsigned long vsize;
   	long rss;

   	stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   	stat_stream.close();

   	long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
   	vm_usage     = vsize / 1024.0;
   	resident_set = rss * page_size_kb;
    }
  
};




#endif
