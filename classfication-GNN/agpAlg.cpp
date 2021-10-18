#include "agpAlg.h"
using namespace std;
using namespace Eigen;

namespace propagation
{

double Agp::agp_operation(string dataset,string agp_alg,uint mm,uint nn,int LL,double rmaxx,double alphaa,double tt,Eigen::Map<Eigen::MatrixXd> &feat)
{

    int NUMTHREAD=40; //Number of threads
    rmax=rmaxx;
    m=mm;
    n=nn;
    L=LL;
    alpha=alphaa;
    t=tt;
    dataset_name=dataset;

    el=vector<uint>(m);
    pl=vector<uint>(n+1);
    string dataset_el="data/"+dataset+"_adj_el.txt";
    const char *p1=dataset_el.c_str();
    if (FILE *f1 = fopen(p1, "rb"))
    {
        size_t rtn = fread(el.data(), sizeof el[0], el.size(), f1);
        if(rtn!=m)
            cout<<"Error! "<<dataset_el<<" Incorrect read!"<<endl;
        fclose(f1);
    }
    else
    {
        cout<<dataset_el<<" Not Exists."<<endl;
        exit(1);
    }
    string dataset_pl="data/"+dataset+"_adj_pl.txt";
    const char *p2=dataset_pl.c_str();

    if (FILE *f2 = fopen(p2, "rb"))
    {
        size_t rtn = fread(pl.data(), sizeof pl[0], pl.size(), f2);
        if(rtn!=n+1)
            cout<<"Error! "<<dataset_pl<<" Incorrect read!"<<endl;
        fclose(f2);
    }
    else
    {
        cout<<dataset_pl<<" Not Exists."<<endl;
        exit(1);
    }

    int dimension=feat.rows();
    vector<thread> threads;
    Du_a=vector<double>(n,0);
    Du_b=vector<double>(n,0);

    random_w = vector<int>(dimension);
    rowsum_pos = vector<double>(dimension,0);
    rowsum_neg = vector<double>(dimension,0);

    for(int i = 0 ; i < dimension ; i++ )
        random_w[i] = i;
    random_shuffle(random_w.begin(),random_w.end());

    double rrr;  //a=1-rrr; b=r;
    if((dataset_name=="Amazon2M_train"||dataset_name=="Amazon2M_full")&& L==4)
        rrr=0.2;
    else
        rrr=0.5;

    for(uint i=0; i<n; i++)
    {
        uint du=pl[i+1]-pl[i];
        Du_a[i]=pow(du,1-rrr);
        Du_b[i]=pow(du,rrr);
    }
    
    for(int i=0; i<dimension; i++)
    {
        for(uint j=0; j<n; j++)
        {
            if(feat(i,j)>0)
                rowsum_pos[i]+=feat(i,j);
            else
                rowsum_neg[i]+=feat(i,j);
        }
    }

    struct timeval t_start,t_end;
    double timeCost;
    clock_t start_t, end_t;
    gettimeofday(&t_start,NULL);

    cout<<"Begin propagation..."<<endl;
    int ti,start;
    int ends=0;
    start_t = clock();
    for( ti=1 ; ti <= dimension%NUMTHREAD ; ti++ )
    {
        start = ends;
        ends+=ceil((double)dimension/NUMTHREAD);
        if(agp_alg=="sgc_agp")
            threads.push_back(thread(&Agp::sgc_agp,this,feat,start,ends));
        else if(agp_alg=="appnp_agp")
            threads.push_back(thread(&Agp::appnp_agp,this,feat,start,ends));
        else if(agp_alg=="gdc_agp")
            threads.push_back(thread(&Agp::gdc_agp,this,feat,start,ends));
    }
    for( ; ti<=NUMTHREAD ; ti++ )
    {
        start = ends;
        ends+=dimension/NUMTHREAD;
        if(agp_alg=="sgc_agp")
            threads.push_back(thread(&Agp::sgc_agp,this,feat,start,ends));
        else if(agp_alg=="appnp_agp")
            threads.push_back(thread(&Agp::appnp_agp,this,feat,start,ends));
        else if(agp_alg=="gdc_agp")
            threads.push_back(thread(&Agp::gdc_agp,this,feat,start,ends));
    }
    for (int t = 0; t < NUMTHREAD ; t++)
        threads[t].join();
    vector<thread>().swap(threads);
    end_t = clock();
    double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    gettimeofday(&t_end, NULL);
    timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
    cout<<"The propagation time: "<<timeCost<<" s"<<endl;
    cout<<"The clock time : "<<total_t<<" s"<<endl;
    double dataset_size=(double)(((long long)m+n)*4+(long long)n*dimension*8)/1024.0/1024.0/1024.0;
    return dataset_size;
}

//SGC_AGP
void Agp::sgc_agp(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed)
{
    uint seed=time(NULL)^pthread_self();
    double** residue=new double*[2];
    for(int i=0; i<2; i++)
        residue[i]=new double[n];
    for(int it=st; it<ed; it++)
    {
        int w=random_w[it];
        double rowsum_p=rowsum_pos[w];
        double rowsum_n=rowsum_neg[w];
        double rmax_p=rowsum_p*rmax;
        double rmax_n=rowsum_n;

        if(dataset_name=="papers100M")
            rmax_n*=(rmax/50);
        else
            rmax_n*=rmax;

        double MaxPR=0; //max positive residue
        double MaxNR=0; //max negative residue(consider absolute value)
        for(uint ik=0; ik<n; ik++)
        {
            double tmpf=feats(w,ik)/Du_b[ik];
            residue[0][ik]=tmpf;
            residue[1][ik]=0;
            if(tmpf>MaxPR)
                MaxPR=tmpf;
            else if(tmpf<MaxNR)
                MaxNR=tmpf;
            feats(w,ik)=0;
        }
        uint j=0,k=0;
        for(int il=0; il<=L; il++)
        {
            if(dataset_name=="papers100M" && L==10)
            {
                if(il<3)
                    rmax_n=rowsum_n*(1.8e-9);
                else if(il>=3&&il<=5)
                    rmax_n=rowsum_n*(1.3e-9);
                else
                    rmax_n=rowsum_n*(0.3e-9);
            }
            j=il%2;
            k=1-j;
            if(((MaxPR<=rmax_p)&&(MaxNR>=rmax_n))||(il==L)){
                for(uint ik=0; ik<n; ik++)
                {
                    feats(w,ik)=residue[j][ik]*Du_b[ik];
                }
                break;
            }

            for(uint ik=0; ik<n; ik++)
            {
                double old=residue[j][ik];
                residue[j][ik]=0;
                if(old>rmax_p||old<rmax_n)
                {
                    uint im,v,dv;
                    double ran;
                    for(im=pl[ik]; im<pl[ik+1]; im++)
                    {
                        v=el[im];
                        dv=pl[v+1]-pl[v];
                       if(old>rmax_p*Du_a[v]||old<rmax_n*Du_a[v])
                       {
                            residue[k][v]+=old/dv;
                            double tmpf=residue[k][v];
                            if(tmpf>MaxPR)
                                MaxPR=tmpf;
                            else if(tmpf<MaxNR)
                                MaxNR=tmpf;
                       }
                        else
                        {
                            ran=rand_r(&seed)%RAND_MAX/(double)RAND_MAX;
                            break;
                        }
                    }
                    for(; im<pl[ik+1]; im++)
                    {
                        v=el[im];
                        if(ran*rmax_p*Du_a[v]<old)
                        {
                            residue[k][v]+=rmax_p/Du_a[v];
                            double tmpf=residue[k][v];
                            if(tmpf>MaxPR)
                                MaxPR=tmpf;
                            else if(tmpf<MaxNR)
                                MaxNR=tmpf;
                        }
                        else if (old<ran*rmax_n*Du_a[v])
                        {
                            residue[k][v]+=rmax_n/Du_a[v];
                            double tmpf=residue[k][v];
                            if(tmpf>MaxPR)
                                MaxPR=tmpf;
                            else if(tmpf<MaxNR)
                                MaxNR=tmpf;
                        }
                        else
                            break;
                    }
                }
            }
        }
    }
    for(int i=0; i<2; i++)
        delete[]residue[i];
    delete[]residue;
}

//APPNP_AGP
void Agp::appnp_agp(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed)
{
    uint seed=time(NULL)^pthread_self();
    double** residue=new double*[2];
    for(int i=0; i<2; i++)
        residue[i]=new double[n];

    for(int it=st; it<ed; it++)
    {
        int w=random_w[it];
        double rowsum_p=rowsum_pos[w];
        double rowsum_n=rowsum_neg[w];
        double rmax_p=rowsum_p*rmax;
        double rmax_n=rowsum_n;
        if(dataset_name=="papers100M")
            rmax_n*=(rmax/50);
        else
            rmax_n*=rmax;

        double MaxPR=0; //max positive residue
        double MaxNR=0; //max negative residue(consider absolute value)
        for(uint ik=0; ik<n; ik++)
        {
            double tmpf=feats(w,ik);
            residue[0][ik]=tmpf;
            if(tmpf>MaxPR)
                MaxPR=tmpf;
            else if(tmpf<MaxNR)
                MaxNR=tmpf;
            residue[1][ik]=0;
            feats(w,ik)=0;
        }
        int j=0,k=0;
        for(int il=0; il<=L; il++)
        {
            j=il%2;
            k=1-j;
            if(((MaxPR<=rmax_p)&&(MaxNR>=rmax_n))||(il==L)){
                for(uint ik=0; ik<n; ik++)
                    feats(w,ik)+=residue[j][ik];
                break;
            }

            for(uint ik=0; ik<n; ik++)
            {
                double old=residue[j][ik];
                residue[j][ik]=0;
                if(old>rmax_p||old<rmax_n)
                {
                    uint im,v;
                    double ran;
                    feats(w,ik)+=old*alpha;
                    for(im=pl[ik]; im<pl[ik+1]; im++)
                    {
                        v=el[im];
                        if(old>rmax_p*Du_a[v]/(1-alpha)||old<rmax_n*Du_a[v]/(1-alpha)){
                            residue[k][v]+=(1-alpha)*old/(Du_a[v]*Du_b[ik]);
                            double tmpf=residue[k][v];
                            if(tmpf>MaxPR)
                                MaxPR=tmpf;
                            else if(tmpf<MaxNR)
                                MaxNR=tmpf;
                        }
                        else
                        {
                            ran=rand_r(&seed)%RAND_MAX/(double)RAND_MAX;
                            break;
                        }
                    }
                    for(; im<pl[ik+1]; im++)
                    {
                        v=el[im];
                        if(ran*rmax_p*Du_a[v]/(1-alpha)<old){
                            residue[k][v]+=rmax_p/Du_a[v];
                            double tmpf=residue[k][v];
                            if(tmpf>MaxPR)
                                MaxPR=tmpf;
                            else if(tmpf<MaxNR)
                                MaxNR=tmpf;
                        }
                        else if (old<ran*rmax_n*Du_a[v]/(1-alpha)){
                            residue[k][v]+=rmax_n/Du_a[v];
                            double tmpf=residue[k][v];
                            if(tmpf>MaxPR)
                                MaxPR=tmpf;
                            else if(tmpf<MaxNR)
                                MaxNR=tmpf;
                        }
                        else
                            break;
                    }
                }
                else{
                    feats(w,ik)+=old;
                }
            }
        }
    }
    for(int i=0; i<2; i++)
        delete[] residue[i];
    delete[] residue;
}

//GDC_AGP
void Agp::gdc_agp(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed)
{
    uint seed=time(NULL)^pthread_self();
    double** residue=new double*[2];
    for(int i=0; i<2; i++)
        residue[i]=new double[n];
    double W[L+1]= {0};
    double Y[L+1]= {0};
    long long tempp[L+1];
    tempp[0]=1;
    tempp[1]=1;
    for(int ik=0; ik<=L; ik++)
    {
        if(ik>1)
            tempp[ik]=tempp[ik-1]*ik;
        W[ik]=exp(-t)*pow(t,(ik))/tempp[ik];
    }
    for(int ik=0; ik<=L; ik++)
    {
        Y[ik]=1;
        for(int ij=0; ij<=ik; ij++)
            Y[ik]-= W[ij];
    }
    for(int it=st; it<ed; it++)
    {
        int w=random_w[it];
        double rowsum_p=rowsum_pos[w];
        double rowsum_n=rowsum_neg[w];
        double rmax_p=rowsum_p*rmax;
        double rmax_n=rowsum_n;
        if(dataset_name=="papers100M")
            rmax_n*=(rmax/50);
        else
            rmax_n*=rmax;

        double MaxPR=0; //max positive residue
        double MaxNR=0; //max negative residue(consider absolute value)
        for(uint ik=0; ik<n; ik++)
        {
            double tmpf=feats(w,ik);
            residue[0][ik]=tmpf;
            if(tmpf>MaxPR)
                MaxPR=tmpf;
            else if(tmpf<MaxNR)
                MaxNR=tmpf;
            residue[1][ik]=0;
            feats(w,ik)=0;
        }

        int j=0,k=0;
        for(int il=0; il<=L; il++)
        {
            j=il%2;
            k=1-j;
            if(((MaxPR<=rmax_p)&&(MaxNR>=rmax_n))||(il==L)){
                for(uint ik=0; ik<n; ik++)
                    feats(w,ik)+=residue[j][ik];
                break;
            }
            for(uint ik=0; ik<n; ik++)
            {
                double old=residue[j][ik];
                residue[j][ik]=0;
                if(old>rmax_p||old<rmax_n)
                {
                    uint im,v;
                    double ran;
                    feats(w,ik)+=old*W[il]/Y[il];
                    for(im=pl[ik]; im<pl[ik+1]; im++)
                    {
                        v=el[im];
                        if(old>rmax_p*Du_a[v]/(Y[il+1]/Y[il])||old<rmax_n*Du_a[v]/(Y[il+1]/Y[il]))
                        {
                            residue[k][v]+=(Y[il+1]/Y[il])*old/(Du_b[ik]*Du_a[v]);
                            double tmpf=residue[k][v];
                            if(tmpf>MaxPR)
                                MaxPR=tmpf;
                            else if(tmpf<MaxNR)
                                MaxNR=tmpf;
                        }
                        else
                        {
                            ran=rand_r(&seed)%RAND_MAX/(double)RAND_MAX;
                            break;
                        }
                    }
                    for(; im<pl[ik+1]; im++)
                    {
                        v=el[im];
                        if(ran*rmax_p*Du_a[v]/(Y[il+1]/Y[il])<old)
                        {
                            residue[k][v]+=rmax_p/Du_a[v];
                            double tmpf=residue[k][v];
                            if(tmpf>MaxPR)
                                MaxPR=tmpf;
                            else if(tmpf<MaxNR)
                                MaxNR=tmpf;
                        }
                        else if (old<ran*rmax_n*Du_a[v]/(Y[il+1]/Y[il]))
                        {
                            residue[k][v]+=rmax_n/Du_a[v];
                            double tmpf=residue[k][v];
                            if(tmpf>MaxPR)
                                MaxPR=tmpf;
                            else if(tmpf<MaxNR)
                                MaxNR=tmpf;
                        }
                        else
                            break;
                    }
                }
            }
        }
    }
    for(int i=0; i<2; i++)
        delete[] residue[i];
    delete[] residue;
}
}
