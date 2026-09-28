#include <cstdint>
extern "C" void bucket_lookup(const uint64_t* keys,const uint64_t* query,const int64_t* directory,int64_t n,int64_t* lower,int64_t* upper){
 for(int64_t i=0;i<n;++i){uint64_t q=query[i];uint64_t bucket=q>>48;int64_t lo=directory[bucket],hi=directory[bucket+1],end=hi;
  while(lo<hi){int64_t m=lo+(hi-lo)/2;if(keys[m]<q)lo=m+1;else hi=m;}lower[i]=lo;hi=end;
  while(lo<hi){int64_t m=lo+(hi-lo)/2;if(keys[m]<=q)lo=m+1;else hi=m;}upper[i]=lo;
 }
}
