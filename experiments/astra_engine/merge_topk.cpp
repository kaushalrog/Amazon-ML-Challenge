#include <vector>
#include <cstdint>
extern "C" void merge_topk(float* best,int32_t* ids,const float* values,const int32_t* cols,const int32_t* ptr,int32_t n,int32_t k,int32_t offset){
 std::vector<float> scores(k);std::vector<int32_t> positions(k);
 for(int32_t r=0;r<n;++r){
  int32_t a=0,b=ptr[r],end=ptr[r+1];
  for(int32_t j=0;j<k;++j){
   bool take=b<end&&(a>=k||values[b]>best[(int64_t)r*k+a]||(values[b]==best[(int64_t)r*k+a]&&cols[b]+offset<ids[(int64_t)r*k+a]));
   if(take){scores[j]=values[b];positions[j]=cols[b++]+offset;}
   else{scores[j]=best[(int64_t)r*k+a];positions[j]=ids[(int64_t)r*k+a++];}
  }
  for(int32_t j=0;j<k;++j){best[(int64_t)r*k+j]=scores[j];ids[(int64_t)r*k+j]=positions[j];}
 }
}
