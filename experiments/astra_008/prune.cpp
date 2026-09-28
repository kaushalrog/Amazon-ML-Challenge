#include <algorithm>
#include <vector>
#include <utility>
#include <cstdint>
extern "C" void prune_columns(float* values,const int32_t* rows,const int32_t* pointers,int32_t columns,int32_t name_boundary,int32_t keep) {
 std::vector<std::pair<float,int32_t>> a,b;
 for(int32_t col=0;col<columns;++col){
  a.clear();b.clear();
  for(int32_t p=pointers[col];p<pointers[col+1];++p) (rows[p]<name_boundary?a:b).emplace_back(values[p],p);
  for(auto* group : {&a,&b}){
   auto& v=*group;
   if(v.size()>(size_t)keep){
    std::nth_element(v.begin(),v.begin()+keep,v.end(),[](const auto& x,const auto& y){return x.first>y.first || (x.first==y.first && x.second<y.second);});
    for(size_t i=keep;i<v.size();++i)values[v[i].second]=0.f;
   }
  }
 }
}
