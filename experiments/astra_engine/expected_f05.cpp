#include <cmath>
#include <cstdint>
extern "C" void expected_f05(const float* p,int n,int k,int32_t* chosen){
 for(int row=0;row<n;++row){
  double pre[33][33]={},suf[33][33]={};pre[0][0]=1.;suf[k][0]=1.;
  for(int i=0;i<k;++i){double x=p[row*k+i];for(int t=0;t<=i+1;++t)pre[i+1][t]=(t<=i?pre[i][t]*(1-x):0)+(t?pre[i][t-1]*x:0);}
  for(int i=k-1;i>=0;--i){double x=p[row*k+i];for(int t=0;t<=k-i;++t)suf[i][t]=(t<=k-i-1?suf[i+1][t]*(1-x):0)+(t?suf[i+1][t-1]*x:0);}
  double best=pre[k][0];int bestk=0;
  for(int c=1;c<=k;++c){double f=0;for(int a=1;a<=c;++a)for(int b=0;b<=k-c;++b)f+=pre[c][a]*suf[c][b]*(5.*a/(4.*c+a+b));if(f>best){best=f;bestk=c;}}
  chosen[row]=bestk;
 }
}
