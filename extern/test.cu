#include "kernel.h"
using namespace diffRender;

GArr2D<float> random_matrix(int n, int m){
    GArr2D<float> arr(n,m);
    CArr2D<float> arr_cpu(n,m);
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            arr_cpu(i,j) = rand()/(float)RAND_MAX;
        }
    }
    arr.assign(arr_cpu);
    return arr;
}

int main(){
    auto a = random_matrix(2, 2);
    auto b = random_matrix(2, 2);
    auto c = random_matrix(2, 2);
    c.reset();
    launch_add(c, a, b);
    LOG(a);
    LOG(b);
    LOG(c);
    return 0;
}