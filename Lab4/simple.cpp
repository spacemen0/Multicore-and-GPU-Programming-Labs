#include<cmath>
#include<iostream>
#include<iomanip>
using namespace std;

const int N = 100;
int main() {
    float a[N];
    for(int i=0;i<N;i++)
        a[i]=sqrt((float)i);
    for (auto &&i : a)
    {
        cout<<fixed << setprecision(6)<<i<<"  ";
    }
    
}