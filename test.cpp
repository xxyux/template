#include<iostream>
#include<algorithm>
using namespace std;

const int N=100010;
int a[N];

int main() {
    int n=5;
    for(int i=0;i<n;i++) cin>>a[i];
    sort(a,a+n);
    for(int i=0;i<n;i++) cout<<a[i]<<" ";
}