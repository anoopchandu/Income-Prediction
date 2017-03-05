#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace std;

int NUMATTR = 14;
int DATASIZE = 32561;
int TDATASIZE = 16281;
const int numattr = NUMATTR;

// This is the global poiner to all data
int (*dat)[15];
// This will store max value of attributes starting from 0
int ranges[15]={1,7,1,15,1,6,13,5,4,1,1,1,1,40,1};

int sc;
bool comp(int a,int b)
{
	return dat[a][sc]<dat[b][sc];
}

// Node object represents creating nodes in decision tree
// attr represents atrribute that node is holding
// value refers to true or false this is valid only for leaf node where attr = -1
// element at index i in childs vector will point a decision tree if value of attr is i 
class Node
{
public:
	int attr;
	bool value;
	vector<Node*> childs;
	Node(int a,bool v,int r)
	{
		attr = a; 
		value = v;
		childs.insert(childs.begin(),r,0);
	}
};

// This object represents :
// training data in form of data vector, which will store index of all instances to consider for constructing tree
// attributes in form of vis vector,
// final tree whose root is stored in root
class ID3
{
private:
	vector<int> data;
	vector<bool> vis;
	int dataSize, mismatches, numattr,tdataSize;
	Node *root;

	// this function calculates entropy for data from s to e 
	// in data vector and for column c
	double entLoss(int s, int e,int c)
	{
		int j,numvalues,k;
		double ent,prob,loss;

		numvalues = ranges[c]+1;
		int t[numvalues] ={}, f[numvalues]={};
		for(j=s;j<=e;j++)
		{
			k = data[j];
			t[dat[k][c]] += dat[k][numattr];
			f[dat[k][c]] += !dat[k][numattr];
		}

		for(loss=0,j=0;j<numvalues;j++)
		{
			if(t[j]!=0 && f[j] !=0)
			{
				prob = (t[j]*1.0)/(t[j]+f[j]);
				ent = -(prob*log(prob)+(1-prob)*log(1-prob));
				loss += (ent*((t[j]+f[j]))/(e-s+1));
			}
		}

		return loss;
	}

	// this function will find decision tree for data from s to e and 
	// attributes which has false in vis vector and return node of that tree
	// This is done by first finding which attrubute has low entropy and sorting 
	// based on that column and dividing data into segments which has same 
	// value for that attribute and recursively finding root of that segment leaving
	// already visited attrubites
	Node* train(int s, int e)
	{	
		// printf("aaaaaaa\n");
		int minc, i, j, k, p, c, cnt, numvalues,ju;
		double min, loss;
		
		// printf("processing %d %d\n",s,e);
		min = 9999;

		for(i=s, cnt=0; i<=e; i++)
		{
			k = data[i];
			// printf("%d %d\n",k,numattr);
			fflush(stdout);
			cnt = cnt+dat[k][numattr];
		}

		if(cnt==0)
		{
			// printf("false");
			return new Node(-1,false,0);
		}
		else if(cnt==e-s+1)
		{
			// printf("true");
			return new Node(-1,true,0);
		}

		// find minimum entropy column number and store
		// in  minc
		for(i=0,j=0;i<numattr;i++)
		{
			if(vis[i])
				continue;
			j=1;
			loss = entLoss(s,e,i);
			if(loss<min)
			{
				min = loss;
				minc = i;
			}
		}

		if(j==0)
		{
			// printf("j==0\n");
			if(cnt>((e-s+1)/2))
				return new Node(-1,true,0);
			else
				return new Node(-1,false,0);
		}

		Node &n = *(new Node(minc,true,ranges[minc]+1));
		
		sc = minc;
		sort(data.begin()+s,data.begin()+e+1,comp);
		
		// printf("best classifier %d\n",minc);
		// print_data();
		vis[minc]  = true;
		numvalues = ranges[minc]+1;
		k = s;
		p = dat[data[s]][minc];

		for(j=s+1;j<=e;j++)
		{
			i = data[j];
			c = dat[i][minc];
			if(p != c)
			{
				// cout<<"going to process"<<k<<" "<<j-1<<"in column"<<minc<<endl;
				// cout.flush();
				// cin>>ju;
				n.childs[p]=train(k,j-1);
				k = j;
			}
			p = c;
		}

		// cout<<"going to process"<<k<<" "<<j-1<<"in column"<<minc<<endl;
		// cout.flush();
		// cin>>ju;
		n.childs[p]=train(k,j-1);

		for(j=0;j<numvalues;j++)
		{
			if(n.childs[j]==0)
			{
				if(cnt>((e-s+1)/2))
					n.childs[j] = new Node(-1,true,0);
				else
					n.childs[j] = new Node(-1,false,0);
			}
		}
		vis[minc] = false;

		return &n;
	}

	// this function will print tree that is genertated
	int print_tree(Node *c)
	{
		int i=0;
		Node *n;
		int h=0,x;
		for(Node*n:c->childs)
		{
			if(n->attr ==-1)
			{
				if((n->value)==true)
					printf("attribute-%d value-%d child-true\n",c->attr,i);
				else
					printf("attribute-%d value-%d child-false\n",c->attr,i);
			}
			else
			{
				printf("attribute-%d value-%d child-%d\n",c->attr,i,n->attr);
				x = print_tree(n);
				if(x>h)
					h=x;
			}
			i++;
		}
		return h+1;
	}

public:
	// this constructor will take total datasize, total number of attributes
	// size of testing data
	ID3(int d,int a,int t)
	{
		int i;
		for(i=0;i<d;i++)
			data.push_back(i);
		for(i=0;i<a;i++)
			vis.push_back(false);
		dataSize = d;
		numattr = a;
		tdataSize = t;
	}

	void print_data()
	{
		int i,j,k;

		// cout<<"ranges"<<endl;
		// for(i=0;i<=numattr;i++)
		// 	cout<<ranges[i]<<" ";
		// cout<<endl;

		printf("Input - Ouptut\n");
		for(i=0;i<dataSize;i++)
		{
			k = data[i];
			for(j=0;j<=numattr;j++)
				printf("%d ",dat[k][j]);
			printf("\n");
		}
	}

	// This function will train tree by calling train(s,e)
	// before calling this function data has to be loaded into
	// "dat" 2D array
	void train()
	{
		root = train(0,dataSize-1);
	}

	int  print_tree()
	{
		return print_tree(root);
	}

	// This function will evaluate i th data instance on this tree
	bool evaluate(int q)
	{
		Node *n = root;
		while(n->attr!=-1)
			n = n->childs[dat[q][n->attr]];
		return n->value;
	}
	
	// This function will validate on training data
	void selfValidate()
	{
		int i, k,cnt=0;
		mismatches = 0;
		for(i=0;i<dataSize;i++)
		{
			k = data[i];
			if(dat[k][numattr] != evaluate(k))
				cnt++;
		}
		mismatches = cnt;
		printf("Mismatched %d, Data size %d, Attributes %d, Accuracy %lf%%\n",mismatches,dataSize,numattr, (1-((double)mismatches/dataSize))*100);
	}

	// This function will validate on testing data before calling this function
	// testing data has to be loaded into "dat" 2D array
	void validate()
	{
		int i, cnt=0;
		mismatches = 0;
		// int c=0,c1=0;
		for(i=0;i<tdataSize;i++)
		{
			// k = data[i];
			if(dat[i][numattr] != evaluate(i))
				cnt++;
			// c += evaluate(i);
			// c1 += dat[i][numattr];
		}
		// printf("ones in output %d ones in target %d\n",c,c1);
		mismatches = cnt;
		printf("Mismatched %d, Test Data size %d, Attributes %d, Accuracy %lf%%\n",mismatches,tdataSize,numattr, (1-((double)mismatches/tdataSize))*100);
	}
};

// This functio will load data from file into "dat" 2D array
int loadAllData(int src)
{
	int i,j;
	FILE *f ;
	if(src==1)
		f = fopen("adultdisc.data","r");
	else if(src==2)
		f = fopen("adultdisc.test","r");
	else
		return 0;
	for(i=0;i<DATASIZE;i++)
	{
		for(j=0;j<15;j++)
		{
			fscanf(f,"%d",&dat[i][j]);
		}
	}
	fclose(f);
}

int main()
{
	dat = (int (*)[15]) malloc(sizeof(int)*(15)*DATASIZE);

	clock_t start = clock();

	// instantiate ID3 class
	ID3 id3(DATASIZE,NUMATTR,TDATASIZE);

	// load training data and train on that data
	loadAllData(1);
	id3.train();

	clock_t end = clock();
	printf("Learning time - %lf seconds\n",((double)(end - start))/CLOCKS_PER_SEC);

	// load testing data and test on that data
	loadAllData(2);
	id3.validate();

}