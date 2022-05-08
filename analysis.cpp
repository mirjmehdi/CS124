#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

using namespace std;

const int maxn = 2e3 + 100;

double mat[maxn][maxn];

void generate_random_graph(int n) {
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++) {
			mat[i][j] = (float) rand() / RAND_MAX;
			mat[j][i] = mat[i][j];
		}
	
}

double euclidean_distance(vector<double> p1, vector<double> p2) {
	double res = 0;
	for (int i = 0; i < p1.size(); i++)
		res += (p1[i] - p2[i]) * (p1[i] - p2[i]);
	return sqrt(res);
}

void generate_dim_graph(int n, int dim) {
	vector<vector<double> > points;
	for (int i = 0; i < n; i++) {
		vector<double> cur;
		for (int d = 0; d < dim; d++)
			cur.push_back((float) rand() / RAND_MAX);
		points.push_back(cur);
	}
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			mat[i][j] = euclidean_distance(points[i], points[j]);
		//	cout << mat[i][j] << endl;
		}
}

pair<double, pair<int, int> > edges[maxn * maxn];
int par[maxn];

int get(int x) {
	if (par[x] == -1)
		return x;
	par[x] = get(par[x]);
	return par[x];
}

double kruskal(int n) {
	memset(par, -1, sizeof par);
	int cnt = 0;
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++) {
			edges[cnt++] = make_pair(mat[i][j], make_pair(i, j));
		}
	double MST = 0.0;
	double max_edge = 0.0;
	sort(edges, edges + cnt);
	for (int i = 0; i < cnt; i++) {
		int fi = edges[i].second.first;
		int se = edges[i].second.second;
		if (get(fi) != get(se)) {
			MST += edges[i].first;
			par[get(fi)] = get(se);
			max_edge = max(max_edge, edges[i].first);
		}
	}
	return max_edge;
}

int main(int argc, char *argv[]) {
	srand(time(0));

//	int n = atoi(argv[1]);;
//	int dim = atoi(argv[2]);
	int dim = 0;
	int trials = 1;

	for (int n = 10; n < 2000; n += 10) {
		double avg = 0;
		double max_edge = 0;
		for (int trial = 0; trial < trials; trial++) {
			if (dim == 0)
				generate_random_graph(n);
			else
				generate_dim_graph(n, dim);
			max_edge = max(max_edge, kruskal(n));
		//	avg += kruskal(n);
		}
		cout << n << "\t" << max_edge << endl;
		//cout << avg / trials << endl;
	}
}
