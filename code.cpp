#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

using namespace std;

const int maxn = 3e5 + 100;
vector<vector<double> > points;

double euclidean_distance(int idx1, int idx2) {
	if (points[idx1].size() == 0) // for d=0, just generate a random number!
		return (float) rand() / RAND_MAX;

	double res = 0;
	for (int i = 0; i < points[idx1].size(); i++)
		res += (points[idx1][i] - points[idx2][i]) * (points[idx1][i] - points[idx2][i]);
	return sqrt(res);
}

void generate_dim_graph(int n, int dim) {
	points.clear();
	for (int i = 0; i < n; i++) {
		vector<double> cur;
		for (int d = 0; d < dim; d++)
			cur.push_back((float) rand() / RAND_MAX);
		points.push_back(cur);
	}
}

pair<float, pair<int, int> > edges[maxn * 300];
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
			double dis = euclidean_distance(i, j);
			if (dis < (1.0 / sqrt(n))) { // d = 2, 0
//			if (dis < (2.0 / pow(n, 1.0 / 3.0))) { // d = 3
//			if (dis < (2.25 / pow(n, 1.0 / 4.0))) { // d = 4
				edges[cnt++] = make_pair(dis, make_pair(i, j));
			}
		}
	double MST = 0.0;
	sort(edges, edges + cnt);
	for (int i = 0; i < cnt; i++) {
		int fi = edges[i].second.first;
		int se = edges[i].second.second;
		if (get(fi) != get(se)) {
			MST += edges[i].first;
			par[get(fi)] = get(se);
		}
	}
	return MST;
}

int main(int argc, char *argv[]) {
	srand(time(0));

	int dim = atoi(argv[1]);
	int trials = atoi(argv[2]);
	vector<int> nums = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};

	for (int i = 0; i < nums.size(); i++) {
		int n = nums[i];
		double avg = 0;
		for (int trial = 0; trial < trials; trial++) {
			generate_dim_graph(n, dim);
			avg += kruskal(n);
		}
		cout << n << " " << avg / trials << endl;
	}
}
