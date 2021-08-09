#include <iostream>
#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
typedef long double db;
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
template <typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
template <typename T>
using ordered_multiset = tree<T, null_type, less_equal<T>, rb_tree_tag, tree_order_statistics_node_update>;#define pll pair<ll, ll>
#define pb push_back
#define eb emplace_back
#define mp make_pair
#define ub(v, val) upper_bound(v.begin(), v.end(), val)
#define np(str) next_permutation(str.begin(), str.end())
#define lb(v, val) lower_bound(v.begin(), v.end(), val)
#define sortv(vec) sort(vec.begin(), vec.end())
#define rev(p) reverse(p.begin(), p.end());
#define v vector
#define pi 3.14159265358979323846264338327950288419716939937510
#define len length()
#define repc(i, s, e) for (ll i = s; i < e; i++)
#define fi first
#define se second
#define mset(a, val) memset(a, val, sizeof(a));
#define mt make_tuple
#define repr(i, n) for (i = n - 1; i >= 0; i--)
#define rep(i, n) for (i = 0; i < n; i++)
#define IOS                  
    ios::sync_with_stdio(0); 
    cin.tie(0);              
    cout.tie(0);
#define at(s, pos) *(s.find_by_order(pos))
#define set_ind(s, val) s.order_of_key(val)
long long int M = 1e9 + 7;
long long int inf = 9 * 1e18;
//CLOCK
ll begtime = clock();
#define time() cout << "\n\nTime elapsed: " << (clock() - begtime) * 1000 / CLOCKS_PER_SEC << " ms\n\n";
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
//CLOCK ENDED

//DEBUGGING
vector<string> vec_splitter(string s)
{
    s += ',';
    vector<string> res;
    while (!s.empty())
    {
        res.push_back(s.substr(0, s.find(',')));
        s = s.substr(s.find(',') + 1);
    }
    return res;
}
void debug_out(
    vector<string> __attribute__((unused)) args,
    __attribute__((unused)) int idx,
    __attribute__((unused)) int LINE_NUM) { cout << endl; }
template <typename Head, typename... Tail>
void debug_out(vector<string> args, int idx, int LINE_NUM, Head H, Tail... T)
{
    if (idx > 0)
        cout << " | ";
    else
        cout << "Line(" << LINE_NUM << ") ";
    stringstream ss;
    ss << H;
    cout << args[idx] << " = " << ss.str();
    debug_out(args, idx + 1, LINE_NUM, T...);
}
#ifndef ONLINE_JUDGE
#define debug(...) debug_out(vec_splitter(#__VA_ARGS__), 0, __LINE__, __VA_ARGS__)
#else
#define debug(...) 42
#endif
// DEBUGGING ENDED

ll n, m;
// modular exponentiation
ll binpow(ll val, ll deg)
{
    if (deg < 0)
        return 0;
    if (!deg)
        return 1 % M;
    if (deg & 1)
        return binpow(val, deg - 1) * val % M;
    ll res = binpow(val, deg >> 1);
    return (res * res) % M;
}
//binomial
ll modinv(ll n)
{
    return binpow(n, M - 2);
}
//GCD
ll gcd(ll a, ll b)
{
    if (b == 0)
        return a;
    else
        return gcd(b, a % b);
}
//modinverse when m is not prime
ll modInverse(ll a, ll m)
{
	ll m0 = m;
	ll y = 0, x = 1;

	if (m == 1)
		return 0;

	while (a > 1) {
		// q is quotient
		ll q = a / m;
		ll t = m;

		// m is remainder now, process same as
		// Euclid's algo
		m = a % m, a = t;
		t = y;

		// Update y and x
		y = x - q * y;
		x = t;
	}

	// Make x positive
	if (x < 0)
		x += m0;

	return x;
}
int main()
{
    // your code goes here
    IOS;
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    ll i, j, t, k, x, y, z, N;

    return 0;
}

//factorial
v<ll> fact, inv;
void factorial(ll n)
{
	fact[0] = 1;
	inv[0] = 1;
	for (ll i = 1; i <= n; i++)
	{
		fact[i] = i * fact[i - 1];
		fact[i] %= M;
		inv[i] = modinv(fact[i]);
	}
}
//combination
ll C(ll n, ll i)
{
	if (n < i)
		return 0;
	ll res = fact[n];
	ll div = (inv[n - i] * inv[i]);
	div %= M;
	return (res  * div ) % M;
}

//combination without modulo
ll C[N][N];
mset(C, 0);
for (i = 0; i < N; i++)
{
    for (j = 0; j <= min(i, ll(N - 1)); j++)
    {
        // Base Cases
        if (j == 0 || j == i)
            C[i][j] = 1;

        // Calculate value using previously
        // stored values
        else
            C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
    }
}
//matrix exponentiation
struct Matrix
{
    ll a[101][101] = {0}; //size can change
    Matrix operator*(const Matrix &other)
    {
        Matrix product;
        rep(i, 101) rep(j, 101) rep(k, 101)
        {
            product.a[i][j] = (product.a[i][j] + (ll)a[i][k] * other.a[k][j]) % M;
        }
        return product;
    }
};
Matrix expo_power(Matrix a, ll k)
{
    Matrix product;
    rep(i, 101) product.a[i][i] = 1;
    while (k > 0)
    {
        if (k % 2)
        {
            product = product * a;
        }
        a = a * a;
        k /= 2;
    }
    return product;
}
//other way around
v<v<ll>> mul(v<v<ll>> &a, v<v<ll>> &b)
{
    ll i, j, k;
    ll sz = a.size();
    v<v<ll>> res(sz, v<ll>(sz, 0));
    rep(i, sz)
    {
        rep(j, sz)
        {
            rep(k, sz)
            {
                (res[i][j] += a[i][k] * b[k][j]) %= M;
            }
        }
    }
    return res;
}
v<v<ll>> matpow(v<v<ll>> &a, ll k)
{
    ll i, j;
    ll sz = a.size();
    v<v<ll>> res(sz, v<ll>(sz, 0));
    rep(i, sz) res[i][i] = 1;
    while (k > 0)
    {
        if (k & 1)
        {
            res = mul(res, a);
        }
        a = mul(a, a);
        k /= 2;
    }
    return res;
}
//BFS normal
v<v<ll>> adj; // adjacency list representation
queue<ll> q;
v<ll> used(n, 0);
v<ll> d(n, inf), p(n, -1);

q.push(s);
used[s] = 1;
d[s] = 0;
while (!q.empty())
{
    ll V = q.front();
    q.pop();
    for (ll u : adj[V])
    {
        if (!used[u])
        {
            used[u] = 1;
            q.push(u);
            d[u] = d[V] + 1;
            p[u] = V;
        }
    }
}

if (!used[u])
{
    cout << "No path!";
}
else
{
    vector<int> path;
    for (int V = u; V != -1; V = p[V])
        path.push_back(V);
    reverse(path.begin(), path.end());
    cout << "Path: ";
    for (int V : path)
        cout << V << " ";
}

//DFS
vector<vector<int>> adj; // graph represented as an adjacency list
int n;                   // number of vertices

vector<bool> visited;

void dfs(ll v)
{
    visited[v] = true;
    for (ll u : adj[v])
    {
        if (!visited[u])
            dfs(u);
    }
}

vector<vector<int>> adj; // graph represented as an adjacency list
int n;                   // number of vertices

vector<int> color;

vector<int> time_in, time_out;
int dfs_timer = 0;

void dfs(int v)
{
    time_in[v] = dfs_timer++;
    color[v] = 1;
    for (int u : adj[v])
        if (color[u] == 0)
            dfs(u);
    color[v] = 2;
    time_out[v] = dfs_timer++;
}
//connected components(only number)
v<v<ll>> adj;
v<bool> used;
void dfs(ll v)
{
    used[v] = true;
    for (size_t i = 0; i < (int)adj[v].size(); ++i)
    {
        int to = adj[v][i];
        if (!used[to])
            dfs(to);
    }
}
ll find_comps()
{
    ll count = 0;
    for (int i = 0; i < used.size(); ++i)
        if (!used[i])
        {
            count++;
            dfs(i);
        }
    return count;
}

//CONNECTED COMPONENTS(also the list)
int n;
v<v<ll>> adj;
v<ll> used;
vector<int> comp;

void dfs(int v)
{
    used[v] = 1;
    comp.push_back(v);
    for (auto to : adj[v])
    {
        if (!used[to])
            dfs(to);
    }
}

void find_comps()
{
    for (int i = 0; i < n; ++i)
        used[i] = 0;
    for (int i = 0; i < n; ++i)
        if (!used[i])
        {
            comp.clear();
            dfs(i);
            cout << "Component:";
            for (size_t j = 0; j < comp.size(); ++j)
                cout << ' ' << comp[j];
            cout << endl;
        }
}
//strongly connected components
v<bool> used;
v<v<ll>> adj, adjr;
v<ll> order, component;
void dfs1(ll v)
{
    used[v] = true;
    for (size_t i = 0; i < adj[v].size(); ++i)
        if (!used[adj[v][i]])
            dfs1(adj[v][i]);
    order.pb(v);
}
void dfs2(ll v)
{
    used[v] = true;
    component.pb(v);
    for (size_t i = 0; i < adjr[v].size(); ++i)
        if (!used[adjr[v][i]])
            dfs2(adjr[v][i]);
}
int main()
{
    rep(i, m)
    {
        cin >> x >> y;
        adj[x - 1].pb(y - 1);
        adjr[y - 1].pb(x - 1);
    }

    used.assign(n, false);
    rep(i, n) if (!used[i])
        dfs1(i);
    used.assign(n, false);
    rep(i, n)
    {
        ll u = order[n - 1 - i];
        if (!used[u])
        {
            dfs2(u);
            ... printing next component... component.clear();
        }
    }
    //bipartite check
    vector<ll> side(n, -1);
    bool is_bipartite = true;
    queue<ll> q;
    for (ll st = 0; st < n; ++st)
    {
        if (side[st] == -1)
        {
            q.push(st);
            side[st] = 0;
            while (!q.empty())
            {
                ll v = q.front();
                q.pop();
                for (ll u : adj[v])
                {
                    if (side[u] == -1)
                    {
                        side[u] = side[v] ^ 1;
                        q.push(u);
                    }
                    else
                    {
                        is_bipartite &= side[u] != side[v];
                    }
                }
            }
        }
    }
    //FINDING BRIDGES
    int n;                   // number of nodes
    vector<vector<int>> adj; // adjacency list of graph

    vector<int> used;
    vector<int> tin, low;
    int timer;

    void dfs(int v, int p = -1)
    {
        used[v] = 1;
        tin[v] = low[v] = timer++;
        for (int to : adj[v])
        {
            if (to == p)
                continue;
            if (used[to])
            {
                low[v] = min(low[v], tin[to]);
            }
            else
            {
                dfs(to, v);
                low[v] = min(low[v], low[to]);
                if (low[to] > tin[v])
                    IS_BRIDGE(v, to);
            }
        }
    }

    void find_bridges()
    {
        timer = 0;
        used.assign(n, false);
        tin.assign(n, -1);
        low.assign(n, -1);
        for (int i = 0; i < n; ++i)
        {
            if (!used[i])
                dfs(i);
        }
    }

    //ARTICULATION POINTS
    ll n;                   // number of nodes
    vector<vector<ll>> adj; // adjacency list of graph

    vector<int> used;
    vector<ll> tin, low;
    ll timer;

    void dfs(ll v, ll p = -1)
    {
        used[v] = 1;
        tin[v] = low[v] = timer++;
        ll children = 0;
        for (ll to : adj[v])
        {
            if (to == p)
                continue;
            if (used[to])
            {
                low[v] = min(low[v], tin[to]);
            }
            else
            {
                dfs(to, v);
                low[v] = min(low[v], low[to]);
                if (low[to] >= tin[v] && p != -1)
                    IS_CUTPOINT(v);
                ++children;
            }
        }
        if (p == -1 && children > 1)
            IS_CUTPOINT(v);
    }
    //LCA binary lifting
    ll l;
    vector<vector<ll>> adj;
    ll timer;
    vector<ll> tin, tout;
    vector<vector<ll>> up;

    void dfs(ll v, ll p)
    {
        tin[v] = ++timer;
        up[v][0] = p;
        for (ll i = 1; i <= l; ++i)
            up[v][i] = up[up[v][i - 1]][i - 1];

        for (ll u : adj[v])
        {
            if (u != p)
                dfs(u, v);
        }
        tout[v] = ++timer;
    }

    bool is_ancestor(ll u, ll v)
    {
        return tin[u] <= tin[v] && tout[u] >= tout[v];
    }

    ll lca(ll u, ll v)
    {
        if (is_ancestor(u, v))
            return u;
        if (is_ancestor(v, u))
            return v;
        for (ll i = l; i >= 0; --i)
        {
            if (!is_ancestor(up[u][i], v))
                u = up[u][i];
        }
        return up[u][0];
    }
    void preprocess(ll root)
    {
        tin.resize(n);
        tout.resize(n);
        timer = 0;
        l = ceil(log2(n));
        up.assign(n, vector<ll>(l + 1));
        dfs(root, root);
    }
    //DIJKSTRA NORMAL
    const int inf = 1000000000;
    vector<vector<pair<ll, ll>>> adj;

    void dijkstra(ll s, vector<ll> & d, vector<ll> & p)
    {
        ll n = adj.size();
        d.assign(n, inf);
        p.assign(n, -1);
        vector<bool> u(n, false);

        d[s] = 0;
        for (ll i = 0; i < n; i++)
        {
            ll v = -1;
            for (ll j = 0; j < n; j++)
            {
                if (!u[j] && (v == -1 || d[j] < d[v]))
                    v = j;
            }

            if (d[v] == inf)
                break;

            u[v] = true;
            for (auto edge : adj[v])
            {
                ll to = edge.first;
                ll len = edge.second;

                if (d[v] + len < d[to])
                {
                    d[to] = d[v] + len;
                    p[to] = v;
                }
            }
        }
    }

    vector<ll> restore_path(ll s, ll t, vector<ll> const &p)
    {
        vector<ll> path;

        for (ll v = t; v != s; v = p[v])
            path.push_back(v);
        path.push_back(s);

        reverse(path.begin(), path.end());
        return path;
    }
    //DIJKSTRA SPARSE
    const int inf = 1000000000;
    vector<vector<pair<ll, ll>>> adj;

    void dijkstra(ll s, vector<ll> & d, vector<ll> & p)
    {
        ll n = adj.size();
        d.assign(n, inf);
        p.assign(n, -1);

        d[s] = 0;
        set<pair<ll, ll>> q;
        q.insert({0, s});
        while (!q.empty())
        {
            ll v = q.begin()->second;
            q.erase(q.begin());

            for (auto edge : adj[v])
            {
                ll to = edge.first;
                ll ln = edge.second;

                if (d[v] + ln < d[to])
                {
                    q.erase({d[to], to});
                    d[to] = d[v] + ln;
                    p[to] = v;
                    q.insert({d[to], to});
                }
            }
        }
    }
    //BEllMAN FORD
    void solve()
    {
        vector<ll> d(n, inf);
        d[v] = 0;
        vector<ll> p(n, -1);

        for (;;)
        {
            bool any = false;
            for (ll j = 0; j < m; ++j)
                if (d[e[j].a] < inf)
                    if (d[e[j].b] > d[e[j].a] + e[j].cost)
                    {
                        d[e[j].b] = d[e[j].a] + e[j].cost;
                        p[e[j].b] = e[j].a;
                        any = true;
                    }
            if (!any)
                break;
        }

        if (d[t] == inf)
            cout << "No path from " << v << " to " << t << ".";
        else
        {
            vector<ll> path;
            for (ll cur = t; cur != -1; cur = p[cur])
                path.push_back(cur);
            reverse(path.begin(), path.end());

            cout << "Path from " << v << " to " << t << ": ";
            for (size_t i = 0; i < path.size(); ++i)
                cout << path[i] << ' ';
        }
    }

    void solve()
    {
        vector<ll> d(n, inf);
        d[v] = 0;
        vector<ll> p(n - 1);
        ll x;
        for (ll i = 0; i < n; ++i)
        {
            x = -1;
            for (ll j = 0; j < m; ++j)
                if (d[e[j].a] < inf)
                    if (d[e[j].b] > d[e[j].a] + e[j].cost)
                    {
                        d[e[j].b] = max(-inf, d[e[j].a] + e[j].cost);
                        p[e[j].b] = e[j].a;
                        x = e[j].b;
                    }
        }

        if (x == -1)
            cout << "No negative cycle from " << v;
        else
        {
            ll y = x;
            for (ll i = 0; i < n; ++i)
                y = p[y];

            vector<ll> path;
            for (ll cur = y;; cur = p[cur])
            {
                path.push_back(cur);
                if (cur == y && path.size() > 1)
                    break;
            }
            reverse(path.begin(), path.end());

            cout << "Negative cycle: ";
            for (size_t i = 0; i < path.size(); ++i)
                cout << path[i] << ' ';
        }
    }

    //0-1 BFS
    vector<ll> d(n, inf);
    d[s] = 0;
    deque<ll> q;
    q.push_front(s);
    while (!q.empty())
    {
        ll v = q.front();
        q.pop_front();
        for (auto edge : adj[v])
        {
            ll u = edge.first;
            ll w = edge.second;
            if (d[v] + w < d[u])
            {
                d[u] = d[v] + w;
                if (w == 1)
                    q.push_back(u);
                else
                    q.push_front(u);
            }
        }
    }

    //FLOYD WARSHEll
    for (ll k = 0; k < n; ++k)
    {
        for (ll i = 0; i < n; ++i)
        {
            for (ll j = 0; j < n; ++j)
            {
                if (d[i][k] < inf && d[k][j] < inf)
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
            }
        }
    }
    // Shortest paths of fixed length(matrix exponentiation)
    ll inf = 3 * 1e18;
    ll n, m;
    v<v<ll>> mul(v<v<ll>> & a, v<v<ll>> & b)
    {
        ll i, j, k;
        ll sz = a.size();
        v<v<ll>> res(sz, v<ll>(sz, inf));
        rep(i, sz)
        {
            rep(j, sz)
            {
                rep(k, sz)
                {
                    res[i][j] = min(a[i][k] + b[k][j], res[i][j]);
                }
            }
        }
        return res;
    }
    v<v<ll>> matpow(v<v<ll>> & a, ll k)
    {
        ll i, j;
        ll sz = a.size();
        v<v<ll>> res(sz, v<ll>(sz, inf));
        rep(i, sz) res[i][i] = 0;
        while (k > 0)
        {
            if (k & 1)
            {
                res = mul(res, a);
            }
            a = mul(a, a);
            k /= 2;
        }
        return res;
    }
    //FINDING CYCLE(UNDIRECTED GRAPH)
    vector<vector<ll>> adj;
    vector<ll> color;
    vector<ll> parent;
    ll cycle_start, cycle_end;

    bool dfs(ll v, ll par)
    { // passing vertex and its parent vertex
        color[v] = 1;
        for (ll u : adj[v])
        {
            if (u == par)
                continue; // skipping edge to parent vertex
            if (color[u] == 0)
            {
                parent[u] = v;
                if (dfs(u, parent[u]))
                    return true;
            }
            else if (color[u] == 1)
            {
                cycle_end = v;
                cycle_start = u;
                return true;
            }
        }
        color[v] = 2;
        return false;
    }

    void find_cycle()
    {
        color.assign(n, 0);
        parent.assign(n, -1);
        cycle_start = -1;

        for (ll v = 0; v < n; v++)
        {
            if (color[v] == 0 && dfs(v, parent[v]))
                break;
        }

        if (cycle_start == -1)
        {
        }
        else
        {
            vector<ll> cycle;
            cycle.push_back(cycle_start);
            for (ll v = cycle_end; v != cycle_start; v = parent[v])
                cycle.push_back(v);
            cycle.push_back(cycle_start);
            reverse(cycle.begin(), cycle.end());
            cout << cycle.size() - 1 << '\n';
            ll i;
            rep(i, cycle.size() - 1)
                    cout<< cycle[i] + 1 << " ";
        }
    }
    //FINDING CYCLE (directed graph)
    ll n;
    vector<vector<ll>> adj;
    vector<ll> color;
    vector<ll> parent;
    ll cycle_start, cycle_end;

    bool dfs(ll v)
    {
        color[v] = 1;
        for (ll u : adj[v])
        {
            if (color[u] == 0)
            {
                parent[u] = v;
                if (dfs(u))
                    return true;
            }
            else if (color[u] == 1)
            {
                cycle_end = v;
                cycle_start = u;
                return true;
            }
        }
        color[v] = 2;
        return false;
    }

    void find_cycle()
    {
        color.assign(n, 0);
        parent.assign(n, -1);
        cycle_start = -1;

        for (ll v = 0; v < n; v++)
        {
            if (color[v] == 0 && dfs(v))
                break;
        }

        if (cycle_start == -1)
        {
            cout << "Acyclic" << endl;
        }
        else
        {
            vector<ll> cycle;
            cycle.push_back(cycle_start);
            for (ll v = cycle_end; v != cycle_start; v = parent[v])
                cycle.push_back(v);
            cycle.push_back(cycle_start);
            reverse(cycle.begin(), cycle.end());

            cout << "Cycle found: ";
            for (ll v : cycle)
                cout << v << " ";
            cout << endl;
        }
    }

    //FINDING NEGATIVE CYCLES
    struct Edge
    {
        ll a, b, cost;
    };

    ll n, m;
    vector<Edge> edges;
    const ll inf = 1000000000;

    void solve()
    {
        vector<ll> d(n);
        vector<ll> p(n, -1);
        ll x;
        for (ll i = 0; i < n; ++i)
        {
            x = -1;
            for (Edge e : edges)
            {
                if (d[e.a] + e.cost < d[e.b])
                {
                    d[e.b] = d[e.a] + e.cost;
                    p[e.b] = e.a;
                    x = e.b;
                }
            }
        }

        if (x == -1)
        {
            cout << "No negative cycle found.";
        }
        else
        {
            for (ll i = 0; i < n; ++i)
                x = p[x];

            vector<ll> cycle;
            for (ll v = x;; v = p[v])
            {
                cycle.push_back(v);
                if (v == x && cycle.size() > 1)
                    break;
            }
            reverse(cycle.begin(), cycle.end());

            cout << "Negative cycle: ";
            for (ll v : cycle)
                cout << v << ' ';
            cout << endl;
        }
    }
    //TOPOLOGICAL SORTING
    v<ll> ans;
    v<ll> used;
    ll flg=0;
    void dfs(ll s)
    {
        used[s] = 1;
        for (ll u : adj[s])
        {
            if (!used[u])
                dfs(u);
            else if(used[u]==1){
                flg=1;
                return;
            }
        }
        used[s]=2;
        ans.push_back(s);
    }

    void topological_sort()
    {
        used.assign(n, 0);
        ans.clear();
        for (ll i = 0; i < n; ++i)
        {
            if (!used[i])
                dfs(i);
        }
        reverse(ans.begin(), ans.end());
    }
    //FLOWS
    //Ford-Fulkerson
    v<v<ll>> adj, capacity;
    void add_edge(ll v, ll u, ll cap)
    {
        capacity[v][u] = cap;
        capacity[u][v] = 0;
        adj[v].pb(u);
        adj[u].pb(v);
    }
    ll bfs(ll s, ll t, vector<ll> & parent)
    {
        fill(parent.begin(), parent.end(), -1);
        parent[s] = -2;
        queue<pair<ll, ll>> q;
        q.push({s, inf});

        while (!q.empty())
        {
            ll cur = q.front().first;
            ll flow = q.front().second;
            q.pop();

            for (ll next : adj[cur])
            {
                if (parent[next] == -1 && capacity[cur][next])
                {
                    parent[next] = cur;
                    ll new_flow = min(flow, capacity[cur][next]);
                    if (next == t)
                        return new_flow;
                    q.push({next, new_flow});
                }
            }
        }

        return 0;
    }
    ll maxflow(ll s, ll t)
    {
        ll flow = 0;
        vector<ll> parent(n); //n is the number of vertices in graph
        ll new_flow;
        while ((new_flow = bfs(s, t, parent)))
        {
            flow += new_flow;
            ll cur = t;
            while (cur != s)
            {
                ll prev = parent[cur];
                capacity[prev][cur] -= new_flow;
                capacity[cur][prev] += new_flow;
                cur = prev;
            }
        }

        return flow;
    }
    //flows
    //edmond karp method
    vector<vector<ll>> capacity, netflow;
vector<vector<ll>> adj;

ll bfs(ll s, ll t, vector<ll>& parent) {
	fill(parent.begin(), parent.end(), -1);
	parent[s] = -2;
	queue<pair<ll, ll>> q;
	q.push({s, inf});

	while (!q.empty()) {
		ll cur = q.front().first;
		ll flow = q.front().second;
		q.pop();

		for (ll next : adj[cur]) {
			if (parent[next] == -1 && capacity[cur][next]) {
				parent[next] = cur;
				ll new_flow = min(flow, capacity[cur][next]);
				if (next == t)
					return new_flow;
				q.push({next, new_flow});
			}
		}
	}

	return 0;
}
ll maxflow(ll s, ll t, ll n) {
	ll flow = 0;
	vector<ll> parent(n);
	ll new_flow;

	while (new_flow = bfs(s, t, parent)) {
		flow += new_flow;
		ll cur = t;
		while (cur != s) {
			ll prev = parent[cur];
			capacity[prev][cur] -= new_flow;
			netflow[prev][cur] += new_flow;
			capacity[cur][prev] += new_flow;
			netflow[cur][prev] -= new_flow;
			cur = prev;
		}
	}
	return flow;
}
    //Dinic's algorithm
    struct FlowEdge
    {
        ll v, u;
        long long cap, flow = 0;
        FlowEdge(ll v, ll u, long long cap) : v(v), u(u), cap(cap) {}
    };

    struct Dinic
    {
        const long long flow_inf = 1e18;
        v<FlowEdge> edges;
        v<v<ll>> adj;
        ll n, m = 0;
        ll s, t;
        v<ll> level, ptr;
        queue<ll> q;

        Dinic(ll n, ll s, ll t) : n(n), s(s), t(t)
        {
            adj.resize(n);
            level.resize(n);
            ptr.resize(n);
        }

        void add_edge(ll v, ll u, long long cap)
        {
            edges.emplace_back(v, u, cap);
            edges.emplace_back(u, v, 0);
            adj[v].push_back(m);
            adj[u].push_back(m + 1);
            m += 2;
        }

        bool bfs()
        {
            while (!q.empty())
            {
                ll v = q.front();
                q.pop();
                for (ll id : adj[v])
                {
                    if (edges[id].cap - edges[id].flow < 1)
                        continue;
                    if (level[edges[id].u] != -1)
                        continue;
                    level[edges[id].u] = level[v] + 1;
                    q.push(edges[id].u);
                }
            }
            return level[t] != -1;
        }

        long long dfs(ll v, long long pushed)
        {
            if (pushed == 0)
                return 0;
            if (v == t)
                return pushed;
            for (ll &cid = ptr[v]; cid < (ll)adj[v].size(); cid++)
            {
                ll id = adj[v][cid];
                ll u = edges[id].u;
                if (level[v] + 1 != level[u] || edges[id].cap - edges[id].flow < 1)
                    continue;
                long long tr = dfs(u, min(pushed, edges[id].cap - edges[id].flow));
                if (tr == 0)
                    continue;
                edges[id].flow += tr;
                edges[id ^ 1].flow -= tr;
                return tr;
            }
            return 0;
        }

        long long flow()
        {
            long long f = 0;
            while (true)
            {
                fill(level.begin(), level.end(), -1);
                level[s] = 0;
                q.push(s);
                if (!bfs())
                    break;
                fill(ptr.begin(), ptr.end(), 0);
                while (long long pushed = dfs(s, flow_inf))
                {
                    f += pushed;
                }
            }
            return f;
        }
    };

    int main()
    {
        Dinic f(total_vertices, source, sink);
        f.add_edge(v, u, cap);
        f.flow();
    }
    //min cost max flow
struct Edge {
	ll v, rpos;
	ll cap, cost, flow = 0;
	Edge() {}
	Edge(ll v, ll p, ll cap, ll c) : v(v), rpos(p), cap(cap), cost(c) {}
};

struct min_cost_maxflow {
	ll n;
	vector <vector<Edge>> E;
	vector<ll> mCap, dis, par, pos, vis;

	min_cost_maxflow (ll n) : n(n), E(n) {
		mCap.resize(n);
		dis.resize(n);
		par.resize(n);
		pos.resize(n);
	}

	void add_edge(ll u, ll v, ll cap, ll cost = 0) {
		E[u].emplace_back(v, E[v].size(), cap, cost);
		E[v].emplace_back(u, E[u].size() - 1, 0, -cost);
	}

	inline bool SPFA(ll S, ll T) {
		vis.assign(n, 0);
		for (ll i = 0; i < n; i++) mCap[i] = dis[i] = inf;
		queue <ll> q; q.push(S);
		dis[S] = 0, vis[S] = 1;

		while (!q.empty()) {
			ll i = 0, u = q.front(); q.pop();
			vis[u] = 0;
			for (auto &e : E[u]) {
				ll v = e.v;
				ll f = e.cap - e.flow, w = dis[u] + e.cost;
				if (f > 0 && dis[v] > w) {
					dis[v] = w, par[v] = u, pos[v] = i;
					mCap[v] = min(mCap[u], f);
					if (!vis[v]) q.push(v);
					vis[v] = 1;
				}
				++i;
			}
		}
		return dis[T] != inf;
	}

	pair <ll, ll> solve(ll S, ll T) {
		ll F = 0, C = 0;
		while (SPFA(S, T)) {
			ll v = T;
			ll f = mCap[v];
			F += f;
			while (v != S) {
				ll u = par[v];
				Edge &e = E[u][pos[v]];
				e.flow += f;
				E[v][e.rpos].flow -= f;
				v = u;
			}
			C += dis[T] * f;
		}
		return make_pair(F, C);
	}
};
//other way(cp algo)
    struct Edge
{
    int from, to, capacity, cost;
};

vector<vector<ll>> adj, cost, capacity;

const ll INF = 1e9;

void shortest_paths(ll n, ll v0, vector<ll>& d, vector<ll>& p) {
    d.assign(n, INF);
    d[v0] = 0;
    vector<bool> inq(n, false);
    queue<ll> q;
    q.push(v0);
    p.assign(n, -1);

    while (!q.empty()) {
        ll u = q.front();
        q.pop();
        inq[u] = false;
        for (ll v : adj[u]) {
            if (capacity[u][v] > 0 && d[v] > d[u] + cost[u][v]) {
                d[v] = d[u] + cost[u][v];
                p[v] = u;
                if (!inq[v]) {
                    inq[v] = true;
                    q.push(v);
                }
            }
        }
    }
}

ll min_cost_flow(ll N, vector<Edge> edges, ll K, ll s, ll t) {
    adj.assign(N, vector<ll>());
    cost.assign(N, vector<ll>(N, 0));
    capacity.assign(N, vector<ll>(N, 0));
    for (Edge e : edges) {
        adj[e.from].push_back(e.to);
        adj[e.to].push_back(e.from);
        cost[e.from][e.to] = e.cost;
        cost[e.to][e.from] = -e.cost;
        capacity[e.from][e.to] = e.capacity;
    }

    ll flow = 0;
    ll cost = 0;
    vector<ll> d, p;
    while (flow < K) {
        shortest_paths(N, s, d, p);
        if (d[t] == INF)
            break;

        // find max flow on that path
        ll f = K - flow;
        ll cur = t;
        while (cur != s) {
            f = min(f, capacity[p[cur]][cur]);
            cur = p[cur];
        }

        // apply flow
        flow += f;
        cost += f * d[t];
        cur = t;
        while (cur != s) {
            capacity[p[cur]][cur] -= f;
            capacity[cur][p[cur]] += f;
            cur = p[cur];
        }
    }

    if (flow < K)
        return -1;
    else
        return cost;
}
Edge add_edge(ll from, ll to, ll capacity, ll cost) {
	Edge e;
	e.from = from;
	e.to = to;
	e.cost = cost;
	e.capacity = capacity;
	return e;
}
    //0-1 KNAPSACK
    ll knapSack(ll W, ll wt[], ll val[], ll n)
    {
        ll i, w;
        ll K[n + 1][W + 1];

        // Build table K[][] in bottom up manner
        for (i = 0; i <= n; i++)
        {
            for (w = 0; w <= W; w++)
            {
                if (i == 0 || w == 0)
                    K[i][w] = 0;
                else if (wt[i - 1] <= w)
                    K[i][w] = max(
                        val[i - 1] + K[i - 1][w - wt[i - 1]],
                        K[i - 1][w]);
                else
                    K[i][w] = K[i - 1][w];
            }
        }

        return K[n][W];
    }

    int main()
    {
        ll val[] = {60, 100, 120};
        ll wt[] = {10, 20, 30};
        ll W = 50;
        ll n = sizeof(val) / sizeof(val[0]);
        printf("%d", knapSack(W, wt, val, n));
        return 0;
    }
    //primality test
    bool isPrime(ll x)
    {
        for (ll d = 2; d * d <= x; d++)
        {
            if (x % d == 0)
                return false;
        }
        return true;
    }
    //LCS
    /* Dynamic Programming C++ implementation of LCS problem */
#include <bits/stdc++.h>
    using namespace std;

    ll max(ll a, ll b);

    /* Returns length of LCS for X[0..m-1], Y[0..n-1] */
    ll lcs(string X, string Y, ll n, ll m)
    {
        ll L[n + 1][m + 1];
        ll i, j;

        /* Following steps build L[m+1][n+1] in
           bottom up fashion. Note that L[i][j]
           contains length of LCS of X[0..i-1]
           and Y[0..j-1] */
        for (i = 0; i <= n; i++)
        {
            for (j = 0; j <= m; j++)
            {
                if (i == 0 || j == 0)
                    L[i][j] = 0;

                else if (X[i - 1] == Y[j - 1])
                    L[i][j] = L[i - 1][j - 1] + 1;

                else
                    L[i][j] = max(L[i - 1][j], L[i][j - 1]);
            }
        }

        /* L[m][n] contains length of LCS
        for X[0..n-1] and Y[0..m-1] */
        return L[n][m];
    }
    // for printing the LCS
    v<char> com; //lcs string
    void lcs(string X, string Y, ll n, ll m)
    {
        ll L[n + 1][m + 1];
        ll i, j;
        for (i = 0; i <= n; i++)
        {
            for (j = 0; j <= m; j++)
            {
                if (i == 0 || j == 0)
                    L[i][j] = 0;

                else if (X[i - 1] == Y[j - 1])
                    L[i][j] = L[i - 1][j - 1] + 1;

                else
                    L[i][j] = max(L[i - 1][j], L[i][j - 1]);
            }
        }
        i = n, j = m;
        while (i > 0 && j > 0)
        {
            if (X[i - 1] == Y[j - 1])
            {
                com.pb(X[i - 1]); // Put current character in result
                i--;
                j--; // reduce values of i, j and index
            }
            else if (L[i - 1][j] > L[i][j - 1])
                i--;
            else
                j--;
        }
        rev(com);
    }

    /* Utility function to get max of 2 integers */
    ll max(ll a, ll b)
    {
        return (a > b) ? a : b;
    }

    // Driver Code
    int main()
    {
        char X[] = "AGGTAB";
        char Y[] = "GXTXAYB";

        ll m = strlen(X);
        ll n = strlen(Y);

        cout << "Length of LCS is "
             << lcs(X, Y, m, n);

        return 0;
    }
    //longest increasing subsequence(N^2)
    /* Dynamic Programming C++ implementation
       of LIS problem */
#include <bits/stdc++.h>
    using namespace std;

    /* lis() returns the length of the longest
      increasing subsequence in arr[] of size n */
    ll lis(ll arr[], ll n)
    {
        ll lis[n];

        lis[0] = 1;

        /* Compute optimized LIS values in
           bottom up manner */
        for (ll i = 1; i < n; i++)
        {
            lis[i] = 1;
            for (ll j = 0; j < i; j++)
                if (arr[i] > arr[j] && lis[i] < lis[j] + 1)
                    lis[i] = lis[j] + 1;
        }

        // Return maximum value in lis[]
        return *max_element(lis, lis + n);
    }

    /* Driver program to test above function */
    int main()
    {
        ll arr[] = {10, 22, 9, 33, 21, 50, 41, 60};
        ll n = sizeof(arr) / sizeof(arr[0]);
        printf("Length of lis is %d\n", lis(arr, n));

        return 0;
    }

    //lis(nlogn) increasing
    ll lis(vector<ll> a)
    {
        ll n = a.size();
        vector<ll> d(n + 1, inf);
        d[0] = -inf;
        for (ll i = 0; i < n; i++)
        {
            ll j = upper_bound(d.begin(), d.end(), a[i]) - d.begin();
            if (d[j - 1] < a[i] && a[i] < d[j])
                d[j] = a[i];
        }
        ll ans = 0;
        for (ll i = 0; i <= n; i++)
        {
            if (d[i] < inf)
                ans = i;
        }
        return ans;
    }
    //lis(nlogn) non-decreasing
    ll lis(vector<ll> a)
    {
        ll n = a.size();
        vector<ll> d(n + 1, inf);
        d[0] = -inf;
        for (ll i = 0; i < n; i++)
        {
            ll j = upper_bound(d.begin(), d.end(), a[i]) - d.begin();
            if (d[j - 1] <= a[i] && a[i] < d[j])
                d[j] = a[i];
        }
        ll ans = 0;
        for (ll i = 0; i <= n; i++)
        {
            if (d[i] < inf)
                ans = i;
        }
        return ans;
    }
    // modular exponentiation
    ll binpow(ll val, ll deg)
    {
        if (!deg)
            return 1 % M;
        if (deg & 1)
            return binpow(val, deg - 1) * val % M;
        ll res = binpow(val, deg >> 1);
        return (res * res) % M;
    }
    //MODULO INVERSE
    ll x, y;
    ll g = extended_euclidean(a, m, x, y);
    if (g != 1)
    {
        cout << "No solution!";
    }
    else
    {
        x = (x % m + m) % m;
        cout << x << endl;
    }
    //factorization
    v<ll> primeFactors(ll n)
{
	v<ll> factors;
	// Print the number of 2s that divide n
	while (n % 2 == 0)
	{
		factors.pb(2);
		n = n / 2;
	}

	// n must be odd at this point. So we can skip
	// one element (Note i = i +2)
	for (ll i = 3; i*i<=n; i = i + 2)
	{
		// While i divides n, print i and divide n
		while (n % i == 0)
		{
			factors.pb(i);
			n = n / i;
		}
	}

	// This condition is to handle the case when n
	// is a prime number greater than 2
	if (n > 2) {
		factors.pb(n);
	}
	return factors;
}
    //factorization in log(n)
    ll spf[N];
    spf[1] = 1;
    for (i = 2; i < N; i++)
        spf[i] = i;
    for (i = 4; i < N; i += 2)
        spf[i] = 2;
    for (i = 3; i * i < N; i++)
    {
        if (spf[i] == i)
        {
            for (j = i * i; j < N; j += i)
                if (spf[j] == j)
                    spf[j] = i;
        }
    }
    while (x != 1)
    {
        ret.push_back(spf[x]);
        x = x / spf[x];
    }
    //sieve of eratosthenese
    vector<ll> is_prime(n + 1, 1);
    is_prime[0] = is_prime[1] = 0;
    for (i = 2; i <= n; i++)
    {
        if (is_prime[i] && (long long)i * i <= n)
        {
            for (j = i * i; j <= n; j += i)
                is_prime[j] = 0;
        }
    }

    ll binomial(ll n, ll m)
    {

        ll j = m + n;
        ll b = (fact[j] * (modinv(fact[n])) % M * (modinv(fact[m])) % M) % M;
        return b;
    }
    //euler totient
    ll phi(ll n)
    {
        ll result = n;
        for (ll i = 2; i * i <= n; i++)
        {
            if (n % i == 0)
            {
                while (n % i == 0)
                    n /= i;
                result -= result / i;
            }
        }
        if (n > 1)
            result -= result / n;
        return result;
    }
    //maximum sum subsegment
   int maxSubArraySum(v<int> a)
{
   int max_so_far = a[0];
   int curr_max = a[0];
 
   for (int i = 1; i < a.size(); i++)
   {
        curr_max = max(a[i], curr_max+a[i]);
        max_so_far = max(max_so_far, curr_max);
   }
   return max_so_far;
}
    //GCD
    ll gcd(ll a, ll b)
    {
        if (b == 0)
            return a;
        else
            return gcd(b, a % b);
    }
    //LCM
    ll lcm(ll a, ll b)
    {
        return (a * b) / gcd(a, b);
    }
    // Extended Euclidean algorithm
ll gcd(ll a, ll b, ll& x, ll& y) {
    x = 1, y = 0;
    ll x1 = 0, y1 = 1, a1 = a, b1 = b;
    while (b1) {
        ll q = a1 / b1;
        tie(x, x1) = make_tuple(x1, x - q * x1);
        tie(y, y1) = make_tuple(y1, y - q * y1);
        tie(a1, b1) = make_tuple(b1, a1 - q * b1);
    }
    return a1;
}
//sqrt decomposition
int query(int l,int r){
int sum = 0;
int c_l = l / blocksize,   c_r = r / blocksize;
if (c_l == c_r)
    for (int i=l; i<=r; ++i)
        sum += a[i];
else {
    for (int i=l, end=(c_l+1)*blocksize-1; i<=end; ++i)
        sum += a[i];
    for (int i=c_l+1; i<=c_r-1; ++i)
        sum += b[i];
    for (int i=c_r*blocksize; i<=r; ++i)
        sum += a[i];
}
return sum;
}
//mos algorithm
int block_size;

struct Query {
	int l, r, idx;
	
	inline pair<int, int> toPair() const {
		return make_pair(l / block, ((l / block) & 1) ? -r : +r);
	}
};

inline bool operator<(const Query &a, const Query &b) {
	return a.toPair() < b.toPair();
}

vector<int> mo_s_algorithm(vector<Query> queries) {
    vector<int> answers(queries.size());
    sort(queries.begin(), queries.end());

    // TODO: initialize data structure

    int cur_l = 0;
    int cur_r = -1;
    // invariant: data structure will always reflect the range [cur_l, cur_r]
    for (Query q : queries) {
        while (cur_l > q.l) {
            cur_l--;
            add(cur_l);
        }
        while (cur_r < q.r) {
            cur_r++;
            add(cur_r);
        }
        while (cur_l < q.l) {
            remove(cur_l);
            cur_l++;
        }
        while (cur_r > q.r) {
            remove(cur_r);
            cur_r--;
        }
        answers[q.idx] = get_answer();
    }
    return answers;
}
//linear diophantine equation
ll gcd(ll a, ll b, ll& x, ll& y) {
	if (b == 0) {
		x = 1;
		y = 0;
		return a;
	}
	ll x1, y1;
	ll d = gcd(b, a % b, x1, y1);
	x = y1;
	y = x1 - y1 * (a / b);
	return d;
}

bool find_any_solution(ll a, ll b, ll c, ll &x0, ll &y0, ll &g) {
	g = gcd(abs(a), abs(b), x0, y0);
	if (c % g) {
		return false;
	}

	x0 *= c / g;
	y0 *= c / g;
	if (a < 0) x0 = -x0;
	if (b < 0) y0 = -y0;
	return true;
}
void shift_solution(ll & x, ll & y, ll a, ll b, ll cnt) {
	x += cnt * b;
	y -= cnt * a;
}

ll find_all_solutions(ll a, ll b, ll c, ll minx, ll maxx, ll miny, ll maxy) {
	ll x, y, g;
	if (!find_any_solution(a, b, c, x, y, g))
		return 0;
	a /= g;
	b /= g;

	ll sign_a = a > 0 ? +1 : -1;
	ll sign_b = b > 0 ? +1 : -1;

	shift_solution(x, y, a, b, (minx - x) / b);
	if (x < minx)
		shift_solution(x, y, a, b, sign_b);
	if (x > maxx)
		return 0;
	ll lx1 = x;

	shift_solution(x, y, a, b, (maxx - x) / b);
	if (x > maxx)
		shift_solution(x, y, a, b, -sign_b);
	ll rx1 = x;

	shift_solution(x, y, a, b, -(miny - y) / a);
	if (y < miny)
		shift_solution(x, y, a, b, -sign_a);
	if (y > maxy)
		return 0;
	ll lx2 = x;

	shift_solution(x, y, a, b, -(maxy - y) / a);
	if (y > maxy)
		shift_solution(x, y, a, b, sign_a);
	ll rx2 = x;

	if (lx2 > rx2)
		swap(lx2, rx2);
	ll lx = max(lx1, lx2);
	ll rx = min(rx1, rx2);

	if (lx > rx)
		return 0;
	return (rx - lx) / abs(b) + 1;
}

    //lcm of n numbers
    ll findlcm(ll arr[], ll n)
    {
        ll ans = arr[0];
        for (ll i = 1; i < n; i++)
            ans = (((arr[i] * ans)) /
                   (gcd(arr[i], ans)));
        return ans;
    }

    //Suffix Array
    void radix_sort(v<pair<pair<ll, ll>, ll>> & a)
    {
        ll n = a.size();
        {
            v<ll> cnt(n, 0);
            for (auto x : a)
            {
                cnt[x.fi.se]++;
            }
            v<ll> pos(n);
            pos[0] = 0;
            repc(i, 1, n)
                pos[i] = pos[i - 1] + cnt[i - 1];
            v<pair<pair<ll, ll>, ll>> anew(n);
            for (auto x : a)
            {
                ll i = x.fi.se;
                anew[pos[i]] = x;
                pos[i]++;
            }
            a = anew;
        }
        {
            v<ll> cnt(n, 0);
            for (auto x : a)
            {
                cnt[x.fi.fi]++;
            }
            v<ll> pos(n);
            pos[0] = 0;
            repc(i, 1, n)
                pos[i] = pos[i - 1] + cnt[i - 1];
            v<pair<pair<ll, ll>, ll>> anew(n);
            for (auto x : a)
            {
                ll i = x.fi.fi;
                anew[pos[i]] = x;
                pos[i]++;
            }
            a = anew;
        }
    }
    int main()
    {
        string s;
        cin >> s;
        s += "$";
        n = s.size();
        v<ll> p(n), c(n);
        //k=0
        {
            v<pair<char, ll>> a(n);
            rep(i, n)
            {
                a[i] = {s[i], i};
            }
            sortv(a);
            rep(i, n) p[i] = a[i].se;
            c[p[0]] = 0;
            repc(i, 1, n)
            {
                if (a[i].fi != a[i - 1].fi)
                    c[p[i]] = c[p[i - 1]] + 1;
                else
                    c[p[i]] = c[p[i - 1]];
            }
        }
        //transition
        k = 0;
        while (pow(2, k) < n)
        {
            v<pair<pair<ll, ll>, ll>> a(n);
            rep(i, n)
            {
                a[i] = {{c[i], c[(i + (1 << k)) % n]}, i};
            }
            radix_sort(a);
            rep(i, n) p[i] = a[i].se;
            c[p[0]] = 0;
            repc(i, 1, n)
            {
                if (a[i].fi != a[i - 1].fi)
                    c[p[i]] = c[p[i - 1]] + 1;
                else
                    c[p[i]] = c[p[i - 1]];
            }
            k++;
        }
    }
    //SEGMENT TREE
    v<ll> tv, a;    //tv is the segment tree ans a is the main array of elements
                    //initialize tl ans tr with 0 ans n-1
    ll neutral = 0; //change
    ll combine(ll a, ll b)
    { //change
        return a + b;
    }
    void build(ll v, ll tl, ll tr)
    {
        if (tl == tr)
        {
            tv[v] = a[tl];
        }
        else
        {
            ll tm = (tl + tr) / 2;
            build(v * 2, tl, tm);
            build(v * 2 + 1, tm + 1, tr);
            tv[v] = combine(tv[v * 2], tv[v * 2 + 1]);
        }
    }

    ll sum(ll v, ll tl, ll tr, ll l, ll r)
    {
        if (l > r)
            return neutral;
        if (l == tl && r == tr)
        {
            return tv[v];
        }
        ll tm = (tl + tr) / 2;
        return combine(sum(v * 2, tl, tm, l, min(r, tm)), sum(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r));
    }

    void update(ll v, ll tl, ll tr, ll pos, ll new_val)
    {
        if (tl == tr)
        {
            tv[v] = new_val;
        }
        else
        {
            ll tm = (tl + tr) / 2;
            if (pos <= tm)
                update(v * 2, tl, tm, pos, new_val);
            else
                update(v * 2 + 1, tm + 1, tr, pos, new_val);
            tv[v] = combine(tv[v * 2], tv[v * 2 + 1]);
        }
    }
    //Finding the maximum and the number of times it appears(segment tree)
    v<pair<ll, ll>> tv;
    v<ll> a;

    pair<ll, ll> combine(pair<ll, ll> a, pair<ll, ll> b)
    {
        if (a.first < b.first)
            return a;
        if (b.first < a.first) //change
            return b;
        return {a.first, a.second + b.second};
    }

    void build(ll v, ll tl, ll tr)
    {
        if (tl == tr)
        {
            tv[v] = {a[tl], 1}; //change
        }
        else
        {
            ll tm = (tl + tr) / 2;
            build(v * 2, tl, tm);
            build(v * 2 + 1, tm + 1, tr);
            tv[v] = combine(tv[v * 2], tv[v * 2 + 1]);
        }
    }

    pair<ll, ll> get_min(ll v, ll tl, ll tr, ll l, ll r)
    {
        if (l > r)
            return {ll_max, 0};
        if (l == tl && r == tr)
            return tv[v];
        ll tm = (tl + tr) / 2;
        return combine(get_min(v * 2, tl, tm, l, min(r, tm)), get_min(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r));
    }

    void update(ll v, ll tl, ll tr, ll pos, ll new_val)
    {
        if (tl == tr)
        {
            tv[v] = {new_val, 1};
        }
        else
        {
            ll tm = (tl + tr) / 2;
            if (pos <= tm)
                update(v * 2, tl, tm, pos, new_val);
            else
                update(v * 2 + 1, tm + 1, tr, pos, new_val);
            tv[v] = combine(tv[v * 2], tv[v * 2 + 1]);
        }
    }
    //maximum sum segment(segment tree)
    v<ll> a;
    struct data
    { //change
        ll sum, pref, suff, ans;
    };
    v<data> tv;
    data combine(data l, data r)
    {
        data res;
        res.sum = l.sum + r.sum;
        res.pref = max(l.pref, l.sum + r.pref); //change
        res.suff = max(r.suff, r.sum + l.suff);
        res.ans = max(max(l.ans, r.ans), l.suff + r.pref);
        return res;
    }
    data make_data(int val)
    {
        data res;
        res.sum = val; //cahnge
        res.pref = res.suff = res.ans = max(0, val);
        return res;
    }
    void build(ll v, ll tl, ll tr)
    {
        if (tl == tr)
        {
            tv[v] = make_data(a[tl]); //change
        }
        else
        {
            ll tm = (tl + tr) / 2;
            build(v * 2, tl, tm);
            build(v * 2 + 1, tm + 1, tr);
            tv[v] = combine(tv[v * 2], tv[v * 2 + 1]);
        }
    }
    void update(ll v, ll tl, ll tr, ll pos, ll new_val)
    {
        if (tl == tr)
        {
            tv[v] = make_data(new_val);
        }
        else
        {
            ll tm = (tl + tr) / 2;
            if (pos <= tm)
                update(v * 2, tl, tm, pos, new_val);
            else
                update(v * 2 + 1, tm + 1, tr, pos, new_val);
            tv[v] = combine(tv[v * 2], tv[v * 2 + 1]);
        }
    }
    data query(ll v, ll tl, ll tr, ll l, ll r)
    {
        if (l > r)
            return make_data(0);
        if (l == tl && r == tr)
            return tv[v];
        ll tm = (tl + tr) / 2;
        return combine(query(v * 2, tl, tm, l, min(r, tm)),
                       query(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r));
    }
    //number of elements greater than equal to VALUE in a segment
    v<ll> a;
    v<v<ll>> tv;
    void build(ll v, ll tl, ll tr)
    {
        if (tl == tr)
            tv[v].pb(a[tl]);
        else
        {
            ll tm = (tl + tr) / 2;
            build(v * 2, tl, tm);
            build(v * 2 + 1, tm + 1, tr);
            merge(tv[v * 2].begin(), tv[2 * v].end(), tv[v * 2 + 1].begin(), tv[v * 2 + 1].end(), back_inserter(tv[v]));
        }
    }
    ll query(ll v, ll tl, ll tr, ll l, ll r, ll value)
    {
        if (r < tl || l > tr)
            return 0;
        if (tl >= l && tr <= r)
        {
            if (lb(tv[v], value) == tv[v].end())
                return 0;
            else
                return tv[v].size() - (lb(tv[v], value) - tv[v].begin());
        }
        ll tm = (tl + tr) / 2;
        return (query(2 * v, tl, tm, l, r, value) + query(2 * v + 1, tm + 1, tr, l, r, value));
    }
    //find kth one
    v<ll> tv;
    v<ll> a;
    void build(ll v, ll tl, ll tr)
    {
        if (tl == tr)
        {
            tv[v] = a[tl];
        }
        else
        {
            ll tm = (tl + tr) / 2;
            build(v * 2, tl, tm);
            build(v * 2 + 1, tm + 1, tr);
            tv[v] = tv[v * 2] + tv[v * 2 + 1]; //change
        }
    }
    void update(ll v, ll tl, ll tr, ll pos)
    {
        if (tl == tr)
        {
            tv[v] ^= 1;
        }
        else
        {
            ll tm = (tl + tr) / 2;
            if (pos <= tm)
                update(v * 2, tl, tm, pos);
            else
                update(v * 2 + 1, tm + 1, tr, pos);
            tv[v] = tv[v * 2] + tv[v * 2 + 1]; //change
        }
    }

    ll find(ll v, ll tl, ll tr, ll k)
    {
        if (k > tv[v])
            return -1;
        if (tl == tr)
            return tl;
        ll tm = (tl + tr) / 2; //everything remains same
        if (tv[v * 2] >= k)
            return find_kth(v * 2, tl, tm, k);
        else
            return find_kth(v * 2 + 1, tm + 1, tr, k - tv[v * 2]);
    }
    //smallest index whose value is >= x
    v<ll> tv, a;
    void build(ll v, ll tl, ll tr)
    {
        if (tl == tr)
        {
            tv[v] = a[tl];
        }
        else
        {
            ll tm = (tl + tr) / 2;
            build(v * 2, tl, tm);
            build(v * 2 + 1, tm + 1, tr);
            tv[v] = max(tv[v * 2], tv[v * 2 + 1]); //change
        }
    }
    void update(ll v, ll tl, ll tr, ll pos, ll new_val)
    {
        if (tl == tr)
        {
            tv[v] = new_val;
        }
        else
        {
            ll tm = (tl + tr) / 2;
            if (pos <= tm)
                update(v * 2, tl, tm, pos, new_val);
            else
                update(v * 2 + 1, tm + 1, tr, pos, new_val);
            tv[v] = max(tv[v * 2], tv[v * 2 + 1]); //change
        }
    }
    ll small(ll v, ll tl, ll tr, ll x)
    {
        if (x > tv[v])
            return -1;
        if (tl == tr)
            return tl;
        else
        {
            ll tm = (tl + tr) / 2;
            if (tv[v * 2] >= x)
                return small(v * 2, tl, tm, x);
            else
                return small(v * 2 + 1, tm + 1, tr, x);
        }
    }
    //smallest index whose value is >= x ans index>=l
    v<ll> tv, a;
    void build(ll v, ll tl, ll tr)
    {
        if (tl == tr)
        {
            tv[v] = a[tl];
        }
        else
        {
            ll tm = (tl + tr) / 2;
            build(v * 2, tl, tm);
            build(v * 2 + 1, tm + 1, tr);
            tv[v] = max(tv[v * 2], tv[v * 2 + 1]); //change
        }
    }
    void update(ll v, ll tl, ll tr, ll pos, ll new_val)
    {
        if (tl == tr)
        {
            tv[v] = new_val;
        }
        else
        {
            ll tm = (tl + tr) / 2;
            if (pos <= tm)
                update(v * 2, tl, tm, pos, new_val);
            else
                update(v * 2 + 1, tm + 1, tr, pos, new_val);
            tv[v] = max(tv[v * 2], tv[v * 2 + 1]); //change
        }
    }
    ll small(ll v, ll tl, ll tr, ll x, ll l)
    {
        if (x > tv[v] || tr < l)
        {
            return -1;
        }
        if (tl == tr)
            return tl;
        else
        {
            ll tm = (tl + tr) / 2;
            ll val1 = -1;
            if (tv[v * 2] >= x && tm >= l)
            {
                val1 = small(v * 2, tl, tm, x, l);
            }
            if (val1 != -1)
                return val1;
            return small(2 * v + 1, tm + 1, tr, x, l);
        }
    }
    //adding a value to a particular segment
    //range updates commutative
    void build(ll v, ll tl, ll tr)
    {
        if (tl == tr)
        {
            tv[v] = a[tl];
        }
        else
        {
            ll tm = (tl + tr) / 2;
            build(v * 2, tl, tm);
            build(v * 2 + 1, tm + 1, tr);
            tv[v] = 0;
        }
    }

    void update(ll v, ll tl, ll tr, ll l, ll r, ll add)
    {
        if (l > r)
            return;
        if (l == tl && r == tr)
        {
            tv[v] += add;
        }
        else
        {
            ll tm = (tl + tr) / 2;
            update(v * 2, tl, tm, l, min(r, tm), add);
            update(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r, add);
        }
    }

    ll get(ll v, ll tl, ll tr, ll pos)
    {
        if (tl == tr)
            return tv[v];
        ll tm = (tl + tr) / 2;
        if (pos <= tm)
            return tv[v] + get(v * 2, tl, tm, pos);
        else
            return tv[v] + get(v * 2 + 1, tm + 1, tr, pos);
    }
    //Range Updates non commutative(Lazy propagation)
    v<ll> tv;
    ll neutral = LLONG_MAX; //change
    ll combine(ll a, ll b)
    { //change
        if (b == neutral)
            return a;
        else
            return b;
    }
    void push(ll v)
    {
        tv[2 * v] = combine(tv[2 * v], tv[v]);
        tv[2 * v + 1] = combine(tv[2 * v + 1], tv[v]);
        tv[v] = neutral;
    }

    void update(ll v, ll tl, ll tr, ll l, ll r, ll new_val)
    {
        if (l > r)
            return;
        if (l == tl && tr == r)
        {
            tv[v] = combine(tv[v], new_val);
        }
        else
        {
            push(v);
            ll tm = (tl + tr) / 2;
            update(v * 2, tl, tm, l, min(r, tm), new_val);
            update(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r, new_val);
        }
    }
    ll get(ll v, ll tl, ll tr, ll pos)
    {
        if (tl == tr)
        {
            return tv[v];
        }
        push(v);
        ll tm = (tl + tr) / 2;
        if (pos <= tm)
            return get(v * 2, tl, tm, pos);
        else
            return get(v * 2 + 1, tm + 1, tr, pos);
    }
    //range update and range queries lazy updates
    v<ll> tv, lazy; //to initialize update each element
    ll neutral1 = 0;
    ll neutral2 = inf;
    ll combine1(ll a, ll b)
    { //for range update
        return a + b;
    }
    ll combine2(ll a, ll b)
    { //for range query
        return min(a, b);
    }

    void push(ll v)
    {
        tv[v * 2] = combine1(tv[v * 2], lazy[v]);
        lazy[v * 2] = combine1(lazy[v * 2], lazy[v]);
        tv[v * 2 + 1] = combine1(tv[v * 2 + 1], lazy[v]);
        lazy[v * 2 + 1] = combine1(lazy[2 * v + 1], lazy[v]);
        lazy[v] = neutral1;
    }

    void update(ll v, ll tl, ll tr, ll l, ll r, ll updaterange)
    {
        if (l > r)
            return;
        if (l == tl && tr == r)
        {
            tv[v] = combine1(tv[v], updaterange);
            lazy[v] = combine1(lazy[v], updaterange);
        }
        else
        {
            push(v);
            ll tm = (tl + tr) / 2;
            update(v * 2, tl, tm, l, min(r, tm), updaterange);
            update(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r, updaterange);
            tv[v] = combine2(tv[v * 2], tv[v * 2 + 1]);
        }
    }
    ll query(ll v, ll tl, ll tr, ll l, ll r)
    {
        if (l > r) 
            return neutral2;
        if (l <= tl && tr <= r)
            return tv[v];
        push(v);
        ll tm = (tl + tr) / 2;
        return combine2(query(v * 2, tl, tm, l, min(r, tm)), query(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r));
    }
    //addition and sum(query and update)
    v<ll> tv, lazy;
    ll neutral1 = 0;
    ll neutral2 = 0;
    ll combine1(ll a, ll b, ll ln)
    { //for range update
        return a + (b * ln);
    }
    ll combine2(ll a, ll b)
    { //for range query
        return a + b;
    }

    void push(ll v, ll l, ll r)
    {
        ll tm = (l + r) / 2;
        tv[v * 2] = combine1(tv[v * 2], lazy[v], (tm - l + 1));
        lazy[v * 2] = combine1(lazy[v * 2], lazy[v], 1);
        tv[v * 2 + 1] = combine1(tv[2 * v + 1], lazy[v], (r - tm));
        lazy[v * 2 + 1] = combine1(lazy[2 * v + 1], lazy[v], 1);
        lazy[v] = neutral1;
    }

    void update(ll v, ll tl, ll tr, ll l, ll r, ll updaterange)
    {
        if (l > r)
            return;
        if (l == tl && tr == r)
        {
            tv[v] = combine1(tv[v], updaterange, (r - l + 1));
            lazy[v] = combine1(lazy[v], updaterange, 1);
        }
        else
        {
            push(v, tl, tr);
            ll tm = (tl + tr) / 2;
            update(v * 2, tl, tm, l, min(r, tm), updaterange);
            update(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r, updaterange);
            tv[v] = combine2(tv[v * 2], tv[v * 2 + 1]);
        }
    }

    ll query(ll v, ll tl, ll tr, ll l, ll r)
    {
        if (l > r)
            return neutral2;
        if (l == tl && tr == r)
            return tv[v];
        push(v, tl, tr);
        ll tm = (tl + tr) / 2;
        return combine2(query(v * 2, tl, tm, l, min(r, tm)), query(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r));
    }

    //persistent segment tree
    v<ll> a;
    struct Vertex
    {
        Vertex *l, *r;
        ll sum;

        Vertex(ll val) : l(nullptr), r(nullptr), sum(val) {}
        Vertex(Vertex *l, Vertex *r) : l(l), r(r), sum(0)
        {
            if (l)
                sum += l->sum;
            if (r)
                sum += r->sum;
        }
    };

    Vertex *build(ll tl, ll tr)
    {
        if (tl == tr)
            return new Vertex(a[tl]);
        ll tm = (tl + tr) / 2;
        return new Vertex(build(tl, tm), build(tm + 1, tr));
    }

    ll get_sum(Vertex * v, ll tl, ll tr, ll l, ll r)
    {
        if (l > r)
            return 0;
        if (l == tl && tr == r)
            return v->sum;
        ll tm = (tl + tr) / 2;
        return get_sum(v->l, tl, tm, l, min(r, tm)) + get_sum(v->r, tm + 1, tr, max(l, tm + 1), r);
    }

    Vertex *update(Vertex * v, ll tl, ll tr, ll pos, ll new_val)
    {
        if (tl == tr)
            return new Vertex(new_val);
        ll tm = (tl + tr) / 2;
        if (pos <= tm)
            return new Vertex(update(v->l, tl, tm, pos, new_val), v->r);
        else
            return new Vertex(v->l, update(v->r, tm + 1, tr, pos, new_val));
    }
    main()
    {
        Vertex *root = build(0, n - 1);
        cout << get_sum(root, 0, n - 1, 1, n - 1);
    }
    //persistent segment tree range update point query(commutative)
struct Vertex
{
	Vertex *l, *r;
	ll sum;

	Vertex() : l(nullptr), r(nullptr), sum(0) {}
	Vertex(Vertex *l, Vertex *r): l(l), r(r), sum(0) {}
	Vertex(Vertex * l, Vertex * r, Vertex * same): l(l), r(r) , sum(same->sum) {}
	Vertex(Vertex *same, ll add): l(same->l), r(same->r), sum(same->sum + add) {}
};

Vertex *build(ll tl, ll tr)
{
	if (tl == tr)
		return new Vertex();
	ll tm = (tl + tr) / 2;
	return new Vertex(build(tl, tm), build(tm + 1, tr));
}
Vertex *update(Vertex *v, ll tl, ll tr, ll L, ll R, ll addend) {
	if (L > R) {
		return v;
	}
	if (L == tl && R == tl) {
		return new Vertex(v, addend);
	}
	ll tm = (tl + tr) / 2;
	return new Vertex(update(v->l, tl, tm, L, min(R, tm), addend),
	                  update(v->r, tm + 1, tr, max(L, tm + 1), R, addend), v);

}
ll get(Vertex *v, ll tl, ll tr, ll pos) {
	if (tl == tr) {
		return v->sum;
	}
	ll tm = (tl + tr) / 2;
	if (pos <= tm) {
		return v->sum + get(v->l, tl, tm, pos);
	}
	return v->sum + get(v->r, tm + 1, tr, pos);
}
    //dynamic segment tree
    struct Vertex
    {
        int left, right;
        int sum = 0;
        Vertex *left_child = nullptr, *right_child = nullptr;

        Vertex(int lb, int rb)
        {
            left = lb;
            right = rb;
        }

        void extend()
        {
            if (!left_child && left < right)
            {
                int t = (left + right) / 2;
                left_child = new Vertex(left, t);
                right_child = new Vertex(t + 1, right);
            }
        }

        void add(int k, int x)
        {
            extend();
            sum += x;
            if (left_child)
            {
                if (k <= left_child->right)
                    left_child->add(k, x);
                else
                    right_child->add(k, x);
            }
        }

        int get_sum(int lq, int rq)
        {
            if (lq <= left && right <= rq)
                return sum;
            if (max(left, lq) > min(right, rq))
                return 0;
            extend();
            return left_child->get_sum(lq, rq) + right_child->get_sum(lq, rq);
        }
    };
    main()
    {
        Vertex *root = new Vertex(0, n);
        root->add(0, 1);
    }
    //string hashing
    //simple hash
    ll compute_hash(string const &s)
    {
        const ll p = 31;
        const ll mod = 1e9 + 9;
        ll hash_value = 0;
        ll p_pow = 1;
        for (char c : s)
        {
            hash_value = (hash_value + (c - 'a' + 1) * p_pow) % mod;
            p_pow = (p_pow * p) % mod;
        }
        return hash_value;
    }
    //complex hash
    string s;
    v<ll> hsh;
    v<ll> p_pow;
    const ll p = 31;
    const ll mod = 1e9 + 7;
    void compute_hash()
    {
        ll i;
        rep(i, s.len)
        {
            if (i == 0)
                p_pow[i] = 1;
            else
                p_pow[i] = (p_pow[i - 1] * p) % mod;
        }
        hsh[0] = 0;
        rep(i, s.len)
        {
            hsh[i + 1] = (hsh[i] + (s[i] - 'a' + 1) * p_pow[i]) % mod;
        }
    }
    ll get_ij(ll i, ll j)
    { //call get_ij(i,j+1);
        ll out = (hsh[j] - hsh[i] + mod) % mod;
        out = (out * p_pow[s.len - i - 1]) % mod;
        return out;
    }
    int main()
    {
        hsh.resize(s.len + 1);
        p_pow.resize(s.len);
        compute_hash();
    }
    //fast hash
    string s;
    v<ll> hsh1, hsh2;
    v<ll> p_pow1;
    v<ll> p_pow2;
    const ll p1 = 31;
    const ll p2 = 13;
    const ll mod = 1e9 + 7;
    void fast_hash()
    {
        hsh1.clear();
        hsh2.clear();
        p_pow1.clear();
        p_pow2.clear();
        inv1.clear();
        inv2.clear();
        inv1.resize(s.len);
        inv2.resize(s.len);
        hsh1.resize(s.len + 1);
        hsh2.resize(s.len + 1);
        p_pow1.resize(s.len);
        p_pow2.resize(s.len);
        ll i;
        rep(i, s.len)
        {
            if (i == 0){
                p_pow1[i] = 1;
                inv1[i]=1;
                p_pow2[i] = 1;
                inv2[i]=1;
            }
            else{
                p_pow1[i] = (p_pow1[i - 1] * p1) % mod;
                inv1[i]=modinv(p_pow1[i]);
                p_pow2[i] = (p_pow2[i - 1] * p2) % mod;
                inv2[i]=modinv(p_pow2[i]);
            }
        }
        hsh1[0] = 0; hsh2[0]=0;
        rep(i, s.len)
        {
            hsh1[i + 1] = (hsh1[i] + (s[i] - 'a' + 1) * p_pow1[i]) % mod;
            hsh2[i + 1] = (hsh2[i] + (s[i] - 'a' + 1) * p_pow2[i]) % mod;
        }
    }
    pair<ll, ll> get_ij(ll i, ll j)
    { //call get_ij(i,j+1);
        ll out1 = (hsh1[j] - hsh1[i] + mod) % mod;
        out1 = (out1 * p_pow1[s.len - i - 1]) % mod;
        ll out2 = (hsh2[j] - hsh2[i] + mod) % mod;
        out2 = (out2 * p_pow2[s.len - i - 1]) % mod;
        return {out1, out2};
    }
 //2d hash
 ll get(ll x1, ll y1, ll x2, ll y2) {
	return ((h1[x1][y1] - h1[x1][y2 - 1] * pow1[y1-y2+1] - h1[x2 - 1][y1] * pow2[x1-x2+1] + ((h1[x2 - 1][y2 - 1] * pow1[y1-y2+1]) % M) * pow2[x1-x2+1]) % M + M) % M ;
}
        mset(h1, 0);
		mset(h2, 0);
		pow1[0] = 1;
		pow2[0] = 1;
		for (i = 1; i <= 1004; i++) {
			pow1[i] = (pow1[i - 1] * p1) % M ;
		}
		for (i = 1; i <= 1004; i++) pow2[i] = (pow2[i - 1] * p2) % M ;


		for (i = 1; i <= n; i++) {
			for (j = 1; j <= m; j++) {
				(h1[i][j] += h1[i][j - 1] * p1) %= M;
				(h1[i][j] += h1[i - 1][j] * p2) %= M;
				(h1[i][j] -= h1[i - 1][j - 1] * p1 * p2) %= M;
				(h1[i][j] += s[i - 1][j - 1] - 'a' + 1) % M;
				(h1[i][j] += M) %= M;
			}
		}
		for (i = 1; i <= x; i++) {
			for (j = 1; j <= y; j++) {
				(h2[i][j] += h2[i][j - 1] * p1) %= M;
				(h2[i][j] += h2[i - 1][j] * p2) %= M;
				(h2[i][j] -= h2[i - 1][j - 1] * p1 * p2) %= M;
				(h2[i][j] += T[i - 1][j - 1] - 'a' + 1) % M;
				(h2[i][j] += M) %= M;
			}
		}   

//aho corasick
//if space is limited use map init
const int K = 26;
struct Vertex {
	int next[K];//map<ll,ll> next;
	int leaf = 0;//change for indices
	int p = -1;
	char pch;
	int link = -1;
	int exit_link = -1;
	int go[K];//map<ll,ll> go;
	Vertex(int p = -1, char ch = '$') : p(p), pch(ch) {
		mset(go, -1);
		mset(next, -1);
	}
};

vector<Vertex> tv(1);//tv(2)

void add_string(string const& s,ll i) {//i is the index of string
	int v = 0;
	for (char ch : s) {
		int c = ch - 'a';//change
		if (tv[v].next[c] == -1) {//==0
			tv[v].next[c] = tv.size();
			tv.emplace_back(v, ch);
		}
		v = tv[v].next[c];
	}
	tv[v].leaf++;//change
}

int go(int v, char ch);

int get_link(int v) {
	if (tv[v].link == -1) {
		if (v == 0 || tv[v].p == 0)//==1
			tv[v].link = 0;//=1
		else
			tv[v].link = go(get_link(tv[v].p), tv[v].pch);
	}
	return tv[v].link;
}

int go(int v, char ch) {
	int c = ch - 'a';//change
	if (tv[v].go[c] == -1) {//==0
		if (tv[v].next[c] != -1)//!=0
			tv[v].go[c] = tv[v].next[c];
		else
			tv[v].go[c] = v == 0 ? 0 : go(get_link(v), ch);// 1 instead of 0
	}
	return tv[v].go[c];
}
int exit_link(int v)
{
	if (tv[v].exit_link == -1)
	{
		if (v == 0)//==1
			tv[v].exit_link = v;
		else if (tv[get_link(v)].leaf)//change
			tv[v].exit_link = get_link(v);
		else
			tv[v].exit_link = exit_link(get_link(v));
		// tv[v].exit_link=get_link(v);
	}
	return tv[v].exit_link;
}

ll query(string s) {//change
	int v = 0;//==1
	ll sum = 0;
	for (auto ch : s) {
		v = go(v, ch);
		ll temp = v;
		while (temp) {//temp!=1
			sum += tv[temp].leaf;//change
			temp = exit_link(temp);
		}
	}
	return sum;
}    
// Zfunction
vector<ll> z_function(string s) {
    ll n = (ll) s.length();
    vector<ll> z(n,0);
    for (ll i = 1, l = 0, r = 0; i < n; ++i) {
        if (i <= r)
            z[i] = min (r - i + 1, z[i - l]);
        while (i + z[i] < n && s[z[i]] == s[i + z[i]])
            ++z[i];
        if (i + z[i] - 1 > r)
            l = i, r = i + z[i] - 1;
    }
    return z;
}    
    // prefix function
    v<int> prefix(string s){
    int n = s.len;
    int i,j;
    vector<int> pre(n, 0);
    for (i = 1; i < n; i++)
    {
        j = pre[i - 1];
        while (j != 0 && s[i] != s[j])
            j = pre[j - 1];
        if (s[i] == s[j])
            j++;
        pre[i] = j;
    }
    return pre;
    }
//automation
v<ll> kmp(string a) {
	string chk = a;
	ll x = chk.len;
	vector<ll>pre(x, 0);
	for (ll i = 1; i < x; i++)
	{
		ll j = pre[i - 1];
		while (j != 0 && chk[i] != chk[j])
			j = pre[j - 1];
		if (chk[i] == chk[j])
			j++;
		pre[i] = j;
	}
	return pre;
}   
    s += "#";
	ll nex[s.len][26];
	v<ll> pre = kmp(s);
	for (int i = 0; i < s.len; i++) {
		for (int j = 0; j < 26; j++) {
			if (i > 0 && 'A' + j != s[i])
				nex[i][j] = nex[pre[i - 1]][j];
			else
				nex[i][j] = i + ('A' + j == s[i]);
		}
	}
    //TRIE
    int child[MAX_NUMBER_OF_NODES][MAX_ASCII_CODE], next = 1; //initially all numbers in child are 0
    void build(string s)
    {
        int i = 0, node = 0;
        while (i < s.size())
        {
            if (!child[node][s[i]])
                node = child[node][s[i++]] = next++;
            else
                node = child[node][s[i++]];
            //can apply any operation on this node ;
        }
    }
    //trie for binary strings
    const ll maxn = 4 * 1e6 + 100;
    ll child[maxn][2];
    ll cnt[maxn]; //change
    ll cntr = 1;
    void add(ll x)
    {
        ll node = 0;
        for (ll i = 29; i >= 0; i--)
        {
            ll bit = (x >> i) & 1;
            if (!child[node][bit])
                child[node][bit] = cntr++;
            node = child[node][bit];
            cnt[node]++; //operation can change
        }
    }
    void query(ll node, ll bitpos)
    {
        ll l = child[node][0], r = child[node][1];
        if (l)
            query(l, bitpos - 1);
        if (r)
            query(r, bitpos - 1);
        if(l||r){

        }
        else{
            
        }
    }
//trie for binary strings(struct)    
    typedef struct trie
{
	typedef struct node
	{
		node* nxt[2];
		int cnt = 0;

		node()
		{
			nxt[0] = nxt[1] = NULL;
			cnt = 0;
		}

	} Node;

	Node* head;

	trie() { head = new Node(); }

	void init() { head = new Node(); }

	void add(int x)
	{
		Node* cur = head;
		for (int i = 30; i >= 0; i--)
		{
			int b = (x >> i) & 1;
			if (!cur -> nxt[b])
				cur -> nxt[b] = new Node();
			cur = cur -> nxt[b];
			cur -> cnt++;
		}
	}

	void del(int x) {
		Node* cur = head;
		for (int i = 30; i >= 0; i--)
		{
			int b = (x >> i) & 1;
			cur = cur -> nxt[b];
			cur -> cnt--;
		}
	}

	int query(int x, int k)
	{
		//change
		return something;
	}

} Trie;
Trie tv;
    //SPARSE TABLE
    //range sum queries
    ll K = log2(n)+1;
    ll st[n][K + 1];

    for (ll i = 0; i < n; i++)
        st[i][0] = a[i];

    for (ll j = 1; j <= K; j++)
        for (ll i = 0; i + (1 << j) <= n; i++)
            st[i][j] = st[i][j - 1] + st[i + (1 << (j - 1))][j - 1];

    ll sum = 0;
    for (ll j = K; j >= 0; j--)
    {
        if ((1 << j) <= R - L + 1)
        {
            sum += st[L][j];
            L += 1 << j;
        }
    }
    //range minimum queries
    ll K = log2(n)+1;
    ll log[n + 1];
    log[1] = 0;
    for (ll i = 2; i <= n; i++)
        log[i] = log[i / 2] + 1;

    ll st[n][K + 1];

    for (ll i = 0; i < n; i++)
        st[i][0] = a[i];

    for (ll j = 1; j <= K; j++)
        for (ll i = 0; i + (1 << j) <= n; i++)
            st[i][j] = min(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);

    ll j = log[R - L + 1];
    ll minimum = min(st[L][j], st[R - (1 << j) + 1][j]);
    //bitset
    template <size_t sz>
    struct bitset_comparer
    {
        bool operator()(const bitset<sz> &b1, const bitset<sz> &b2) const
        {
            return b1.to_ulong() < b2.to_ulong();
        }
    };
    //unordered map
    struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
    //custom sort for multiset,set
    multiset<pair<ll, ll>, std::function<bool(const pair<ll, ll> &, const pair<ll, ll> &)>> mx(c1); //c1 is a function
    //random shuffle array
    shuffle(a.begin(),a.end(),rng);
    //different numbers in a segment (binary indexed tree)
    const int MAX = 1000001;

    struct Query
    {
        ll l, r, idx;
    };

    // cmp function to sort queries according to r
    bool cmp(Query x, Query y)
    {
        return x.r < y.r;
    }

    // updating the bit array
    void update(ll idx, ll val, ll bit[], ll n)
    {
        for (; idx <= n; idx += idx & -idx)
            bit[idx] += val;
    }

    // querying the bit array
    ll query(ll idx, ll bit[], ll n)
    {
        ll sum = 0;
        for (; idx > 0; idx -= idx & -idx)
            sum += bit[idx];
        return sum;
    }

    void answeringQueries(ll arr[], ll n, Query queries[], ll q)
    {
        // initialising bit array
        ll bit[n + 1];
        memset(bit, 0, sizeof(bit));

        // holds the rightmost index of any number
        // as numbers of a[i] are less than or equal to 10^6
        ll last_visit[MAX];
        memset(last_visit, -1, sizeof(last_visit));

        // answer for each query
        ll ans[q];
        ll query_counter = 0;
        for (ll i = 0; i < n; i++)
        {
            // If last visit is not -1 update -1 at the
            // idx equal to last_visit[arr[i]]
            if (last_visit[arr[i]] != -1)
                update(last_visit[arr[i]] + 1, -1, bit, n);

            // Setting last_visit[arr[i]] as i and updating
            // the bit array accordingly
            last_visit[arr[i]] = i;
            update(i + 1, 1, bit, n);

            // If i is equal to r of any query  store answer
            // for that query in ans[]
            while (query_counter < q && queries[query_counter].r == i)
            {
                ans[queries[query_counter].idx] =
                    query(queries[query_counter].r + 1, bit, n) -
                    query(queries[query_counter].l, bit, n);
                query_counter++;
            }
        }

        // print answer for each query
        for (ll i = 0; i < q; i++)
            cout << ans[i] << endl;
    }

    int main()
    {
        // your code goes here
        IOS;

#ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
#endif
        ll i, j, t, k, x, y, z, m, N;
        cin >> n >> m;
        ll a[n];
        rep(i, n)
                cin >>
            a[i];
        Query queries[m];
        rep(i, m)
        {
            cin >> x >> y;
            x--, y--;
            queries[i].l = x;
            queries[i].r = y;
            queries[i].idx = i;
        }
        ll q = sizeof(queries) / sizeof(queries[0]);
        sort(queries, queries + q, cmp);
        answeringQueries(a, n, queries, q);
        return 0;
    }
    //DSU
    struct DSU
    {
        v<ll> parent, size;
        DSU(ll n)
        {
            size.resize(n);
            parent.resize(n);
            for (ll i = 0; i < n; i++)
            {
                make_set(i);
            }
        }
        ll find_set(ll v)
        {
            if (v == parent[v])
            {
                return v;
            }
            return parent[v] = find_set(parent[v]);
        }
        void make_set(ll v)
        { //by size
            parent[v] = v;
            size[v] = 1;
        }
        void union_sets(ll a, ll b)
        {
            a = find_set(a);
            b = find_set(b);
            if (a != b)
            {
                if (size[a] < size[b])
                    swap(a, b);
                parent[b] = a;
                size[a] += size[b];
            }
        }
    };
    v<ll> parent, rank;
    ll find_set(ll v)
    {
        if (v == parent[v])
            return v;
        return parent[v] = find_set(parent[v]);
    }
    void make_set(int v)
    { //by rank
        parent[v] = v;
        rank[v] = 0;
    }
    void union_sets(int a, int b)
    {
        a = find_set(a);
        b = find_set(b);
        if (a != b)
        {
            if (rank[a] < rank[b])
                swap(a, b);
            parent[b] = a;
            if (rank[a] == rank[b])
                rank[a]++;
        }
    }

 //venice technique
struct VeniceSet {
	multiset<ll> S;
	ll water_level = 0;
	void add(ll v) {
		S.insert(v + water_level);
	}
	void remove(ll v) {
		S.erase(S.find(v + water_level));
	}
	void updateAll(ll v) {
		water_level += v;
	}
	ll getMin() {
		if (S.size())
			return *S.begin() - water_level;
		else return inf;
	}
	ll size() {
		return S.size();
	}
};
 //minimum spanning tree
 struct DSU
{
	v<ll> parent, size;
	DSU(ll n)
	{
		size.resize(n);
		parent.resize(n);
		for (ll i = 0; i < n; i++)
		{
			make_set(i);
		}
	}
	ll find_set(ll v)
	{
		if (v == parent[v])
		{
			return v;
		}
		return parent[v] = find_set(parent[v]);
	}
	void make_set(ll v)
	{	//by size
		parent[v] = v;
		size[v] = 1;
	}
	void union_sets(ll a, ll b)
	{
		a = find_set(a);
		b = find_set(b);
		if (a != b)
		{
			if (size[a] < size[b])
				swap(a, b);
			parent[b] = a;
			size[a] += size[b];
		}
	}
};
struct Edge {
    ll u, v, weight;
    bool operator<(Edge const& other) {
        return weight < other.weight;
    }
};
vector<Edge> edges;   
main(){
    DSU dsu(n);
    sort(edges.begin(), edges.end());
    for (Edge e : edges) {
        if (dsu.find_set(e.u) != dsu.find_set(e.v)) {
            cost += e.weight;
            result.push_back(e);
            dsu.union_sets(e.u, e.v);
        }
    }
}
    //ternary search
    ll l = -1e9, r = 1e9, m1, m2; //if it is decreasing and then increasing
    ll cost = inf;
    while (l <= r)
    {
        m1 = l + (r - l) / 3;
        m2 = r - (r - l) / 3;
        ll dum1 = func(m1);
        ll dum2 = func(m2);
        cost = min(cost, min(dum1, dum2));
        if (dum1 < dum2)
            r = m2 - 1;
        else
            l = m1 + 1;
    }
//fft
using cd = complex<double>;

void fft(vector<cd> & a, bool invert) {
	int n = a.size();

	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1)
			j ^= bit;
		j ^= bit;

		if (i < j)
			swap(a[i], a[j]);
	}

	for (int ln = 2; ln <= n; ln <<= 1) {
		double ang = 2 * PI / ln * (invert ? -1 : 1);
		cd wlen(cos(ang), sin(ang));
		for (int i = 0; i < n; i += ln) {
			cd w(1);
			for (int j = 0; j < ln / 2; j++) {
				cd u = a[i + j], v = a[i + j + ln / 2] * w;
				a[i + j] = u + v;
				a[i + j + ln / 2] = u - v;
				w *= wlen;
			}
		}
	}

	if (invert) {
		for (cd & x : a)
			x /= n;
	}
}
vector<int> multiply(vector<int> const& a, vector<int> const& b) {
	vector<cd> fa(a.begin(), a.end()), fb(b.begin(), b.end());
	int n = 1;
	while (n < a.size() + b.size())
		n <<= 1;
	fa.resize(n);
	fb.resize(n);

	fft(fa, false);
	fft(fb, false);
	for (int i = 0; i < n; i++)
		fa[i] *= fb[i];
	fft(fa, true);

	vector<int> result(n);
	for (int i = 0; i < n; i++)
		result[i] = round(fa[i].real());
	return result;
}    
//2d fft
using cd = complex<double>;

void fft(vector<cd> & a, bool invert) {
	int n = a.size();

	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1)
			j ^= bit;
		j ^= bit;

		if (i < j)
			swap(a[i], a[j]);
	}

	for (int ln = 2; ln <= n; ln <<= 1) {
		double ang = 2 * PI / ln * (invert ? -1 : 1);
		cd wlen(cos(ang), sin(ang));
		for (int i = 0; i < n; i += ln) {
			cd w(1);
			for (int j = 0; j < ln / 2; j++) {
				cd u = a[i + j], v = a[i + j + ln / 2] * w;
				a[i + j] = u + v;
				a[i + j + ln / 2] = u - v;
				w *= wlen;
			}
		}
	}

	if (invert) {
		for (cd & x : a)
			x /= n;
	}
}
void fft2(vector<vector<cd>> &a, bool invert) {//2d fft
	for (auto &x : a) fft(x, invert);
	for (int j = 0; j < a[0].size(); j++) {
		vector<cd> res;
		for (int i = 0; i < a.size(); i++) res.pb(a[i][j]);
		fft(res, invert);
		for (int i = 0; i < a.size(); i++) a[i][j] = res[i];
	}
}
vector<vector<cd>> multiply(vector<vector<cd>> &a, vector<vector<cd>> &b) {
	int n = 1;
	while (n < a.size() + b.size()) n <<= 1;
	int m = 1;
	while (m < a.front().size() + b.front().size()) m <<= 1;

	a.resize(n);
	b.resize(n);
	for (auto &x : a) x.resize(m);
	for (auto &x : b) x.resize(m);

	fft2(a, false);
	fft2(b, false);
	v<v<cd>> result(n, v<cd>(m));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			result[i][j] = a[i][j] * b[i][j];
		}
	}
	fft2(result, true);
	return result;
}
//convex hull 
typedef ll ftype;
typedef complex<ftype> point;
#define x real
#define y imag

ftype dot(point a, point b) {
	return (conj(a) * b).x();
}

ftype f(point a,  ftype x) {
	return dot(a, {x, 1});
}

struct Vertex {
	ll left, right;
	point line = point(0, inf);
	Vertex *left_child = nullptr, *right_child = nullptr;

	Vertex(ll lb, ll rb) {
		left = lb;
		right = rb;
	}

	void extend() {
		if (!left_child && left + 1 < right) {
			ll t = (left + right) / 2;
			left_child = new Vertex(left, t);
			right_child = new Vertex(t, right);
		}
	}
	void add_line(point nw) {
		extend();
		ll m = (left + right) / 2;
		bool lef = f(nw, left) < f(line, left);
		bool mid = f(nw, m) < f(line, m);
		if (mid) {
			swap(line, nw);
		}
		if (right - left == 1) return;
		else if (lef != mid) left_child->add_line(nw);
		else right_child->add_line(nw);
	}
	ll get(ll x) {
		ll m = (left + right) / 2;
		if (right - left == 1) return f(line, x);
		extend();
		if (x < m) {
			return min(f(line, x), left_child->get(x));
		}
		else return min(f(line, x), right_child->get(x));
	}
};
//sos dp
//iterative version
for(int mask = 0; mask < (1<<N); ++mask){
	dp[mask][-1] = A[mask];	//handle base case separately (leaf states)
	for(int i = 0;i < N; ++i){
		if(mask & (1<<i))
			dp[mask][i] = dp[mask][i-1] + dp[mask^(1<<i)][i-1];
		else
			dp[mask][i] = dp[mask][i-1];
	}
	F[mask] = dp[mask][N-1];
}
//memory optimized, super easy to code.
for(int i = 0; i<(1<<N); ++i)
	F[i] = A[i];
for(int i = 0;i < N; ++i) for(int mask = 0; mask < (1<<N); ++mask){
	if(mask & (1<<i))
		F[mask] += F[mask^(1<<i)];
}

//minimum distance of a point from a line segment
#define Point pair<db,db>
db minDistance(Point A, Point B, Point E)
{

	// vector AB
	pair<db, db> AB;
	AB.fi = B.fi - A.fi;
	AB.se = B.se - A.se;

	// vector BP
	pair<db, db> BE;
	BE.fi = E.fi - B.fi;
	BE.se = E.se - B.se;

	// vector AP
	pair<db, db> AE;
	AE.fi = E.fi - A.fi,
	   AE.se = E.se - A.se;

	// Variables to store dot product
	db AB_BE, AB_AE;

	// Calculating the dot product
	AB_BE = (AB.fi * BE.fi + AB.se * BE.se);
	AB_AE = (AB.fi * AE.fi + AB.se * AE.se);

	// Minimum distance from
	// point E to the line segment
	db reqAns = 0;

	// Case 1
	if (AB_BE > 0) {

		// Finding the magnitude
		db y = E.se - B.se;
		db x = E.fi - B.fi;
		reqAns = sqrt(x * x + y * y);
	}

	// Case 2
	else if (AB_AE < 0) {
		db y = E.se - A.se;
		db x = E.fi - A.fi;
		reqAns = sqrt(x * x + y * y);
	}

	// Case 3
	else {

		// Finding the perpendicular distance
		db x1 = AB.fi;
		db y1 = AB.se;
		db x2 = AE.fi;
		db y2 = AE.se;
		db mod = sqrt(x1 * x1 + y1 * y1);
		reqAns = abs(x1 * y2 - y1 * x2) / mod;
	}
	return reqAns;
}

//Eulerian path
	v<multiset<pll>> adj(n);
	rep(i, m) {
		cin >> x >> y;
		x--; y--;
		adj[x].insert({0, y});
		adj[y].insert({0, x});
	}
	ll v1 = -1, v2 = -1;
	for (int i = 0; i < n; ++i) {
		if ((int)adj[i].size() & 1) {
			if (v1 == -1)
				v1 = i;
			else if (v2 == -1)
				v2 = i;
			else
			{
				cout << "IMPOSSIBLE";
				return 0;
			}
		}
	}
	ll first = 0;
	while (first < n && adj[first].size() == 0) {
		first++;
	}
	if (first == n) {
		cout << "IMPOSSIBLE";
		return 0;
	}
	if (v1 != -1) {
		adj[v1].insert({0, v2});
		adj[v2].insert({0, v1});
	}
	stack<ll> stk;
	stk.push(first);
	v<ll> res;
	while (stk.size()) {
		ll v = stk.top();
		auto itr = *adj[v].begin();
		if (itr.fi) {
			res.pb(v);
			stk.pop();
		}
		else {
			adj[v].erase(adj[v].find({0, itr.se}));
			adj[v].insert({1, itr.se});
			adj[itr.se].erase(adj[itr.se].find({0, v}));
			adj[itr.se].insert({1, v});
			stk.push(itr.se);
		}
	}

	rep(i, n) {
		if (adj[i].size()) {
			auto itr = *adj[i].begin();
			if (itr.fi == 0) {
				cout << "IMPOSSIBLE";
				return 0;
			}
		}
	}
	if (v1 != -1) {
		for (size_t i = 0; i + 1 < res.size(); ++i) {
			if ((res[i] == v1 && res[i + 1] == v2) ||
			        (res[i] == v2 && res[i + 1] == v1)) {
				vector<ll> res2;
				for (size_t j = i + 1; j < res.size(); ++j)
					res2.push_back(res[j]);
				for (size_t j = 1; j <= i; ++j)
					res2.push_back(res[j]);
				res = res2;
				break;
			}
		}
	}

//manacher algorithm
vector<int> d1(n);
for (int i = 0, l = 0, r = -1; i < n; i++) {
    int k = (i > r) ? 1 : min(d1[l + r - i], r - i + 1);
    while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) {
        k++;
    }
    d1[i] = k--;
    if (i + k > r) {
        l = i - k;
        r = i + k;
    }
}
vector<int> d2(n);
for (int i = 0, l = 0, r = -1; i < n; i++) {
    int k = (i > r) ? 0 : min(d2[l + r - i + 1], r - i + 1);
    while (0 <= i - k - 1 && i + k < n && s[i - k - 1] == s[i + k]) {
        k++;
    }
    d2[i] = k--;
    if (i + k > r) {
        l = i - k - 1;
        r = i + k ;
    }
}
//heap sort
//max heap
void heapify(vector<int> &vec, int pos, int n) {
	int large = pos;
	int left = 2 * pos + 1;
	int right = left + 1;
	if (left < n && vec[left] > vec[large]) {
		large = left;
	}
	if (right < n && vec[right] > vec[large]) {
		large = right;
	}
	if (large != pos) {
		swap(vec[large], vec[pos]);
		heapify(vec, large, n);
	}
}
void heapsort(vector<int> &vec) {
	for (int i = vec.size() - 1; i >= 0; i--) {
		heapify(vec, i, vec.size());
	}
	for (int i = vec.size() - 1; i >= 0; i--) {
		swap(vec[0], vec[i]);
		heapify(vec, 0, i);
	}
}

 
//Function to merge the arrays.
void merge(int arr1[], int arr2[], int n, int m)
{ 
    //code here
    
    for(int i=m/2-1;i>=0;i--) heapify(arr2,i,m);
    for(int i=0;i<n;i++){
        if(arr1[i]<arr2[0]){
            continue;
        }
        else{
            swap(arr1[i],arr2[0]);
            heapify(arr2,0,m);
        }
    }
    heap_sort(arr2,m);
    
    
}
//AVL Tree insertion
int height(Node *root){
        if(!root) return 0;
        return root->height;
    }
    int bal(Node *root){
        if(!root) return 0;
        return height(root->left)-height(root->right);
    }
    Node *newnode(int key){
        Node *hey=new Node(key);
        // hey->data=key;
        // hey->left=NULL;
        // hey->right=NULL;
        // hey->height=1;
        return hey;
    }
    Node *rightrotate(Node *root){
        Node *temp=root->left;
        Node *y=root->left->right;
       
        temp->right=root;
        root->left=y;
        
        root->height=max(height(root->right),height(root->left))+1;
        temp->height=max(height(temp->right),height(temp->left))+1;
        
        return temp;
    }
    Node *leftrotate(Node *root){
        Node *y=root->right->left;
        Node *temp=root->right;
        temp->left=root;
        root->right=y;
        root->height=max(height(root->left),height(root->right))+1;
        temp->height=max(height(temp->left),height(temp->right))+1;
        return temp;
    }
    
    /*You are required to complete this method */
    Node* insertToAVL(Node* root, int key)
    {
        //Your code here
        if(!root) return newnode(key);
        if(key<root->data) root->left=insertToAVL(root->left,key);
        else if(key>root->data) root->right=insertToAVL(root->right,key);
        else return root;
        
        root->height=max(height(root->left),height(root->right))+1;
        
        if(bal(root)<-1){
            if(key>root->right->data) return leftrotate(root);
            else{
                root->right=rightrotate(root->right);
                return leftrotate(root);
            }
        }
        if(bal(root)>1){
            if(key<root->left->data) return rightrotate(root);
            else{
               root->left= leftrotate(root->left);
               return rightrotate(root);
            }
        }
        return root;
        
    }