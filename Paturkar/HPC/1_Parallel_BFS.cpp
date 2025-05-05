#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

class TreeGraph {
    int numNodes;
    vector<vector<int>> adjacencyList;

public:
    TreeGraph(int nodes) {
        numNodes = nodes;
        adjacencyList.resize(nodes);
    }

    void connect(int from, int to) {
        adjacencyList[from].push_back(to);
        adjacencyList[to].push_back(from); // Undirected edge
    }

    void parallelBFS(int startNode) {
        vector<bool> isVisited(numNodes, false);
        queue<int> nodeQueue;

        isVisited[startNode] = true;
        nodeQueue.push(startNode);

        while (!nodeQueue.empty()) {
            int current = nodeQueue.front();
            nodeQueue.pop();

            cout << current << " ";

            // Traverse neighbors in parallel
            #pragma omp parallel for
            for (int i = 0; i < adjacencyList[current].size(); ++i) {
                int neighbor = adjacencyList[current][i];

                if (!isVisited[neighbor]) {
                    #pragma omp critical
                    {
                        if (!isVisited[neighbor]) {
                            isVisited[neighbor] = true;
                            nodeQueue.push(neighbor);
                        }
                    }
                }
            }
        }
        cout << endl;
    }
};

int main() {
    // Demo 1
    cout << "Tree Example 1:\n";
    TreeGraph t1(6);
    t1.connect(0, 1);
    t1.connect(0, 2);
    t1.connect(1, 3);
    t1.connect(1, 4);
    t1.connect(2, 5);
    cout << "BFS from node 0: ";
    t1.parallelBFS(0);

    cout << "\n";

    // Demo 2
    cout << "Tree Example 2:\n";
    TreeGraph t2(7);
    t2.connect(0, 1);
    t2.connect(0, 2);
    t2.connect(1, 3);
    t2.connect(2, 4);
    t2.connect(3, 5);
    t2.connect(4, 6);
    cout << "BFS from node 0: ";
    t2.parallelBFS(0);

    cout << "\n";

    // Custom input
    int vertices, edges;
    cout << "Enter total vertices: ";
    cin >> vertices;

    TreeGraph userTree(vertices);

    cout << "Enter number of edges: ";
    cin >> edges;

    cout << "Enter each edge (u v):" << endl;
    for (int i = 0; i < edges; ++i) {
        int u, v;
        cin >> u >> v;
        userTree.connect(u, v);
    }

    cout << "Parallel BFS result:\n";
    userTree.parallelBFS(0);

    return 0;
}
