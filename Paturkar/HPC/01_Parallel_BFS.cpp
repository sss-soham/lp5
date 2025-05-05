/*
 * Problem Statement:
 * Design and implement Parallel Breadth First Search based on existing algorithms using OpenMP.
 * Use a Tree or an undirected graph for BFS.
 *
 * How to run:
 * 1. Open terminal in the directory containing the file
 * 2. Compile: g++ -fopenmp 01_Parallel_BFS.cpp -o 01_Parallel_BFS
 *    (if above not worked): g++ 01_Parallel_BFS.cpp -o 01_Parallel_BFS
 *    (General command): g++ -fopenmp fileName.cpp -o fileName or g++ fileName.cpp -o fileName
 * 3. Run: ./01_Parallel_BFS or .\01_Parallel_BFS
 */

#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

struct Graph
{
    int V;
    vector<vector<int>> adj;

    Graph(int V)
    {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected graph
    }

    void BFS(int start)
    {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            cout << u << " ";

#pragma omp parallel for
            for (int i = 0; i < adj[u].size(); i++)
            {
                int v = adj[u][i];
                if (!visited[v])
                {
#pragma omp critical
                    {
                        if (!visited[v])
                        { // Double check inside critical
                            visited[v] = true;
                            q.push(v);
                        }
                    }
                }
            }
        }
        cout << endl;
    }
};

int main()
{
    // Example 1: Small graph for basic understanding
    cout << "Example 1: Small Graph (5 vertices)\n";
    Graph g1(5);
    g1.addEdge(0, 1);
    g1.addEdge(0, 2);
    g1.addEdge(1, 3);
    g1.addEdge(2, 4);
    cout << "BFS traversal starting from vertex 0: ";
    g1.BFS(0);
    cout << "\n";

    // Example 2: Slightly larger graph
    cout << "Example 2: Larger Graph (7 vertices)\n";
    Graph g2(7);
    g2.addEdge(0, 1);
    g2.addEdge(0, 2);
    g2.addEdge(1, 3);
    g2.addEdge(2, 4);
    g2.addEdge(3, 5);
    g2.addEdge(4, 6);
    cout << "BFS traversal starting from vertex 0: ";
    g2.BFS(0);

    cout << "\n";

    // User input for a graph
    int V;
    cout << "Enter the number of vertices: ";
    cin >> V;

    Graph g(V);

    int edges;
    cout << "Enter the number of edges: ";
    cin >> edges;

    cout << "Enter edges: (u, v)" << endl;
    for (int i = 0; i < edges; i++)
    {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    cout << "Parallel BFS traversal:" << endl;
    g.BFS(0);

    return 0;
}

/*
 * Parallel Breadth First Search (BFS) Implementation using OpenMP
 * =============================================================
 *
 * Overview:
 * ---------
 * This program implements a parallel version of the Breadth-First Search algorithm
 * for traversing graphs. It demonstrates both basic graph operations and parallel
 * processing concepts using OpenMP.
 *
 * Key Technologies:
 * ----------------
 * 1. OpenMP (#pragma omp)
 *    - A parallel programming API for shared-memory multiprocessing
 *    - Used here for parallel task execution and thread management
 *    - Examples in other contexts: parallel for loops, matrix multiplication
 *    - Directives used:
 *      #pragma omp parallel      // Creates parallel region with multiple threads
 *      #pragma omp single nowait // Single thread executes, others don't wait
 *      #pragma omp task          // Creates new task for parallel execution
 *      #pragma omp parallel for  // Distributes loop iterations among threads
 *
 * 2. STL Containers
 *    - vector<>: Dynamic arrays for adjacency lists
 *    - queue<>: FIFO data structure for BFS traversal
 *
 * Data Structures:
 * ---------------
 * 1. Graph (struct)
 *    - Adjacency list representation using vector<vector<int>>
 *    - Allows O(1) access to vertices and O(degree) traversal of neighbors
 *
 * 2. BFS-specific structures:
 *    - visited[]: Boolean array tracking visited vertices
 *    - queue: Maintains vertices to be explored
 *
 * Complexity Analysis:
 * -------------------
 * Time Complexity:
 * - Sequential: O(V + E) where V = vertices, E = edges
 * - Parallel: O((V + E)/p) theoretical, where p = number of processors
 *
 * Space Complexity:
 * - O(V + E) for adjacency list
 * - O(V) for visited array and queue
 *
 * Parallel Performance Factors:
 * ---------------------------
 * 1. Critical Section Overhead
 * 2. Load Balancing
 * 3. Graph Structure/Density
 * 4. Memory Access Patterns
 * 5. Thread Synchronization Cost
 *
 * Q&A Section:
 * -----------
 * Q1: What is BFS and why use it?
 * A1: BFS is a graph traversal algorithm that explores all vertices at the current
 *     depth before moving to vertices at the next depth level. Used in shortest
 *     path finding, web crawling, and network analysis.
 *
 * Q2: Why use adjacency lists over matrices?
 * A2: Adjacency lists are more space-efficient for sparse graphs O(V+E) vs O(V²),
 *     and provide faster iteration over neighbors.
 *
 * Q3: What does #pragma omp parallel for do?
 * A3: It distributes loop iterations across multiple threads, enabling parallel
 *     processing of independent operations.
 *
 * Q4: Why is the critical section necessary?
 * A4: To prevent race conditions when multiple threads try to update shared data
 *     (visited array and queue) simultaneously.
 *
 * Q5: What's the purpose of double-checking visited[v]?
 * A5: It's a performance optimization that reduces critical section contention
 *     by checking visited status before entering the critical section.
 *
 * Q6: How does parallel BFS differ from sequential BFS?
 * A6: Parallel BFS processes multiple neighbors concurrently, but requires
 *     synchronization for shared data access.
 *
 * Q7: What factors affect parallel performance?
 * A7: Graph structure, thread overhead, memory access patterns, load balancing,
 *     and critical section contention.
 *
 * Q8: Why use vector<bool> for visited array?
 * A8: It's space-efficient as it typically uses 1 bit per boolean instead of
 *     1 byte, though it may have performance tradeoffs.
 *
 * Q9: How does load balancing affect performance?
 * A9: Uneven distribution of edges among vertices can lead to some threads
 *     doing more work than others, reducing parallel efficiency.
 *
 * Q10: What's the impact of graph density on performance?
 * A10: Denser graphs have more edges to process but may suffer from increased
 *      critical section contention.
 *
 * Q11: How can we optimize the parallel implementation?
 * A11: Use larger grain sizes, minimize critical sections, employ better
 *      load balancing, and consider graph partitioning.
 *
 * Q12: What's the significance of the queue in BFS?
 * A12: The queue maintains the FIFO order necessary for level-by-level
 *      traversal characteristic of BFS.
 *
 * Q13: How does thread count affect performance?
 * A13: More threads can improve performance up to a point, after which
 *      overhead and contention may degrade performance.
 *
 * Q14: Why is the graph undirected?
 * A14: Each edge is bidirectional (u->v and v->u), suitable for applications
 *      like social networks or road systems.
 *
 * Q15: What are the memory access patterns in this implementation?
 * A15: Random access patterns when accessing adjacency lists and visited array,
 *      which can affect cache performance.
 *
 * Q16: What are the limitations of this implementation?
 * A16: - No handling of disconnected components
 *      - Potential overhead for small graphs
 *      - Memory contention in dense graphs
 *
 * Q17: What is the purpose of #pragma omp critical?
 * A17: Ensures mutual exclusion, allowing only one thread to execute a code block at a time,
 *      preventing race conditions in shared resource access.
 *
 * Q18: How does #pragma omp taskwait work in OpenMP?
 * A18: Forces the current thread to wait until all its child tasks complete,
 *      useful for synchronization points in the algorithm.
 *
 * Q19: What is the role of OMP_NUM_THREADS environment variable?
 * A19: Sets the default number of threads for parallel regions in OpenMP programs,
 *      allowing runtime control of parallelism level.
 *
 * Q20: How does #pragma omp atomic differ from critical?
 * A20: Provides faster, hardware-supported atomic operations for simple updates to shared variables,
 *      more efficient than critical sections for single operations.
 *
 * Q21: What is the purpose of #pragma omp barrier?
 * A21: Creates a synchronization point where all threads in a parallel region must wait
 *      before any can proceed, ensuring coordinated execution.
 *
 */
