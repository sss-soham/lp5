/*
 * Problem Statement:
 * Design and implement Parallel Depth First Search based on existing algorithms using OpenMP.
 * Use a Tree or an undirected graph for DFS.
 *
 * How to run:
 * 1. Open terminal in the directory containing the file
 * 2. Compile: g++ -fopenmp 02_Parallel_DFS.cpp -o 02_Parallel_DFS
 *    (if above not worked): g++ 02_Parallel_DFS.cpp -o 02_Parallel_DFS
 *    (General command): g++ -fopenmp fileName.cpp -o fileName or g++ fileName.cpp -o fileName
 * 3. Run: ./02_Parallel_DFS or .\02_Parallel_DFS
 */

#include <iostream>
#include <vector>
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

    void DFS(int start)
    {
        vector<bool> visited(V, false);
#pragma omp parallel
        {
#pragma omp single nowait
            {
                DFSUtil(start, visited);
            }
        }
        cout << endl;
    }

    void DFSUtil(int u, vector<bool> &visited)
    {
        visited[u] = true;
        cout << u << " ";

#pragma omp parallel for
        for (int i = 0; i < adj[u].size(); i++)
        {
            int v = adj[u][i];
            if (!visited[v])
            {
#pragma omp task
                {
                    DFSUtil(v, visited);
                }
            }
        }
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
    cout << "DFS traversal starting from vertex 0: ";
    g1.DFS(0);
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
    cout << "DFS traversal starting from vertex 0: ";
    g2.DFS(0);

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

    cout << "Parallel DFS traversal:" << endl;
    g.DFS(0);

    return 0;
}

/*
 * Parallel Depth-First Search (DFS) Implementation using OpenMP
 * ==========================================================
 *
 * Overview:
 * ---------
 * This program implements a parallel version of the Depth-First Search algorithm
 * using OpenMP for graph traversal. It includes both predefined examples and
 * user-input functionality for graph creation and traversal.
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
 * Data Structures:
 * ---------------
 * 1. Graph (struct)
 *    - Adjacency list representation using vector<vector<int>>
 *    - Boolean visited array for tracking traversal
 *
 * Complexity Analysis:
 * -------------------
 * Time Complexity:
 * - Sequential DFS: O(V + E) where V = vertices, E = edges
 * - Parallel DFS: O((V + E)/p) theoretical, where p = number of processors
 *
 * Space Complexity:
 * - O(V) for visited array
 * - O(V + E) for adjacency list
 * - O(V) additional space for recursion stack
 *
 * Parallel Performance Factors:
 * ---------------------------
 * 1. Task Granularity: Balance between parallel overhead and work distribution
 * 2. Graph Structure: Dense vs sparse affects parallelization efficiency
 * 3. Load Balancing: Uneven subtree sizes can lead to workload imbalance
 * 4. Memory Access Patterns: Cache performance and false sharing
 *
 * Q&A Section:
 * -----------
 * Q1: What is the purpose of #pragma omp parallel?
 * A1: Creates a team of threads for parallel execution. Each thread executes the same code.
 *
 * Q2: Why use #pragma omp single nowait?
 * A2: Ensures only one thread starts the initial DFS while others wait for tasks.
 *     'nowait' allows other threads to proceed without synchronization.
 *
 * Q3: What's the role of #pragma omp task?
 * A3: Creates a new task for parallel execution of DFS subtrees.
 *     Example: #pragma omp task { DFSUtil(v, visited); }
 *
 * Q4: How is thread safety ensured in the visited array?
 * A4: Each vertex is marked visited before spawning tasks, preventing duplicate visits.
 *
 * Q5: Why use adjacency list over adjacency matrix?
 * A5: Better space efficiency (O(V+E) vs O(V²)) and faster traversal for sparse graphs.
 *
 * Q6: What's the impact of graph connectivity on parallel performance?
 * A6: Higher connectivity means more potential parallel tasks but also more synchronization overhead.
 *
 * Q7: How does task scheduling work in this implementation?
 * A7: OpenMP runtime schedules tasks dynamically to available threads using a work-stealing algorithm.
 *
 * Q8: What's the purpose of vector<bool> visited?
 * A8: Tracks visited vertices to prevent cycles and repeated processing.
 *
 * Q9: Why is the graph undirected in this implementation?
 * A9: addEdge() adds both (u,v) and (v,u), creating symmetric connections.
 *
 * Q10: How could the implementation be optimized further?
 * A10: - Use task cutoff for small subtrees
 *      - Implement cache-friendly vertex numbering
 *      - Add parallel task merging
 *
 * Q11: What's the significance of #pragma omp parallel for?
 * A11: Distributes loop iterations across threads, useful for processing adjacent vertices.
 *
 * Q12: How does false sharing affect performance?
 * A12: Adjacent elements in visited array accessed by different threads can cause cache line bouncing.
 *
 * Q13: What determines optimal task granularity?
 * A13: Balance between parallelization overhead and work per task. Depends on hardware and graph structure.
 *
 * Q14: How does this compare to sequential DFS?
 * A14: Parallel version can be faster for large graphs but has overhead for small graphs.
 *
 * Q15: What are the limitations of this implementation?
 * A15: - No handling of disconnected components
 *      - Potential overhead for small graphs
 *      - Memory contention in dense graphs
 *
 * Q16: What is the purpose of #pragma omp critical?
 * A16: Ensures mutual exclusion, allowing only one thread to execute a code block at a time,
 *      preventing race conditions in shared resource access.
 *
 * Q17: How does #pragma omp taskwait work in OpenMP?
 * A17: Forces the current thread to wait until all its child tasks complete,
 *      useful for synchronization points in the algorithm.
 *
 * Q18: What is the role of OMP_NUM_THREADS environment variable?
 * A18: Sets the default number of threads for parallel regions in OpenMP programs,
 *      allowing runtime control of parallelism level.
 *
 * Q19: How does #pragma omp atomic differ from critical?
 * A19: Provides faster, hardware-supported atomic operations for simple updates to shared variables,
 *      more efficient than critical sections for single operations.
 *
 * Q20: What is the purpose of #pragma omp barrier?
 * A20: Creates a synchronization point where all threads in a parallel region must wait
 *      before any can proceed, ensuring coordinated execution.
 *
 */