#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Node structure representing a tree node
struct TreeNode {
    int data;
    vector<TreeNode*> children;

    TreeNode(int val) : data(val) {}
};

// Tree class representing the tree structure
class Tree {
    TreeNode* root;

public:
    Tree(int val) {
        root = new TreeNode(val);
    }

    void addChild(TreeNode* parent, int val) {
        TreeNode* newNode = new TreeNode(val);
        parent->children.push_back(newNode);
    }

    TreeNode* getRoot() {
        return root;
    }

    // ✅ Parallel DFS using OpenMP tasks
    void parallelDFS(TreeNode* node) {
        if (!node) return;

        cout << node->data << " ";

        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                for (TreeNode* child : node->children) {
                    #pragma omp task firstprivate(child)
                    parallelDFS(child);
                }
            }
        }
    }

    // ✅ Parallel BFS using level-by-level OpenMP parallelism
    void parallelBFS() {
        if (!root) return;

        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            int levelSize = q.size();
            vector<TreeNode*> nextLevel;

            #pragma omp parallel for
            for (int i = 0; i < levelSize; ++i) {
                TreeNode* current;

                // Synchronize access to the queue
                #pragma omp critical
                {
                    current = q.front();
                    q.pop();
                }

                #pragma omp critical
                cout << current->data << " ";

                // Save children for next level (synchronized)
                #pragma omp critical
                {
                    for (TreeNode* child : current->children) {
                        nextLevel.push_back(child);
                    }
                }
            }

            // Push all next level nodes into the queue
            for (TreeNode* node : nextLevel) {
                q.push(node);
            }
        }
    }
};

int main() {
    // Build the tree
    Tree tree(1);
    TreeNode* root = tree.getRoot();
    tree.addChild(root, 2);
    tree.addChild(root, 3);
    tree.addChild(root, 4);

    TreeNode* node2 = root->children[0];
    tree.addChild(node2, 5);
    tree.addChild(node2, 6);

    TreeNode* node4 = root->children[2];
    tree.addChild(node4, 7);
    tree.addChild(node4, 8);

    /*
               1
             / | \
            2  3  4
           / \    / \
          5   6  7   8
    */

    cout << "Parallel Depth-First Search (DFS): ";
    tree.parallelDFS(root);
    cout << endl;

    cout << "Parallel Breadth-First Search (BFS): ";
    tree.parallelBFS();
    cout << endl;

    return 0;
}
