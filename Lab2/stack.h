#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H

// pthread_mutex_t cccccc= PTHREAD_MUTEX_INITIALIZER;
struct stack
{
  struct Node *head;
  pthread_mutex_t lock;
};
typedef struct stack stack_t;
typedef struct Node
{
  int data;
  struct Node *next;
} Node;

typedef struct NodePool
{
  Node *pool; // 池中的 Node 数组
  int size;   // 池的大小
  int count;  // 当前已分配的 Node 数量
} NodePool;
Node *getNodeFromPool(NodePool *nodePool);
void returnNodeToPool(NodePool *nodePool, Node *node);
void destroyNodePool(NodePool *nodePool);
void initNodePool(NodePool *nodePool, int poolSize);
void /* Return the type you prefer */
stack_push(stack_t *stack, int value, Node *newNode);
int stack_pop(stack_t *stack);
int stack_pop_return(stack_t *stack, NodePool nodePool);
/* Use this to check if your stack is in a consistent state from time to time */
int stack_check(stack_t *stack);
#endif /* STACK_H */
