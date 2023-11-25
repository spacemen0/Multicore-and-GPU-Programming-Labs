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
  struct Node *next;
} Node;


void /* Return the type you prefer */
stack_push(stack_t *stack, Node *newNode);
Node* stack_pop(stack_t *stack);
// int stack_pop_return(stack_t *stack, NodePool nodePool);
/* Use this to check if your stack is in a consistent state from time to time */
int stack_check(stack_t *stack);
#endif /* STACK_H */
