#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H

// pthread_mutex_t cccccc= PTHREAD_MUTEX_INITIALIZER;
struct stack
{
  struct node *head;
};
struct node
{
  int value;
  struct node *next;
};
typedef struct stack stack_t;

void stack_push(stack_t *stack, int value);
void stack_pop(stack_t *stack);

/* Use this to check if your stack is in a consistent state from time to time */
int stack_check(stack_t *stack);
#endif /* STACK_H */
