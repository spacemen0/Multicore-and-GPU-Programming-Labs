/*
 * stack.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 *
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

int stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
  // Use assert() to check if your stack is in a state that makes sens
  // This test should always pass
  assert(1 == 1);

  // This test fails if the task is not allocated or if the allocation failed
  assert(stack != NULL);
#endif
  // The stack is always fine
  return 1;
}

Node * /* Return the type you prefer */
stack_pop(stack_t *stack)
{
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&stack->lock);

  if (stack->head == NULL)
  {
    pthread_mutex_unlock(&stack->lock);
    return NULL;
  }

  Node *oldHead = stack->head;
  stack->head = oldHead->next;
  pthread_mutex_unlock(&stack->lock);
  return oldHead;

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  if (stack->head == NULL)
  {
    return NULL;
  }
  Node *oldHead;
  do
  {
    oldHead = stack->head;
  } while (cas((size_t *)&(stack->head), (size_t)oldHead, (size_t)oldHead->next) != (size_t)oldHead);

  return oldHead;
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t *)1);
}

// int stack_pop_return(stack_t *stack, NodePool nodePool)
// {
// #if NON_BLOCKING == 0
//   // Implement a lock_based stack
//   pthread_mutex_lock(&stack->lock);

//   if (stack->head == NULL)
//   {
//     pthread_mutex_unlock(&stack->lock);
//     return -1;
//   }

//   Node *oldHead = stack->head;
//   int data = oldHead->data;
//   stack->head = oldHead->next;

//   pthread_mutex_unlock(&stack->lock);

//     returnNodeToPool(&nodePool, oldHead);

//   return data;
// #elif NON_BLOCKING == 1
//   // Implement a harware CAS-based stack
//   if (stack->head == NULL)
//   {
//     return -1;
//   }
//   Node *oldHead;
//   int data;
//   do
//   {
//     oldHead = stack->head;
//     data = oldHead->data;
//   } while (cas((size_t *)&(stack->head), (size_t)oldHead, (size_t)oldHead->next) != (size_t)oldHead);
//   returnNodeToPool(&nodePool, oldHead);

//   return data;
// #else
//   /*** Optional ***/
//   // Implement a software CAS-based stack
// #endif

//   // Debug practice: you can check if this operation results in a stack in a consistent check
//   // It doesn't harm performance as sanity check are disabled at measurement time
//   // This is to be updated as your implementation progresses
//   stack_check((stack_t *)1);
// }

void /* Return the type you prefer */
stack_push(stack_t *stack, Node *newNode)
{
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  if (newNode == NULL)
  {
    return;
  }
  pthread_mutex_lock(&stack->lock);
  newNode->next = stack->head;
  stack->head = newNode;

  pthread_mutex_unlock(&stack->lock);
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  if (newNode == NULL)
  {
    return;
  }
  Node *oldHead;

  do
  {
    oldHead = stack->head;
    newNode->next = oldHead;
  } while (cas((size_t *)&(stack->head), (size_t)oldHead, (size_t)newNode) != (size_t)oldHead);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif
}

// void initNodePool(NodePool *nodePool, int poolSize)
// {
//   nodePool->pool = (Node *)malloc(sizeof(Node) * poolSize);
//   nodePool->size = poolSize;
//   nodePool->count = 0;

//   for (int i = 0; i < poolSize - 1; ++i)
//   {
//     nodePool->pool[i].next = &nodePool->pool[i + 1];
//   }

//   nodePool->pool[poolSize - 1].next = NULL;
// }

// Node *getNodeFromPool(NodePool *nodePool)
// {
//   if (nodePool->count < nodePool->size)
//   {
//     Node *node = &nodePool->pool[nodePool->count++];
//     node->next = NULL;
//     return node;
//   }
//   else
//   {
//     return NULL;
//   }
// }

// void returnNodeToPool(NodePool *nodePool, Node *node)
// {
//   if (nodePool->count > 0)
//   {
//     --nodePool->count;
//     node->next = &nodePool->pool[nodePool->count];
//   }
// }

// void destroyNodePool(NodePool *nodePool)
// {
//   free(nodePool->pool);
//   nodePool->pool = NULL;
//   nodePool->size = 0;
//   nodePool->count = 0;
// }
