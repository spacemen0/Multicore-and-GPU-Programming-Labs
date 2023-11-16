#include "stack.h"
#include <stdio.h>
int main()
{
    struct stack *a;
    a->head = (struct node *)malloc(sizeof(struct node *));
    a->head->value = 3;
    stack_push(a, 2);
    stack_push(a, 1);
    printf("what%d\n", a->head->next->next->value);
}