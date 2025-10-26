#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include <stddef.h>

// Linked List
typedef struct ListNode {
    void *data;
    struct ListNode *next;
} ListNode;

ListNode* list_create_node(void *data);
void list_append(ListNode **head, void *data);
void list_free(ListNode *head, void (*free_data)(void *));

// Dynamic Array
typedef struct {
    void **items;
    size_t size;
    size_t capacity;
} DynArray;

DynArray* dynarray_create(size_t initial_capacity);
void dynarray_append(DynArray *arr, void *item);
void dynarray_free(DynArray *arr, void (*free_item)(void *));
void dynarray_sort(DynArray *arr, int (*cmp)(const void *, const void *));
size_t dynarray_size(DynArray *arr);
void* dynarray_get(DynArray *arr, size_t index);

// Hashmap
typedef struct HashNode {
    char *key;
    void *value;
    struct HashNode *next;
} HashNode;

typedef struct {
    HashNode **buckets;
    size_t bucket_count;
} HashMap;

HashMap* hashmap_create(size_t bucket_count);
void hashmap_put(HashMap *map, const char *key, void *value);
void* hashmap_get(HashMap *map, const char *key);
void hashmap_free(HashMap *map, void (*free_value)(void *));

#endif // DATASTRUCTURES_H
