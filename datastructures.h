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

/* Small, performance-critical accessors: provide as static inline in the header
   so each translation unit can inline them without requiring an external symbol. */
static inline size_t dynarray_size(DynArray *arr) {
    if (!arr) return 0;
    return arr->size;
}

static inline void* dynarray_get(DynArray *arr, size_t index) {
    if (!arr) return NULL;
    if (index >= arr->size) return NULL;
    return arr->items[index];
}

// Hashmap
typedef struct HashNode {
    void *key;
    void *value;
    struct HashNode *next;
} HashNode;

typedef struct {
    HashNode **buckets;
    size_t bucket_count;
    size_t key_size;
    size_t (*hash_fn)(const void *key, size_t bucket_count);
} HashMap;

/* Create a hashmap.
 * bucket_count: number of buckets to allocate
 * key_size: size in bytes of the key used for this map
 * hash_fn: function that maps a key to a bucket index (0..bucket_count-1). If NULL,
 *          a default byte-mixing hash will be used.
 */
HashMap* hashmap_create(size_t bucket_count, size_t key_size, size_t (*hash_fn)(const void *key, size_t bucket_count));
void hashmap_put(HashMap *map, const void *key, void *value);
void* hashmap_get(HashMap *map, const void *key);
void hashmap_free(HashMap *map, void (*free_value)(void *));

#endif // DATASTRUCTURES_H
