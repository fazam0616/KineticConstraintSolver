#include "datastructures.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Linked List
ListNode* list_create_node(void *data) {
    ListNode *node = (ListNode*)malloc(sizeof(ListNode));
    node->data = data;
    node->next = NULL;
    return node;
}

void list_append(ListNode **head, void *data) {
    ListNode *new_node = list_create_node(data);
    if (!*head) {
        *head = new_node;
        return;
    }
    ListNode *curr = *head;
    while (curr->next) curr = curr->next;
    curr->next = new_node;
}

void list_free(ListNode *head, void (*free_data)(void *)) {
    ListNode *curr = head;
    while (curr) {
        ListNode *next = curr->next;
        if (free_data) free_data(curr->data);
        free(curr);
        curr = next;
    }
}

// Dynamic Array
DynArray* dynarray_create(size_t initial_capacity) {
    DynArray *arr = (DynArray*)malloc(sizeof(DynArray));
    arr->items = (void**)malloc(sizeof(void*) * initial_capacity);
    arr->size = 0;
    arr->capacity = initial_capacity;
    return arr;
}

void dynarray_append(DynArray *arr, void *item) {
    if (arr->size == arr->capacity) {
        arr->capacity *= 2;
        arr->items = (void**)realloc(arr->items, sizeof(void*) * arr->capacity);
    }
    arr->items[arr->size++] = item;
}

void dynarray_free(DynArray *arr, void (*free_item)(void *)) {
    if (free_item) {
        for (size_t i = 0; i < arr->size; ++i) {
            free_item(arr->items[i]);
        }
    }
    free(arr->items);
    free(arr);
}

void dynarray_sort(DynArray *arr, int (*cmp)(const void *, const void *)) {
    qsort(arr->items, arr->size, sizeof(void*), cmp);
}

/* dynarray_size and dynarray_get are defined as static inline in the header
   (datastructures.h) to allow inlining across translation units and avoid
   requiring external linkage. */

// Hashmap
#define HASHMAP_PRIME 31
// Default byte-mix hash (FNV-like mixing). Returns a bucket index in [0, bucket_count)
static size_t default_hash_fn(const void *key, size_t key_size, size_t bucket_count) {
    const unsigned char *p = (const unsigned char*)key;
    uint64_t h = 14695981039346656037ULL; // FNV offset basis
    for (size_t i = 0; i < key_size; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL; // FNV prime
    }
    return (size_t)(h % (uint64_t)bucket_count);
}

HashMap* hashmap_create(size_t bucket_count, size_t key_size, size_t (*hash_fn)(const void *key, size_t bucket_count)) {
    HashMap *map = (HashMap*)malloc(sizeof(HashMap));
    map->buckets = (HashNode**)calloc(bucket_count, sizeof(HashNode*));
    map->bucket_count = bucket_count;
    map->key_size = key_size;
    if (hash_fn) map->hash_fn = hash_fn;
    else {
        // wrap default to match expected signature
        map->hash_fn = NULL; // indicate use of internal default
    }
    return map;
}

void hashmap_put(HashMap *map, const void *key, void *value) {
    if (!map || !key) return;
    size_t idx;
    if (map->hash_fn) idx = map->hash_fn(key, map->bucket_count) % map->bucket_count;
    else idx = default_hash_fn(key, map->key_size, map->bucket_count);

    HashNode *node = map->buckets[idx];
    while (node) {
        if (memcmp(node->key, key, map->key_size) == 0) {
            node->value = value;
            return;
        }
        node = node->next;
    }
    node = (HashNode*)malloc(sizeof(HashNode));
    node->key = malloc(map->key_size);
    memcpy(node->key, key, map->key_size);
    node->value = value;
    node->next = map->buckets[idx];
    map->buckets[idx] = node;
}

void* hashmap_get(HashMap *map, const void *key) {
    if (!map || !key) return NULL;
    size_t idx;
    if (map->hash_fn) idx = map->hash_fn(key, map->bucket_count) % map->bucket_count;
    else idx = default_hash_fn(key, map->key_size, map->bucket_count);

    HashNode *node = map->buckets[idx];
    while (node) {
        if (memcmp(node->key, key, map->key_size) == 0) return node->value;
        node = node->next;
    }
    return NULL;
}

void hashmap_free(HashMap *map, void (*free_value)(void *)) {
    if (!map) return;
    for (size_t i = 0; i < map->bucket_count; ++i) {
        HashNode *node = map->buckets[i];
        while (node) {
            HashNode *next = node->next;
            if (node->key) free(node->key);
            if (free_value) free_value(node->value);
            free(node);
            node = next;
        }
    }
    free(map->buckets);
    free(map);
}
