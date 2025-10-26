#include "datastructures.h"
#include <stdlib.h>
#include <string.h>

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

size_t dynarray_size(DynArray *arr) {
    if (!arr) return 0;
    return arr->size;
}

void* dynarray_get(DynArray *arr, size_t index) {
    if (!arr) return NULL;
    if (index >= arr->size) return NULL;
    return arr->items[index];
}

// Hashmap
#define HASHMAP_PRIME 31
static size_t hash(const char *key, size_t bucket_count) {
    size_t h = 0;
    while (*key) h = h * HASHMAP_PRIME + (unsigned char)(*key++);
    return h % bucket_count;
}

HashMap* hashmap_create(size_t bucket_count) {
    HashMap *map = (HashMap*)malloc(sizeof(HashMap));
    map->buckets = (HashNode**)calloc(bucket_count, sizeof(HashNode*));
    map->bucket_count = bucket_count;
    return map;
}

void hashmap_put(HashMap *map, const char *key, void *value) {
    size_t idx = hash(key, map->bucket_count);
    HashNode *node = map->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            node->value = value;
            return;
        }
        node = node->next;
    }
    node = (HashNode*)malloc(sizeof(HashNode));
    node->key = strdup(key);
    node->value = value;
    node->next = map->buckets[idx];
    map->buckets[idx] = node;
}

void* hashmap_get(HashMap *map, const char *key) {
    size_t idx = hash(key, map->bucket_count);
    HashNode *node = map->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0)
            return node->value;
        node = node->next;
    }
    return NULL;
}

void hashmap_free(HashMap *map, void (*free_value)(void *)) {
    for (size_t i = 0; i < map->bucket_count; ++i) {
        HashNode *node = map->buckets[i];
        while (node) {
            HashNode *next = node->next;
            free(node->key);
            if (free_value) free_value(node->value);
            free(node);
            node = next;
        }
    }
    free(map->buckets);
    free(map);
}
