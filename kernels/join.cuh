// kernels/join.cuh
#ifndef JOIN_KERNELS_H
#define JOIN_KERNELS_H

#include "get.cuh"
#include <stdlib.h>
#include <stdio.h>
#define BLOCK_SIZE 256
#define MAX_JOINED_ROWS 1000000

struct Join_Condition {
    int col_index_a;    
    int col_index_b;   
    Operator op ;  
    int type;        
};



struct JoinConditionToken {
    ConditionTokenType type;
    Join_Condition joinCond; // valid if type == TOKEN_CONDITION
};

// CUDA Kernel for Nested Loop Join without Shared Memory
__device__ bool eval_join_condition(char *row_ptr_a, int *acc_col_size_a,char *row_ptr_b, int *acc_col_size_b, const Join_Condition &cond);

__global__ void nested_loop_join(char* table_a, int size_a,int row_size_a,int *acc_col_size_a,char *table_b, int size_b,int row_size_b,int *acc_col_size_b, char *result, int *resultCount,Join_Condition *joinConds, int cond_count);

__host__ char *call_join_kernel(char* table_a, int size_a,int row_size_a,int *acc_col_size_a,char *table_b, int size_b,int row_size_b,int *acc_col_size_b, int& resultCount,Join_Condition *joinConds, int cond_count, int column_num_a, int column_num_b);

#endif // JOIN_KERNELS_H