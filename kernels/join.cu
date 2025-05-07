#include "join.cuh"

__device__ bool eval_join_condition(char *row_ptr_a, int *acc_col_size_a,char *row_ptr_b, int *acc_col_size_b, const Join_Condition &cond)
{
    // start of needed col
    char *field_ptr_a = row_ptr_a + acc_col_size_a[cond.col_index_a];
    char *field_ptr_b = row_ptr_b + acc_col_size_b[cond.col_index_b];
    if (cond.type == 0)
    { // numerical
        double *val1 = (double *)malloc(sizeof(double));
        double *val2 = (double *)malloc(sizeof(double));
        
        memcpy(val1, field_ptr_a, sizeof(double));
        memcpy(val2, field_ptr_b, sizeof(double));
        printf("val1: %f, val2: %f\n", *val1, *val2);
        switch (cond.op)
        {
        case OP_GT:
            return *val1 > *val2;
        case OP_LT:
            return *val1 < *val2;
        case OP_EQ:
            return *val1 == *val2;
        case OP_NEQ:
            return *val1 != *val2;
        case OP_GTE:
            return *val1 >= *val2;
        case OP_LTE:
            return *val1 <= *val2;
        }
    }
    else
    { // text
        char *val1 = (char *)malloc(150);
        char *val2 = (char *)malloc(150);
        
        memcpy(val1, field_ptr_a, 150);
        memcpy(val2, field_ptr_b, 150);
        switch (cond.op)
        {
        case OP_EQ:
            return device_strcmp(val1, val2) == 0;
        case OP_NEQ:
            return device_strcmp(val1, val2) != 0;
        }
    }
    return false;
}

__device__ bool eval_condition_tokens(char *row_ptr_a,char *row_ptr_b, int *acc_col_size_a,int *acc_col_size_b, JoinConditionToken *tokens, int token_count)
{
    bool stack[16]; // adjust size if needed
    int sp = 0;     // stack pointer

    for (int i = 0; i < token_count; ++i)
    {
        if (tokens[i].type == TOKEN_CONDITION)
        {
            bool res = eval_join_condition(row_ptr_a, acc_col_size_a, row_ptr_b, acc_col_size_b, tokens[i].joinCond);
            stack[sp++] = res;
        }
        else if (tokens[i].type == TOKEN_AND)
        {
            if (sp < 2)
                return false; // stack underflow
            bool b = stack[--sp];
            bool a = stack[--sp];
            stack[sp++] = a && b;
        }
        else if (tokens[i].type == TOKEN_OR)
        {
            if (sp < 2)
                return false;
            bool b = stack[--sp];
            bool a = stack[--sp];
            stack[sp++] = a || b;
        }
    }

    return sp == 1 ? stack[0] : false;
}

__global__ void nested_loop_join(char* table_a, int size_a,int row_size_a,int *acc_col_size_a,char *table_b, int size_b,int row_size_b,int *acc_col_size_b, char *result, int *resultCount,JoinConditionToken *joinConds, int cond_count) 
{
    int aIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (aIdx >= size_a) return;
    char *row_ptr_a = &table_a[aIdx * row_size_a];
    for (int j = 0; j < size_b; ++j) {
        char *row_ptr_b = &table_b[j * row_size_b];
        printf("Thread %d processing row %d\n", threadIdx.x, j);
        bool pass = eval_condition_tokens(row_ptr_a,row_ptr_b, acc_col_size_a,acc_col_size_b, joinConds, cond_count);

        if (pass) {
            int index = atomicAdd(resultCount, 1);
            if (index < MAX_JOINED_ROWS) {
                // TODO: future work
                char *out_ptr = result + index * (row_size_a + row_size_b);
                for (int i = 0; i < row_size_a; ++i)
                {
                    out_ptr[i] = row_ptr_a[i];
                }
                for (int i = row_size_a; i <(row_size_a + row_size_b); ++i)
                {
                    out_ptr[i] = row_ptr_b[i - row_size_a];
                }
            }
        }
    }
}

__host__ char *call_join_kernel(char* table_a, int size_a,int row_size_a,int *acc_col_size_a,char *table_b, int size_b,int row_size_b,int *acc_col_size_b, int& resultCount,JoinConditionToken *joinConds, int cond_count, int column_num_a, int column_num_b)
{
    int *h_output_counter = (int *)malloc(sizeof(int));

    // Allocate device memory

    char *d_table_a, *d_table_b, *d_result;
    int *d_acc_col_size_a, *d_acc_col_size_b, *d_resultCount;
    JoinConditionToken *d_joinConds;

    cudaMalloc((void**)&d_table_a, size_a * row_size_a);
    cudaMalloc((void**)&d_table_b, size_b * row_size_b);
    cudaMalloc((void**)&d_result, MAX_JOINED_ROWS * (row_size_a + row_size_b));

    cudaMalloc((void**)&d_acc_col_size_a, sizeof(int) * column_num_a); 
    cudaMalloc((void**)&d_acc_col_size_b, sizeof(int) * column_num_b); 
    cudaMalloc((void**)&d_resultCount, sizeof(int));
    cudaMalloc((void**)&d_joinConds, sizeof(JoinConditionToken) * cond_count);

    cudaMemcpy(d_table_a, table_a, size_a * row_size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_table_b, table_b, size_b * row_size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_acc_col_size_a, acc_col_size_a, sizeof(int) * column_num_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_acc_col_size_b, acc_col_size_b, sizeof(int) * column_num_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_joinConds, joinConds, sizeof(JoinConditionToken) * cond_count, cudaMemcpyHostToDevice);
    
    cudaMemset(d_resultCount, 0, sizeof(int));
    int blockSize = 256;
    int numBlocks = (size_a + blockSize - 1) / blockSize;

    nested_loop_join<<<numBlocks, blockSize>>>(d_table_a,size_a,row_size_a,d_acc_col_size_a,d_table_b,size_b,row_size_b,d_acc_col_size_b,d_result,d_resultCount,d_joinConds,cond_count);

    cudaMemcpy(h_output_counter,d_resultCount,sizeof(int),cudaMemcpyDeviceToHost);
    
    char *h_output_data = (char *)malloc(*h_output_counter * (row_size_a + row_size_b) * sizeof(char));


    cudaMemcpy(h_output_data,d_result, (*h_output_counter * (row_size_a + row_size_b)),cudaMemcpyDeviceToHost);

    resultCount = *h_output_counter;
    // Free device memory
    cudaFree(d_table_a);
    cudaFree(d_table_b);
    cudaFree(d_result);
    cudaFree(d_acc_col_size_a);
    cudaFree(d_acc_col_size_b);
    cudaFree(d_resultCount);
    cudaFree(d_joinConds);

    return h_output_data;

}