#include <ATen/ATen.h>
#include <immintrin.h>
#include <sys/time.h>
#include <time.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

// Uncomment for ISPC
// #include "module_ispc.h"
// using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX) + y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
                         const int &sizeX, const int &sizeY, const int &sizeZ) {
    // calc index using 4D flattening formula
    int index = x * (sizeX * sizeY * sizeZ) +
                y * (sizeY * sizeZ) +
                z * sizeZ +
                b;
    return tensor[index];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
                         const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    // calc the flattened index in the 1D array for the 4D coordinate
    int index = x * (sizeX * sizeY * sizeZ) +
                y * (sizeY * sizeZ) +
                z * sizeZ +
                b;
    tensor[index] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 *
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                               int B, int H, int N, int d) {
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // QK^t Intermediate Tensor has Shape (N, N)

    // Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    // Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
               for (int j = 0; j < N; j++) {
                   float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
                   twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */

    // -------- YOUR CODE HERE  -------- //

    // Loop through each batch
    int batch = 0;
    while (batch < B) {
        for (int head = 0; head < H; head++) {
            // find QK^T
            // init QK_t to zeros
            for (int row = 0; row < N; row++) {
                int col = 0;
                while (col < N) {
                    float zero_val = 0.0;
                    twoDimWrite(QK_t, row, col, N, zero_val);
                    col++;
                }
            }

            // matrix multiplication Q * K^T
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float dot_product_result = 0.0;

                    // dot products
                    int dim_idx = 0;
                    while (dim_idx < d) {
                        // Q[batch, head, i, dim]
                        float q_val = fourDimRead(Q, batch, head, i, dim_idx, H, N, d);

                        // K[batch, head, j, dim]
                        float k_val = fourDimRead(K, batch, head, j, dim_idx, H, N, d);

                        // sum of dot products
                        dot_product_result += q_val * k_val;
                        dim_idx++;
                    }

                    // store result in QK_t
                    twoDimWrite(QK_t, i, j, N, dot_product_result);
                }
            }

            // part b
            for (int row = 0; row < N; row++) {
                // get exponentials and their sum of each row
                float row_sum = 0.0;

                for (int col = 0; col < N; col++) {
                    float current_val = twoDimRead(QK_t, row, col, N);

                    float exp_val = exp(current_val);

                    // save exponent val to QK_t
                    twoDimWrite(QK_t, row, col, N, exp_val);

                    row_sum += exp_val;
                }

                // divide by sum
                for (int col = 0; col < N; col++) {
                    float exp_val = twoDimRead(QK_t, row, col, N);

                    float normalized_val = exp_val / row_sum;

                    // store back to QK_t matrix
                    twoDimWrite(QK_t, row, col, N, normalized_val);
                }
            }

            // part c
            // O = QK_t * V
            for (int i = 0; i < N; i++) {
                int j = 0;
                while (j < d) {
                    float cell_result = 0.0;

                    for (int k = 0; k < N; k++) {
                        float qk_val = twoDimRead(QK_t, i, k, N);

                        float v_val = fourDimRead(V, batch, head, k, j, H, N, d);

                        cell_result += qk_val * v_val;
                    }

                    fourDimWrite(O, batch, head, i, j, H, N, d, cell_result);
                    j++;
                }
            }
        }
        batch++;
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                                        int B, int H, int N, int d) {
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // QK^t Intermediate Tensor has Shape (N, N)

    // Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    // Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //

    // block sizes for tiling
    const int blockSizeN = 24;
    const int blockSizeD = 24;

    int batchIdx = 0;
    while (batchIdx < B) {
        int headIdx = 0;
        while (headIdx < H) {
            // init QK_t matrix wiht 0
            int i_init = 0;
            while (i_init < N) {
                for (int j_init = 0; j_init < N; j_init++) {
                    float initialize_val = 0.0f;
                    twoDimWrite(QK_t, i_init, j_init, N, initialize_val);
                }
                i_init++;
            }

            for (int n_block_i = 0; n_block_i < N; n_block_i += blockSizeN) {
                int actual_block_i = std::min(blockSizeN, N - n_block_i);

                for (int n_block_j = 0; n_block_j < N; n_block_j += blockSizeN) {
                    int actual_block_j = std::min(blockSizeN, N - n_block_j);

                    for (int d_block = 0; d_block < d; d_block += blockSizeD) {
                        int actual_block_d = std::min(blockSizeD, d - d_block);

                        int block_row = 0;
                        while (block_row < actual_block_i) {
                            int global_row = n_block_i + block_row;

                            for (int block_col = 0; block_col < actual_block_j; block_col++) {
                                int global_col = n_block_j + block_col;

                                float accumulated_value = 0.0f;
                                if (d_block > 0) {
                                    accumulated_value = twoDimRead(QK_t, global_row, global_col, N);
                                }

                                for (int d_idx = 0; d_idx < actual_block_d; d_idx++) {
                                    int global_d = d_block + d_idx;

                                    float q_elem = fourDimRead(Q, batchIdx, headIdx, global_row, global_d, H, N, d);
                                    float k_elem = fourDimRead(K, batchIdx, headIdx, global_col, global_d, H, N, d);

                                    // sum of dot products
                                    accumulated_value += q_elem * k_elem;
                                }

                                // store back
                                twoDimWrite(QK_t, global_row, global_col, N, accumulated_value);
                            }
                            block_row++;
                        }
                    }
                }
            }

            // softmax
            for (int row_idx = 0; row_idx < N; row_idx++) {
                float sum_exponentials = 0.0f;

                int col_idx = 0;
                while (col_idx < N) {
                    float current_val = twoDimRead(QK_t, row_idx, col_idx, N);

                    float exponential_val = exp(current_val);

                    twoDimWrite(QK_t, row_idx, col_idx, N, exponential_val);

                    // sum of exponentials
                    sum_exponentials += exponential_val;

                    col_idx++;
                }

                for (int col_norm = 0; col_norm < N; col_norm++) {
                    float exp_val = twoDimRead(QK_t, row_idx, col_norm, N);

                    float normalized = exp_val / sum_exponentials;

                    // save back
                    twoDimWrite(QK_t, row_idx, col_norm, N, normalized);
                }
            }

            // block matrix multiplication for QK_t * V
            // zero initialize
            for (int out_i = 0; out_i < N; out_i++) {
                int out_j = 0;
                while (out_j < d) {
                    float zero_val = 0.0f;
                    fourDimWrite(O, batchIdx, headIdx, out_i, out_j, H, N, d, zero_val);
                    out_j++;
                }
            }

            // process blocks of the output matrix O (N x d)
            for (int n_block_i = 0; n_block_i < N; n_block_i += blockSizeN) {
                int actual_block_i = std::min(blockSizeN, N - n_block_i);

                for (int d_block_j = 0; d_block_j < d; d_block_j += blockSizeD) {
                    int actual_block_d = std::min(blockSizeD, d - d_block_j);

                    for (int n_block_k = 0; n_block_k < N; n_block_k += blockSizeN) {
                        int actual_block_k = std::min(blockSizeN, N - n_block_k);

                        for (int block_i = 0; block_i < actual_block_i; block_i++) {
                            int global_i = n_block_i + block_i;

                            int block_j = 0;
                            while (block_j < actual_block_d) {
                                int global_j = d_block_j + block_j;

                                float result_sum = fourDimRead(O, batchIdx, headIdx, global_i, global_j, H, N, d);

                                for (int block_k = 0; block_k < actual_block_k; block_k++) {
                                    int global_k = n_block_k + block_k;

                                    float qk_elem = twoDimRead(QK_t, global_i, global_k, N);
                                    float v_elem = fourDimRead(V, batchIdx, headIdx, global_k, global_j, H, N, d);

                                    result_sum += qk_elem * v_elem;
                                }

                                // store the result
                                fourDimWrite(O, batchIdx, headIdx, global_i, global_j, H, N, d, result_sum);

                                block_j++;
                            }
                        }
                    }
                }
            }

            headIdx++;
        }
        batchIdx++;
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

// helper function to calculate flattened 4d index
inline int calc_4d_index(const int &batch, const int &head, const int &seq, const int &embed,
                         const int &heads_dim, const int &seq_dim, const int &embed_dim) {
    int res = batch * (heads_dim * seq_dim * embed_dim) +
              head * (seq_dim * embed_dim) +
              seq * embed_dim +
              embed;

    return res;
}

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                               int B, int H, int N, int d) {
    // Q, K, V are passed in with Shape: (B, H, N, d)

    // Make O Tensor with Shape (B, H, N, d)
    // and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    // Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format ORow Tensor into a 1D vector
    //  You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);

// -------- YOUR CODE HERE  -------- //
// We give you a template of the first three loops for your convenience
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < N; i++) {
                // ORow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});
                std::vector<float> ORow = formatTensor(ORowTensor);

                // calculate one row of Q*K^T
                // dot product between Q_i and all K_j
                int j_idx = 0;
                while (j_idx < N) {
                    float row_dot_product = 0.0f;

                    int k_dim = 0;
                    while (k_dim < d) {
                        // get values from Q and K using indexing func
                        float q_element = Q[calc_4d_index(b, h, i, k_dim, H, N, d)];
                        float k_element = K[calc_4d_index(b, h, j_idx, k_dim, H, N, d)];

                        // sum of dot products
                        row_dot_product += q_element * k_element;
                        k_dim++;
                    }

                    // store the result in ORow
                    ORow[j_idx] = row_dot_product;
                    j_idx++;
                }

                // apply softmax
                float max_val = ORow[0];
                for (int m = 1; m < N; m++) {
                    if (ORow[m] > max_val) {
                        max_val = ORow[m];
                    }
                }

                // exponentials and sum
                float exponential_sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    float shifted_val = ORow[j] - max_val;
                    float exp_val = exp(shifted_val);
                    ORow[j] = exp_val;
                    exponential_sum += exp_val;
                }

                int norm_idx = 0;
                while (norm_idx < N) {
                    ORow[norm_idx] /= exponential_sum;
                    norm_idx++;
                }

                // get output by multiplying normalized attention weights with V
                for (int out_d = 0; out_d < d; out_d++) {
                    // start with zero accumulator
                    float output_val = 0.0f;

                    // weight sum across the sequence dimension
                    int seq_idx = 0;
                    while (seq_idx < N) {
                        // multiply attention weight by corresponding value element
                        float attention_weight = ORow[seq_idx];
                        float v_element = V[calc_4d_index(b, h, seq_idx, out_d, H, N, d)];

                        // accumulate weighted value
                        output_val += attention_weight * v_element;
                        seq_idx++;
                    }

                    // store the result in the output tensor
                    O[calc_4d_index(b, h, i, out_d, H, N, d)] = output_val;
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
                               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
                               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
                               torch::Tensor OiTensor, torch::Tensor LTensor, torch::Tensor LiTensor,
                               torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                               int B, int H, int N, int d) {
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    // Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    // Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //

    // func to access 4D tensor elements
    auto get_elem = [&](const std::vector<float> &tensor, int batch, int head, int row, int col) -> float {
        int idx = batch * (H * N * d) + head * (N * d) + row * d + col;
        return tensor[idx];
    };

    // func to set 4D tensor elements
    auto set_elem = [&](std::vector<float> &tensor, int batch, int head, int row, int col, float val) {
        int idx = batch * (H * N * d) + head * (N * d) + row * d + col;
        tensor[idx] = val;
    };

    // func to access 2D tensor elements
    auto get_2d = [](const std::vector<float> &tensor, int row, int col, int cols) -> float {
        return tensor[row * cols + col];
    };

    // func to set 2D tensor elements
    auto set_2d = [](std::vector<float> &tensor, int row, int col, int cols, float val) {
        tensor[row * cols + col] = val;
    };

    // temporary vector to help with accumulation
    std::vector<float> row_sums(Br, 0.0f);

    // process each batch
    int batch_idx = 0;
    while (batch_idx < B) {
        int head_idx = 0;
        while (head_idx < H) {
            for (int row_block = 0; row_block < N; row_block += Br) {
                // init accumulators
                for (int local_row = 0; local_row < Br; local_row++) {
                    // check if within bounds
                    int global_row = row_block + local_row;
                    if (global_row < N) {
                        // init li with -infinity for max finding
                        li[local_row] = -std::numeric_limits<float>::infinity();

                        // initialize lnew accumulator to zero
                        lnew[local_row] = 0.0f;

                        // initialize row sum to zero
                        row_sums[local_row] = 0.0f;

                        // initialize output accumulator for this row to zeros
                        int dim_idx = 0;
                        while (dim_idx < d) {
                            set_2d(Oi, local_row, dim_idx, d, 0.0f);
                            dim_idx++;
                        }
                    }
                }

                // process blocks of columns
                int col_block = 0;
                while (col_block < N) {
                    // adjust end point for last block that might be smaller
                    int col_block_end = std::min(col_block + Bc, N);
                    int actual_block_cols = col_block_end - col_block;

                    for (int local_row = 0; local_row < Br; local_row++) {
                        int global_row = row_block + local_row;
                        if (global_row < N) {
                            for (int dim = 0; dim < d; dim++) {
                                float val = get_elem(Q, batch_idx, head_idx, global_row, dim);
                                set_2d(Qi, local_row, dim, d, val);
                            }
                        }
                    }

                    // load Kj and Vj blocks from K and V
                    int local_col = 0;
                    while (local_col < actual_block_cols) {
                        int global_col = col_block + local_col;

                        // copy column data
                        for (int dim = 0; dim < d; dim++) {
                            float k_val = get_elem(K, batch_idx, head_idx, global_col, dim);
                            set_2d(Kj, local_col, dim, d, k_val);

                            float v_val = get_elem(V, batch_idx, head_idx, global_col, dim);
                            set_2d(Vj, local_col, dim, d, v_val);
                        }
                        local_col++;
                    }

                    // calc Sij = Qi * Kj^T (matrix multiplication)
                    for (int local_row = 0; local_row < Br; local_row++) {
                        int global_row = row_block + local_row;
                        if (global_row < N) {
                            for (int local_col = 0; local_col < actual_block_cols; local_col++) {
                                // compute dot product for this cell
                                float dot_prod = 0.0f;

                                for (int k = 0; k < d; k++) {
                                    float qi_val = get_2d(Qi, local_row, k, d);
                                    float kj_val = get_2d(Kj, local_col, k, d);
                                    dot_prod += qi_val * kj_val;
                                }

                                // store result in Sij
                                set_2d(Sij, local_row, local_col, Bc, dot_prod);
                            }
                        }
                    }

                    // find the max value for each row of Sij
                    for (int local_row = 0; local_row < Br; local_row++) {
                        int global_row = row_block + local_row;
                        if (global_row < N) {
                            float row_max = get_2d(Sij, local_row, 0, Bc);

                            for (int local_col = 1; local_col < actual_block_cols; local_col++) {
                                float val = get_2d(Sij, local_row, local_col, Bc);
                                if (val > row_max) {
                                    row_max = val;
                                }
                            }

                            // lij = max
                            lij[local_row] = row_max;
                        }
                    }

                    // Pij = exp(Sij - lij) and row sums
                    for (int local_row = 0; local_row < Br; local_row++) {
                        int global_row = row_block + local_row;
                        if (global_row < N) {
                            // get max value for this row
                            float max_val = lij[local_row];

                            // compute sum of exponentials for this row
                            float sum_exp = 0.0f;

                            // calculate exp(Sij - lij) for each element and sum
                            for (int local_col = 0; local_col < actual_block_cols; local_col++) {
                                // compute exp(Sij[i,j] - lij[i])
                                float s_val = get_2d(Sij, local_row, local_col, Bc);
                                float shifted = s_val - max_val;
                                float exp_val = std::exp(shifted);

                                // store in Pij
                                set_2d(Pij, local_row, local_col, Bc, exp_val);

                                // add to sum
                                sum_exp += exp_val;
                            }

                            // store row sum
                            row_sums[local_row] = sum_exp;
                        }
                    }

                    // compute PV = Pij * Vj
                    for (int local_row = 0; local_row < Br; local_row++) {
                        int global_row = row_block + local_row;
                        if (global_row < N) {
                            for (int dim = 0; dim < d; dim++) {
                                float pv_sum = 0.0f;

                                for (int local_col = 0; local_col < actual_block_cols; local_col++) {
                                    float p_val = get_2d(Pij, local_row, local_col, Bc);
                                    float v_val = get_2d(Vj, local_col, dim, d);
                                    pv_sum += p_val * v_val;
                                }

                                set_2d(PV, local_row, dim, d, pv_sum);
                            }
                        }
                    }

                    // update accumulators (Oi and lnew)
                    for (int local_row = 0; local_row < Br; local_row++) {
                        int global_row = row_block + local_row;
                        if (global_row < N) {
                            // get previous and current max values
                            float prev_max = li[local_row];
                            float curr_max = lij[local_row];

                            // compute scaling factor
                            float scale_factor = std::exp(prev_max - curr_max);

                            // update the normalizing factor
                            float old_lnew = lnew[local_row];
                            float new_lnew = scale_factor * old_lnew + row_sums[local_row];
                            lnew[local_row] = new_lnew;

                            // update output accumulators
                            for (int dim = 0; dim < d; dim++) {
                                float old_oi = get_2d(Oi, local_row, dim, d);
                                float pv_val = get_2d(PV, local_row, dim, d);

                                // scale and add
                                float new_oi = scale_factor * old_oi + pv_val;
                                set_2d(Oi, local_row, dim, d, new_oi);
                            }

                            // li = max
                            li[local_row] = curr_max;
                        }
                    }

                    col_block += Bc;
                }

                // normalize final outputs and write to O
                for (int local_row = 0; local_row < Br; local_row++) {
                    int global_row = row_block + local_row;
                    if (global_row < N) {
                        float norm_factor = lnew[local_row];

                        for (int dim = 0; dim < d; dim++) {
                            float oi_val = get_2d(Oi, local_row, dim, d);
                            float normalized = oi_val / norm_factor;

                            // write to output tensor
                            set_elem(O, batch_idx, head_idx, global_row, dim, normalized);
                        }
                    }
                }
            }
            head_idx++;
        }
        batch_idx++;
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
    m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
    m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
    m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
    m.def("twoDimRead", &twoDimRead, "twoDimRead");
    m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
