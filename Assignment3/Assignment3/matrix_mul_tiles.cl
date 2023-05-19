__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, int m, int n, int k, int n_works)
{
	int TS = get_local_size(0);

	for(int offset_x = 0; offset_x < k; offset_x += get_global_size(0)){
		for(int offset_y = 0; offset_y < k; offset_y += get_global_size(1)){
			int l_idx = get_local_id(0); // local_col
			int l_idy = get_local_id(1); // local_row

			int g_idx = get_group_id(0) * TS + l_idx + offset_x; // global_col
			int g_idy = get_group_id(1) * TS + l_idy + offset_y; // global_row

			__local float Asub[16][16];
			__local float Bsub[16][16]; 

			float res = 0.0;
			int num_of_tiles = get_global_size(0) / TS + (k % TS == 0 ? 0 : 1);

			for(int i = 0; i < num_of_tiles * (k / get_global_size(0) + (k % get_global_size(0) == 0 ? 0 : 1)); i++){
				int t_idx = i * TS + l_idx; // tiled_col
				int t_idy = i * TS + l_idy; // tiled_row


				if(t_idx < k && g_idy < m){
					Asub[l_idy][l_idx] = A[g_idy * k + t_idx];
				} else {
					Asub[l_idy][l_idx] = 0;
				}
				
				if(t_idy < k && g_idx < n) {
					Bsub[l_idy][l_idx] = B[t_idy * n + g_idx];
				} else if(g_idx < n) {
					Bsub[l_idy][l_idx] = 0;
				}

				barrier(CLK_LOCAL_MEM_FENCE);

				for(int j = 0; j < TS; j++)
					res += Asub[l_idy][j] * Bsub[j][l_idx];
				
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			
			if(g_idx < n && g_idy < m){
				C[g_idy * n + g_idx] = res;
			}
		}
	}
}
