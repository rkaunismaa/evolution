extern "C" __global__
void setup_eat_grid(int* positions, int* food_types, float* eat_pcts, float* pct_eaten, int num_creatures, int num_cells) {
    /*
    Positions is [N, 2], where the 2 coordinates are the location (in cells)

    eat_pcts is [N], where the value is the percentage of the cell that the creature eats
     
     food_types is [N], the food grid each creature eats from (0 when there is one food type)

     pct_eaten is [T, S, S], where the coordinate is the percentage of the cell that creatures are trying to eat
    */
    int creature = blockIdx.x * blockDim.x + threadIdx.x;

    if (creature >= num_creatures) {
        return;
    }

    int x = positions[creature * 2 + 0];
    int y = positions[creature * 2 + 1];
    float eat_pct = eat_pcts[creature];

    int grid_idx = (food_types[creature] * num_cells + y) * num_cells + x;
    atomicAdd(&pct_eaten[grid_idx], eat_pct);
}
