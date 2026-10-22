extern "C" __global__ 
void grow(float* food_grid_updates, float* food_grid, int num_cells, int pad, float step_size, float max_food,
          const float* capacity_mult) {
    /* 
    Read from:
        food_grid_updates: [S, S], where the coordinate is the amount of food that has been lost to eating

    Write to:  
        food_grid: [S, S], the coordinate is the amount of food in the cell


    Constants:
        num_cells: number of cells in the grid. (width + 2*pad)
        pad: padding of the grid.
        step_size: increment to scale food growth by
        max_food: food level that cells grow towards (and decay down to when above it)
        capacity_mult: [S, S] per-cell multiplier on max_food, or null for a uniform max_food
*/
    // Get the index of the current thread
    int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_y = blockIdx.y * blockDim.y + threadIdx.y;
    // dont grow the food in the padding region
    if (cell_x < pad || cell_y < pad || cell_x >= num_cells - pad || cell_y >= num_cells - pad) {
        return;
    }

    int cell_idx = cell_y * num_cells + cell_x;
    float food = food_grid[cell_idx];
    food -= food_grid_updates[cell_idx];  // apply eating update

    float cap = capacity_mult ? max_food * capacity_mult[cell_idx] : max_food;
    float growth_amt = step_size * (cap - food);

    // grow the food
    if (food < 0.0){
        food += growth_amt*CFG_food_recovery_rate;
    }
    else if (food > cap){
        food += growth_amt*CFG_food_decay_rate;
    }
    else{
        food += growth_amt*CFG_food_growth_rate;
    }
    food_grid[cell_idx] = food;
}
