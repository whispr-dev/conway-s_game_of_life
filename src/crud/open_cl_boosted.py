import pyopencl as cl
import numpy as np

# OpenCL kernel for Conway's Game of Life
kernel_code = """
__kernel void game_of_life(
    __global const int* grid,
    __global int* new_grid,
    const int width,
    const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Compute the linear index of this cell
    int idx = y * width + x;

    // Count alive neighbors
    int neighbors = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            neighbors += grid[ny * width + nx];
        }
    }

    // Apply the rules of Conway's Game of Life
    if (grid[idx] == 1) {
        new_grid[idx] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
    } else {
        new_grid[idx] = (neighbors == 3) ? 1 : 0;
    }
}
"""

# Set up OpenCL
ctx = cl.create_some_context()  # Use the integrated GPU
queue = cl.CommandQueue(ctx)

# Create OpenCL program
program = cl.Program(ctx, kernel_code).build()

def run_game_of_life(grid, num_iterations):
    """Run Conway's Game of Life using OpenCL."""
    width, height = grid.shape

    # Allocate buffers
    mf = cl.mem_flags
    grid_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=grid)
    new_grid_buf = cl.Buffer(ctx, mf.READ_WRITE, grid.nbytes)

    for _ in range(num_iterations):
        # Run the kernel
        program.game_of_life(queue, grid.shape, None, grid_buf, new_grid_buf, np.int32(width), np.int32(height))

        # Swap buffers
        grid_buf, new_grid_buf = new_grid_buf, grid_buf

    # Retrieve the result
    final_grid = np.empty_like(grid)
    cl.enqueue_copy(queue, final_grid, grid_buf)
    return final_grid

# Example usage
if __name__ == "__main__":
    # Initialize grid
    width, height = 350, 350
    grid = np.random.choice([0, 1], size=(height, width), p=[0.8, 0.2]).astype(np.int32)

    # Run simulation
    num_iterations = 5000
    result = run_game_of_life(grid, num_iterations)

    print(result)
