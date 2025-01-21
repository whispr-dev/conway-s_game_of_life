import tkinter as tk
from tkinter import ttk
import numpy as np
from numba import jit

@jit(nopython=True, parallel=True)
def compute_next_generation(grid, kernel):
    """Numba-accelerated grid computation."""
    height, width = grid.shape
    neighbor_count = np.zeros_like(grid)

    # Compute neighbors for each cell
    for y in range(height):
        for x in range(width):
            # Use periodic boundary conditions (wrapping)
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    neighbors += grid[(y + dy) % height, (x + dx) % width]
            neighbor_count[y, x] = neighbors

    # Apply Conway's rules
    next_grid = (neighbor_count == 3) | ((grid == 1) & (neighbor_count == 2))
    return next_grid.astype(np.uint8), neighbor_count

class LifeVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Conway's Game of Life")

        # Variables
        self.cell_size = 4
        self.width = 150
        self.height = 150
        self.grid = np.random.choice([0, 1], (self.height, self.width), p=[0.9, 0.1]).astype(np.uint8)
        self.neighbor_counts = np.zeros_like(self.grid)
        self.running = False

        # Create controls
        self.create_controls()

        # Create canvas for grid display
        self.canvas = tk.Canvas(
            root,
            width=self.width * self.cell_size,
            height=self.height * self.cell_size,
            bg='black'
        )
        self.canvas.pack(padx=10, pady=10)

        self.update_canvas()

    def create_controls(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(padx=10, pady=5)

        ttk.Button(control_frame, text="Start", command=self.start_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop", command=self.stop_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Randomize", command=self.randomize_grid).pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(control_frame, text="Alive Cells: 0")
        self.status_label.pack(side=tk.LEFT, padx=5)

    def randomize_grid(self):
        """Randomize the initial grid."""
        self.grid = np.random.choice([0, 1], (self.height, self.width), p=[0.9, 0.1]).astype(np.uint8)
        self.neighbor_counts = np.zeros_like(self.grid)
        self.update_canvas()

    def start_simulation(self):
        """Start the Game of Life simulation."""
        self.running = True
        self.run_simulation()

    def stop_simulation(self):
        """Stop the Game of Life simulation."""
        self.running = False

    def run_simulation(self):
        """Run the simulation step by step."""
        if not self.running:
            return

        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.grid, self.neighbor_counts = compute_next_generation(self.grid, kernel)
        self.update_canvas()
        self.root.after(100, self.run_simulation)

    def update_canvas(self):
        """Update the canvas to reflect the current grid."""
        self.canvas.delete("all")
        alive_cells = np.sum(self.grid)

        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 1:
                    neighbors = self.neighbor_counts[y, x]
                    if neighbors < 0 or neighbors > 8:
                        print(f"Invalid neighbor count at ({x}, {y}): {neighbors}")
                    self.draw_cell(x, y, neighbors)

        self.status_label.config(text=f"Alive Cells: {alive_cells}")

    def interpolate_color(self, base_color, weight):
        """Interpolate between a base color and white."""
        def clamp(val): return max(0, min(int(val), 255))
        r, g, b = base_color
        r = clamp(r + (255 - r) * weight)
        g = clamp(g + (255 - g) * weight)
        b = clamp(b + (255 - b) * weight)
        return f'#{r:02x}{g:02x}{b:02x}'

    def draw_cell(self, x, y, alive_neighbors):
        """Smooth the coloring."""
        if alive_neighbors > 5:
            base_color = (0, 0, 255)  # Blue
        elif alive_neighbors > 3:
            base_color = (255, 255, 255)  # White
        elif alive_neighbors > 1:
            base_color = (0, 255, 0)  # Green
        elif alive_neighbors == 1:
            base_color = (255, 255, 0)  # Yellow
        else:
            base_color = (255, 0, 0)  # Red

        # Normalize weight (number of neighbors divided by max possible: 8)
        weight = alive_neighbors / 8.0
        color = self.interpolate_color(base_color, weight)

        self.canvas.create_rectangle(
            x * self.cell_size,
            y * self.cell_size,
            (x + 1) * self.cell_size,
            (y + 1) * self.cell_size,
            fill=color,
            outline=''
        )

def main():
    root = tk.Tk()
    app = LifeVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
