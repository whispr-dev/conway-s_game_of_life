import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.signal import convolve2d
from concurrent.futures import ThreadPoolExecutor

class LifeVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Conway's Game of Life")

        # Variables
        self.cell_size = 3  # Size of each cell in pixels
        self.width = 400  # Number of cells horizontally
        self.height = 400  # Number of cells vertically
        self.grid = np.random.choice([0, 1], (self.height, self.width), p=[0.8, 0.2])
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

        # Start button
        ttk.Button(
            control_frame,
            text="Start",
            command=self.start_simulation
        ).pack(side=tk.LEFT, padx=5)

        # Stop button
        ttk.Button(
            control_frame,
            text="Stop",
            command=self.stop_simulation
        ).pack(side=tk.LEFT, padx=5)

        # Randomize button
        ttk.Button(
            control_frame,
            text="Randomize",
            command=self.randomize_grid
        ).pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = ttk.Label(control_frame, text="Alive Cells: 0")
        self.status_label.pack(side=tk.LEFT, padx=5)

    def randomize_grid(self):
        """Randomize the initial grid."""
        self.grid = np.random.choice([0, 1], (self.height, self.width), p=[0.8, 0.2])
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

        self.grid = self.parallel_next_generation(self.grid)
        self.update_canvas()

        # Schedule next step
        self.root.after(100, self.run_simulation)

    def update_chunk(self, start, end, grid, kernel):
        """Update a chunk of the grid."""
        neighbor_count = convolve2d(grid[start:end], kernel, mode='same', boundary='wrap')
        return (neighbor_count == 3) | ((grid[start:end] == 1) & (neighbor_count == 2))

    def parallel_next_generation(self, grid, num_threads=4):
        """Divide grid processing across multiple threads."""
        chunk_size = grid.shape[0] // num_threads
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(num_threads):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < num_threads - 1 else grid.shape[0]
                results.append(executor.submit(self.update_chunk, start, end, grid, kernel))
        return np.vstack([r.result() for r in results])

    def update_canvas(self):
        """Update the canvas to reflect the current grid."""
        self.canvas.delete("all")
        alive_cells = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 1:
                    alive_cells += 1
                    self.draw_cell(x, y)

        self.status_label.config(text=f"Alive Cells: {alive_cells}")

    def draw_cell(self, x, y):
        """Draw a single cell on the canvas."""
        self.canvas.create_rectangle(
            x * self.cell_size,
            y * self.cell_size,
            (x + 1) * self.cell_size,
            (y + 1) * self.cell_size,
            fill='white',
            outline=''
        )

def main():
    root = tk.Tk()
    app = LifeVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
