# a very lightweight implementation of Conway's Game of Life
# if it still runs a lil heavy, try changing variables in lines 15/16/17
# to 8/75/50
# when it runs a window will open with a lil interface
# start/stop/randomise buttons and living cell counter - self explanatory
# look out for Gosper's glider guns creating gliders, puffer-type breeders,
# etc. - see the wiki article for a vcomprehensive list and animation examples:
# https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life#Examples_of_patterns
# maybe you'll see a rteal rare one or even something new!
# this is the beginning of my work on a CA based Eurorack module - wofl, 2025.
# 
# [350x350 @ 4px is about the limits for smooth running oin this lil surface]
# gonna try some risc-v etc. optimizations now...

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
        self.cell_size = 4  # Size of each cell in pixels
        self.width = 350  # Number of cells horizontally
        self.height = 350  # Number of cells vertically
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
                    alive_neighbors = self.count_alive_neighbors(x, y)
                    self.draw_cell(x, y, alive_neighbors)

        self.status_label.config(text=f"Alive Cells: {alive_cells}")

    def count_alive_neighbors(self, x, y):
        """Count the number of alive neighbors for a cell."""
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                count += self.grid[ny, nx]
        return count

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

        weight = alive_neighbors / 8  # Normalize weight
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
