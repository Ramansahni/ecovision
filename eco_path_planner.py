import numpy as np
import matplotlib.pyplot as plt
import heapq
import rasterio
import os

# =========================================================
# 1️⃣ LOAD BANDS FROM EO FOLDER
# =========================================================

eo_folder = "EO"  # folder inside EcoVision

b2_path = os.path.join(eo_folder, "b2.tiff")
b3_path = os.path.join(eo_folder, "b3.tiff")
b4_path = os.path.join(eo_folder, "b4.tiff")
b8_path = os.path.join(eo_folder, "b8.tiff")

with rasterio.open(b2_path) as src:
    blue = src.read(1).astype(float)

with rasterio.open(b3_path) as src:
    green = src.read(1).astype(float)

with rasterio.open(b4_path) as src:
    red = src.read(1).astype(float)

with rasterio.open(b8_path) as src:
    nir = src.read(1).astype(float)

# Stack bands
image = np.stack([blue, green, red, nir], axis=-1)

# =========================================================
# 2️⃣ NDVI
# =========================================================

ndvi = (nir - red) / (nir + red + 1e-7)

# =========================================================
# 3️⃣ FVC (Fractional Vegetation Cover)
# =========================================================

ndvi_min = np.percentile(ndvi, 5)
ndvi_max = np.percentile(ndvi, 95)

fvc = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-7)) ** 2
fvc = np.clip(fvc, 0, 1)

# =========================================================
# 4️⃣ COST MAP
# =========================================================

cost_map = 1 + (fvc * 10)

# Penalize water
cost_map[ndvi < 0] = 100

# =========================================================
# 5️⃣ A* ALGORITHM
# =========================================================

def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def astar(cost_map, start, goal):
    rows, cols = cost_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for d in directions:
            neighbor = (current[0]+d[0], current[1]+d[1])

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:

                tentative_g = g_score[current] + cost_map[neighbor]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return None

# =========================================================
# 6️⃣ INTERACTIVE POINT SELECTION
# =========================================================

rgb = np.stack([blue, green, red], axis=-1)
rgb = rgb / np.max(rgb)

points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        y = int(event.ydata)
        x = int(event.xdata)
        points.append((y, x))
        plt.scatter(x, y, color='yellow', s=100)
        plt.draw()

        if len(points) == 2:
            plt.close()

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(rgb)
ax.set_title("Click START and END points")
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

if len(points) != 2:
    print("Select exactly 2 points.")
    exit()

start, goal = points
print("Start:", start)
print("Goal :", goal)

# =========================================================
# 7️⃣ RUN A*
# =========================================================

path = astar(cost_map, start, goal)

# =========================================================
# 8️⃣ VISUALIZE RESULT
# =========================================================

plt.figure(figsize=(8,8))
plt.imshow(rgb)

if path:
    path = np.array(path)
    plt.plot(path[:,1], path[:,0], color='cyan', linewidth=2)

plt.scatter(start[1], start[0], color='green', s=150, label="Start")
plt.scatter(goal[1], goal[0], color='red', s=150, label="Goal")

plt.legend()
plt.title("Eco-Optimal Path")
plt.show()

