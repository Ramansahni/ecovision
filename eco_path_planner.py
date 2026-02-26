import numpy as np
import matplotlib.pyplot as plt
import heapq
import rasterio
import os

# =========================================================
# 1️⃣ LOAD EO BANDS
# =========================================================

eo_folder = "EO"

b2_path = os.path.join(eo_folder, "b2.tiff")  # Blue
b3_path = os.path.join(eo_folder, "b3.tiff")  # Green
b4_path = os.path.join(eo_folder, "b4.tiff")  # Red
b8_path = os.path.join(eo_folder, "b8.tiff")  # NIR

with rasterio.open(b2_path) as src:
    blue = src.read(1).astype(float)

with rasterio.open(b3_path) as src:
    green = src.read(1).astype(float)

with rasterio.open(b4_path) as src:
    red = src.read(1).astype(float)

with rasterio.open(b8_path) as src:
    nir = src.read(1).astype(float)

print("Bands loaded successfully!")

# =========================================================
# 2️⃣ CREATE RGB MAP (FOR VISUALIZATION)
# =========================================================

rgb = np.stack([red, green, blue], axis=-1)

# Normalize for display
rgb = rgb / (np.percentile(rgb, 99) + 1e-7)
rgb = np.clip(rgb, 0, 1)

plt.figure(figsize=(8,8))
plt.imshow(rgb)
plt.title("RGB Satellite Image")
plt.axis("off")
plt.show()

# =========================================================
# 3️⃣ NDVI CALCULATION
# =========================================================

ndvi = (nir - red) / (nir + red + 1e-7)

plt.figure(figsize=(8,8))
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar(label="NDVI Value")
plt.title("NDVI Map")
plt.axis("off")
plt.show()

# =========================================================
# 4️⃣ FVC (Fractional Vegetation Cover)
# =========================================================

ndvi_min = np.percentile(ndvi, 5)
ndvi_max = np.percentile(ndvi, 95)

fvc = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-7)) ** 2
fvc = np.clip(fvc, 0, 1)

# =========================================================
# 5️⃣ COST MAP
# =========================================================

cost_map = 1 + (fvc * 15)

# Water penalty (NDVI < 0)
cost_map[ndvi < 0] = 100

# Very dense vegetation penalty
cost_map[fvc > 0.85] = 60

plt.figure(figsize=(8,8))
plt.imshow(cost_map, cmap="inferno")
plt.colorbar(label="Traversal Cost")
plt.title("Environmental Cost Map")
plt.axis("off")
plt.show()

# =========================================================
# 6️⃣ A* ALGORITHM
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
# 7️⃣ INTERACTIVE POINT SELECTION
# =========================================================

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
# 8️⃣ RUN A*
# =========================================================

path = astar(cost_map, start, goal)

# =========================================================
# 9️⃣ FINAL VISUALIZATION
# =========================================================

plt.figure(figsize=(8,8))
plt.imshow(rgb)

if path:
    path = np.array(path)
    plt.plot(path[:,1], path[:,0], color='red', linewidth=2)

plt.scatter(start[1], start[0], color='blue', s=150, label="Start")
plt.scatter(goal[1], goal[0], color='blue', s=150, label="Goal")

plt.legend()
plt.title("NDVI-Based Eco-Optimal Path")
plt.axis("off")
plt.show()