import plotext as plt
import numpy as np

# how to be a deans lister
data = (0.2248508781194687, 0.17783311009407043, 0.00813400186598301, 0.006144113838672638, 0.005904252640902996, 0.005167767871171236, 0.005007470492273569, 0.004975105170160532, 0.0048326170071959496, 0.004341860301792622, 0.004032623488456011, 0.0037661134265363216, 0.003682920942083001, 0.0032567158341407776, 0.002680206671357155, 0.0025268231984227896, 0.002321115229278803, 0.002164179692044854, 0.0018622403731569648, 0.0018245566170662642)
# Compute the average value
average_value = np.mean(data)

# Generate x-axis indices
x = list(range(len(data)))

# Clear previous plots
plt.clear_data()

# Enable dark mode
plt.theme("dark")

# Scatter plot for thick points with 'O' marker
plt.scatter(x, data, marker="◆", color="blue")

# Average line
plt.plot(x, [average_value] * len(data), color="red", marker="┉	")

# Customize the plot
plt.title("1D Array Plot with Average Line")
plt.xlabel("Index")
plt.ylabel("Value")
plt.ylim(0, max(data) * 1)  # Adjust y-axis limits for better visibility
plt.show()