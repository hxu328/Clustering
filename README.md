# Clustering
Using the publicly available Pokemon stats, you'll be performing clustering on these stats. Each Pokemon is defined by a row in the data set. Because there are various ways to characterize how strong a Pokemon is, it is often desirable to convert a raw stats sheet into a shorter feature vector. You will represent a Pokemon's strength by two numbers: "x" and "y". "x" will represent the Pokemon's total offensive strength, which is defined by Attack + Sp. Atk + Speed. Similarly, "y" will represent the Pokemon's total defensive strength, which is defined by Defense + Sp. Def + HP. After each Pokemon becomes that two-dimensional feature vector, you will cluster the first 20 Pokemon with hierarchical agglomerative clustering (HAC).

## Program Specification
1. load_data(filepath) — takes in a string with a path to a CSV file formatted as in the link above, and returns the first 20 data points (without the Generation and Legendary columns but retaining all other columns) in a single structure.
2. calculate_x_y(stats) — takes in one row from the data loaded from the previous function, calculates the corresponding x, y values for that Pokemon as specified above, and returns them in a single structure.
3. hac(dataset) — performs single linkage hierarchical agglomerative clustering on the Pokemon with the (x,y) feature representation, and returns a data structure representing the clustering.
4. random_x_y(m) — takes in the number of samples we want to randomly generate, and returns these samples in a single structure.
5. imshow_hac(dataset) — performs single linkage hierarchical agglomerative clustering on the Pokemon with the (x,y) feature representation, and imshow the clustering process.
