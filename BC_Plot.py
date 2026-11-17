import csv
import matplotlib.pyplot as plt

filename = "BC_R.csv"

with open(filename, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise SystemExit("BC_R.csv is empty. Run the C program first.")

x = [int(row["Iteration"]) for row in rows]

columns = [
    ("A0 Sequential", "Sequential DBSCAN"),
    ("A1 OpenMP CSR", "RROpenMP DBSCAN"),
    ("A3 Grid OpenMP CSR", "TEGDBSCAN"),
    ("A4 LSH OpenMP CSR", "IP.LSH.DBSCAN"),
    ("A6 HPDBSCAN", "HPDBSCAN"),
    ("A5 ECR LSH", r"$\mathbf{\mathit{CGLSHP-DBSCAN}}$")
]

for column, label in columns:
    y = [float(row[column]) for row in rows]
    plt.plot(x, y, marker="o", linewidth=2, markersize=4, label=label)

plt.xlabel("Iteration")
plt.ylabel("Speedup")
plt.title("DBSCAN Methods: Iteration vs Speedup")

plt.xticks(x)
plt.grid(False)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=6,
    fontsize=7,
    frameon=False,
    handlelength=1.5,
    columnspacing=1.2
)

plt.tight_layout()
plt.savefig("BC.png", dpi=300, bbox_inches="tight")

print("Graph saved as BC.png")
