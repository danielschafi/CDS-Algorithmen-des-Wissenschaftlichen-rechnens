import matplotlib.pyplot as plt
import tensorly as tl
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tensorly.decomposition import tucker

faces = fetch_olivetti_faces()
x = faces.data
img = faces.images
y_true = faces.target


print(f"x: {x.shape}")
print(f"img: {img.shape}")
print(f"y_true: {y_true.shape}")


def show_some_images(n=4):
    for i in range(n):
        plt.imshow(img[i], cmap="gray")
        plt.show()


G, factors = tucker(img, (32, 32, 32))

print(f"G: {G.shape}")
print(f"factors: {[factor.shape for factor in factors]}")


X_rec = tl.tucker_to_tensor((G, factors))

x_train, x_test = X_rec[: int(len(X_rec) * 0.9)], y_true[int(len(y_true) * 0.9) :]
y_train, y_test = X_rec[: int(len(X_rec) * 0.9)], y_true[int(len(y_true) * 0.9) :]

for k in range(5, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    preds = knn.predict(X_rec[int(len(x) * 0.9) :].reshape(-1, 4096))

    acc = accuracy_score(y_test, preds)

    print(f"K: {k}, score: {acc:.4f}")
