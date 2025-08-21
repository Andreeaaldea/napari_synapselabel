import sys
print("Python:", sys.executable)

try:
    from sklearn.mixture import GaussianMixture
    print(" scikit-learn is available.")
except ImportError:
    print("scikit-learn is NOT available.")