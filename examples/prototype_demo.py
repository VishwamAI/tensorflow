import sys
import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_models.base_model import BaseModel


def check_clustering_status(model):
    clustered = False
    try:
        for layer in model.layers:
            if hasattr(layer, 'kernel') and isinstance(layer.kernel, tfmot.clustering.keras.cluster.ClusterWeights):
                clustered = True
                break
    except AttributeError:
        pass
    return clustered


def main():
    # Create an instance of the BaseModel with individual MLU optimization flags
    model = BaseModel(
        advanced_math=True,
        mlu_optimization=True,
        apply_matrix_multiplication=True,
        apply_pca=True,
        apply_quantization=True,
        apply_pruning=True,
        apply_clustering=True,
    )

    # Build the model to ensure all layers are initialized
    model.build(input_shape=(None, 2))

    # Apply clustering to the model if enabled
    if model.apply_clustering:
        model = model.apply_clustering_to_model(model)

    # Check clustering status
    clustering_status = check_clustering_status(model)
    print("Clustering Applied:", clustering_status)

    # Example data for matrix multiplication
    matrix_a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    matrix_b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

    # Perform matrix multiplication
    result_matrix_multiplication = model.advanced_math_operations(matrix_a)
    print("Matrix Multiplication Result:")
    print(result_matrix_multiplication)

    # Example data for PCA
    data = tf.constant(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2, 1.6],
            [1, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ],
        dtype=tf.float32,
    )

    # Perform PCA
    result_pca = model.advanced_math_operations(data, n_components=1)
    print("PCA Result:")
    print(result_pca)

    # Example data for MLU optimizations
    input_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

    # Perform MLU optimizations
    result_mlu_optimizations = model.mlu_optimizations(
        input_tensor,
        quantization=model.apply_quantization,
        pruning=model.apply_pruning,
    )
    print("MLU Optimization (Quantization, Pruning) Result:")
    print(result_mlu_optimizations)


if __name__ == "__main__":
    main()
