from pyspark.ml.param.shared import HasFeaturesCol, HasOutputCol, HasLabelCol
from pyspark.ml.param import *
from pyspark.ml import Estimator, Model
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.linalg import DenseVector
from pyspark.sql import DataFrame
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from typing import Dict, List, Tuple
import heapq
from collections import defaultdict
from pyspark import RDD

class ReliefFSelector(Estimator, HasFeaturesCol, HasOutputCol, HasLabelCol,
                     DefaultParamsReadable, DefaultParamsWritable):
    
    numNeighbors = Param(Params._dummy(), "numNeighbors", 
                        "Number of neighbors (k) for ReliefF algorithm",
                        typeConverter=TypeConverters.toInt)
    
    sampleSize = Param(Params._dummy(), "sampleSize",
                      "Number of instances to sample (m)",
                      typeConverter=TypeConverters.toInt)
    
    contextualMerit = Param(Params._dummy(), "contextualMerit",
                           "Use Contextual Merit algorithm variant",
                           typeConverter=TypeConverters.toBoolean)
    
    selectionThreshold = Param(Params._dummy(), "selectionThreshold",
                             "Proportion of features to select (0,1]",
                             typeConverter=TypeConverters.toFloat)
    
    useCache = Param(Params._dummy(), "useCache",
                     "Cache dataset in memory",
                     typeConverter=TypeConverters.toBoolean)
    
    useRamp = Param(Params._dummy(), "useRamp",
                    "Use ramp function for numeric differences",
                    typeConverter=TypeConverters.toBoolean)

    def __init__(self, featuresCol="features", labelCol="label", outputCol="selectedFeatures"):
        super(ReliefFSelector, self).__init__()
        self._setDefault(numNeighbors=10,
                        sampleSize=0,
                        contextualMerit=False,
                        selectionThreshold=0.15,
                        useCache=False,
                        useRamp=False)
        self.set(self.featuresCol, featuresCol)
        self.set(self.labelCol, labelCol)
        self.set(self.outputCol, outputCol)

    def setFeaturesCol(self, value):
        """Set features column name"""
        return self.set(self.featuresCol, value)

    def getFeaturesCol(self):
        """Get features column name"""
        return self.getOrDefault(self.featuresCol)

    def setLabelCol(self, value):
        """Set label column name"""
        return self.set(self.labelCol, value)

    def getLabelCol(self):
        """Get label column name"""
        return self.getOrDefault(self.labelCol)

    def setOutputCol(self, value):
        """Set output column name"""
        return self.set(self.outputCol, value)

    def getOutputCol(self):
        """Get output column name"""
        return self.getOrDefault(self.outputCol)

    def setNumNeighbors(self, value):
        return self._set(numNeighbors=value)

    def setSampleSize(self, value):
        return self._set(sampleSize=value)

    def setContextualMerit(self, value):
        return self._set(contextualMerit=value)

    def setSelectionThreshold(self, value):
        return self._set(selectionThreshold=value)

    def setUseCache(self, value):
        return self._set(useCache=value)

    def setUseRamp(self, value):
        return self._set(useRamp=value)

    def getNumNeighbors(self):
        """Get number of neighbors parameter value"""
        return self.getOrDefault(self.numNeighbors)

    def getSampleSize(self):
        """Get sample size parameter value"""
        return self.getOrDefault(self.sampleSize)

    def getContextualMerit(self):
        """Get contextual merit parameter value"""
        return self.getOrDefault(self.contextualMerit)

    def getUseCache(self):
        """Get use cache parameter value"""
        return self.getOrDefault(self.useCache)

    def getUseRamp(self):
        """Get use ramp parameter value"""
        return self.getOrDefault(self.useRamp)

    def getSelectionThreshold(self):
        """Get selection threshold parameter value"""
        return self.getOrDefault(self.selectionThreshold)

    def _fit(self, dataset: DataFrame) -> "ReliefFSelectorModel":
        features_col = self.getFeaturesCol()
        label_col = self.getLabelCol()
        num_neighbors = self.getNumNeighbors()
        sample_size = self.getSampleSize()
        use_ramp = self.getUseRamp()
        contextual_merit = self.getContextualMerit()
        selection_threshold = self.getSelectionThreshold()

        # Convert DataFrame to RDD without additional repartitioning
        data_rdd: RDD = dataset.select(features_col, label_col).rdd.map(
            lambda row: (np.array(row[features_col].toArray(), dtype=np.float32), float(row[label_col]))
        )

        # Compute number of instances and features
        first_row = data_rdd.first()
        num_features = len(first_row[0])
        num_instances = data_rdd.countApprox(timeout=1000)

        # Calculate class priors
        label_counts = data_rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
        priors = {label: count / num_instances for label, count in label_counts.items()}

        # Compute min/max values in one pass
        min_max_rdd = data_rdd.map(lambda x: x[0])
        min_values, max_values = min_max_rdd.aggregate(
            (np.full(num_features, np.inf, dtype=np.float32), np.full(num_features, -np.inf, dtype=np.float32)),
            lambda acc, x: (np.minimum(acc[0], x), np.maximum(acc[1], x)),
            lambda acc1, acc2: (np.minimum(acc1[0], acc2[0]), np.maximum(acc1[1], acc2[1]))
        )

        denominators = max_values - min_values
        denominators[denominators == 0] = 1.0

        def diff(feat_idx: int, val1: float, val2: float) -> float:
            if denominators[feat_idx] == 1.0:
                return 0.0
            if use_ramp:
                max_val = max_values[feat_idx]
                min_val = min_values[feat_idx]
                t_equ = 0.05 * (max_val - min_val)
                t_dif = 0.10 * (max_val - min_val)
                dist = abs(val1 - val2)
                if dist <= t_equ:
                    return 0.0
                elif dist > t_dif:
                    return 1.0
                else:
                    return (dist - t_equ) / (t_dif - t_equ)
            else:
                return abs(val1 - val2) / denominators[feat_idx]

        def distance(features1: np.ndarray, features2: np.ndarray) -> float:
            return np.sum(np.fromiter(
                (diff(i, features1[i], features2[i]) for i in range(num_features)), 
                dtype=np.float32
            ))

        # Sample and broadcast
        num_samples = min(sample_size or num_instances, 500)
        sampled_data = data_rdd.takeSample(False, num_samples, seed=42)
        broadcast_samples = data_rdd.context.broadcast(
            [(feats.astype(np.float32), lbl) for feats, lbl in sampled_data]
        )

        def compute_weight_contribution(partition):
            weights = np.zeros(num_features, dtype=np.float32)
            instance_count = 0
            distance_counter = 0  # Unique counter to break ties in heap
            
            for instance_features, instance_label in partition:
                instance_count += 1
                
                # Compute distances to sampled instances
                distances = [
                    (distance(instance_features, sample_features), sample_label, sample_features)
                    for sample_features, sample_label in broadcast_samples.value
                ]

                # Separate hits and misses
                hits = [(d, feats) for d, lbl, feats in distances if lbl == instance_label]
                misses = [(d, lbl, feats) for d, lbl, feats in distances if lbl != instance_label]

                # Get k nearest hits
                k = num_neighbors
                nearest_hits = heapq.nsmallest(k, hits, key=lambda x: x[0]) if hits else []

                # Get k nearest misses per class with a tiebreaker
                nearest_misses_by_class = {}
                for dist, lbl, feats in misses:
                    if lbl not in nearest_misses_by_class:
                        nearest_misses_by_class[lbl] = []
                    # Add a unique counter to avoid comparing feats
                    heapq.heappush(nearest_misses_by_class[lbl], (dist, distance_counter, feats))
                    distance_counter += 1  # Increment counter for uniqueness
                    if len(nearest_misses_by_class[lbl]) > k:
                        heapq.heappop(nearest_misses_by_class[lbl])

                # Update weights for hits
                if not contextual_merit and len(nearest_hits) > 0:
                    hit_factor = 1.0 / (k * num_samples)
                    for _, sample_features in nearest_hits:
                        diff_values = np.fromiter(
                            (diff(i, instance_features[i], sample_features[i]) for i in range(num_features)),
                            dtype=np.float32
                        )
                        weights -= diff_values * hit_factor

                # Update weights for misses
                for miss_label, miss_list in nearest_misses_by_class.items():
                    if len(miss_list) > 0:
                        prior_factor = priors[miss_label] / (1.0 - priors[instance_label]) / (k * num_samples)
                        for dist, counter, sample_features in miss_list:  # Unpack with counter
                            diff_values = np.fromiter(
                                (diff(i, instance_features[i], sample_features[i]) for i in range(num_features)),
                                dtype=np.float32
                            )
                            weights += diff_values * prior_factor

            if instance_count > 0:
                yield (weights, instance_count)
                
        # Process and aggregate using existing partitions
        weights_rdd = data_rdd.mapPartitions(compute_weight_contribution)
        total_weights, total_instance_count = weights_rdd.reduce(
            lambda a, b: (a[0] + b[0], a[1] + b[1])
        )

        # Normalize weights
        if total_instance_count > 0:
            total_weights /= total_instance_count

        # Clean up resources
        if not self.getUseCache():
            data_rdd.unpersist(blocking=True)
            min_max_rdd.unpersist(blocking=True)
        weights_rdd.unpersist(blocking=True)
        broadcast_samples.unpersist()

        return ReliefFSelectorModel(total_weights, selection_threshold, features_col, self.getOutputCol())

class ReliefFSelectorModel(Model, HasFeaturesCol, HasOutputCol,
                          DefaultParamsReadable, DefaultParamsWritable):
    
    def __init__(self, weights, threshold, featuresCol="features", 
                 outputCol="selectedFeatures"):
        super(ReliefFSelectorModel, self).__init__()
        self.weights = weights
        self.threshold = threshold
        self.set(self.featuresCol, featuresCol)
        self.set(self.outputCol, outputCol)

    def setFeaturesCol(self, value):
        """Set features column name"""
        return self.set(self.featuresCol, value)

    def getFeaturesCol(self):
        """Get features column name"""
        return self.getOrDefault(self.featuresCol)

    def setOutputCol(self, value):
        """Set output column name"""
        return self.set(self.outputCol, value)

    def getOutputCol(self):
        """Get output column name"""
        return self.getOrDefault(self.outputCol)
    
    def _transform(self, dataset: DataFrame):
        # Select top features based on weights
        n_select = int(len(self.weights) * self.threshold)
        selected = np.argsort(self.weights)[-n_select:]
        
        # Use VectorSlicer to select features
        slicer = VectorSlicer(inputCol=self.getFeaturesCol(),
                             outputCol=self.getOutputCol(),
                             indices=[int(i) for i in selected])
        
        return slicer.transform(dataset)
